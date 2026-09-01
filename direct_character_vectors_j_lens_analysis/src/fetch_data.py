"""Fetch every input the analysis needs into ``data/``.

Four sources, all skipped when already present:

* **Role vectors** -- ``lu-christina/assistant-axis-vectors``: per-layer mean
  residual activations, ``[64, 5120]`` bf16, one file per persona.
* **Jacobian lens** -- ``neuronpedia/jacobian-lens``: the fitted lens for
  Qwen3-32B (~6.6 GB).
* **Unembedding** -- ``Qwen/Qwen3-32B``: only ``lm_head.weight`` and
  ``model.norm.weight``. These live in the last shard, which is 3.0 GB; we
  parse the safetensors header and range-request just the two tensors we need
  (~1.56 GB), rather than pulling the whole shard.
* **Tokenizer** -- ``Qwen/Qwen3-32B``: ``tokenizer.json``, to decode token ids.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(
    os.environ.get("JLENS_DATA_ROOT", Path(__file__).resolve().parent.parent / "data")
)

AXIS_REPO = "lu-christina/assistant-axis-vectors"
AXIS_PREFIX = "qwen-3-32b"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
MODEL_REPO = "Qwen/Qwen3-32B"
UNEMBED_SHARD = "model-00017-of-00017.safetensors"
UNEMBED_TENSORS = ("lm_head.weight", "model.norm.weight")

#: Persona vectors to pull. ``default_vector`` is the Assistant baseline and
#: ``assistant_axis`` the published axis; both act as controls for the role.
VECTORS = ("role_vectors/bard.pt", "assistant_axis.pt", "default_vector.pt")

_CHUNK = 1 << 22
#: Attempts per request before giving up, with exponential backoff.
_RETRIES = 6


def _url(repo: str, filename: str, *, dataset: bool = False) -> str:
    kind = "datasets/" if dataset else ""
    return f"https://huggingface.co/{kind}{repo}/resolve/main/{filename}"


def _token() -> str | None:
    """The Hub token, from the environment or the standard token file.

    Anonymous requests are rate-limited hard enough that concurrent multi-GB
    fetches start returning 503; a read token avoids that. Never put it in a
    URL -- it goes in the Authorization header only.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _get(url: str, start: int | None = None, end: int | None = None):
    """Open ``url`` (optionally one byte range), retrying transient failures.

    The Hub answers 503 under load and rate-limits unauthenticated clients, so
    a bare ``urlopen`` does not survive a multi-gigabyte fetch running
    alongside anything else. Only 5xx, 429 and timeouts are retried; a 404 or
    401 is a real error and raises immediately.
    """
    headers = {"User-Agent": "curl/8"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(_RETRIES):
        try:
            return urllib.request.urlopen(request, timeout=180)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            fatal = code is not None and code < 500 and code != 429
            if fatal or attempt == _RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("unreachable")


def _download(url: str, dest: Path, start: int = 0, end: int | None = None,
              *, quiet: bool = False) -> Path:
    """Stream ``url`` (optionally one byte range) to ``dest``, resuming if partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = None if end is None else end - start + 1

    # A dropped connection ends the read loop without raising, which silently
    # leaves a truncated file. When the expected length is known, verify it and
    # resume from wherever the last attempt stopped.
    for attempt in range(_RETRIES):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have == total:
            if not quiet and attempt == 0:
                print(f"  have {dest.name}")
            return dest
        if total is not None and have > total:  # stale or corrupt; start over
            have = 0
        if total is None and have:  # size unknown; re-fetch rather than guess
            have = 0

        with _get(url, start + have if total is not None else None, end) as resp, \
                open(dest, "ab" if have else "wb") as fh:
            done = have
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if not quiet and done % (1 << 28) < _CHUNK:
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  {dest.name}: {done / 1e9:.2f} GB{pct}", flush=True)

        if total is None or dest.stat().st_size == total:
            break
        if not quiet:
            print(f"  {dest.name}: short read, resuming "
                  f"({dest.stat().st_size / 1e6:.0f}/{total / 1e6:.0f} MB)", flush=True)
    else:
        raise IOError(f"{dest.name}: incomplete after {_RETRIES} attempts")

    if not quiet:
        print(f"  wrote {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def safetensors_header(url: str) -> tuple[dict, int]:
    """Return the safetensors header dict and the byte offset of its data section."""
    with _get(url, 0, 7) as resp:
        header_len = struct.unpack("<Q", resp.read(8))[0]
    with _get(url, 8, 8 + header_len - 1) as resp:
        return json.loads(resp.read()), 8 + header_len


def fetch_role(role: str, *, quiet: bool = False) -> Path:
    """Download one role vector on demand, returning its local path.

    The dataset ships 275 roles; ``run_analysis --role <name>`` routes through
    here so only the one being analysed is fetched.
    """
    dest = DATA / "vectors" / f"{role}.pt"
    if not dest.exists():
        if not quiet:
            print(f"fetching role vector {role!r}:")
        _download(_url(AXIS_REPO, f"{AXIS_PREFIX}/role_vectors/{role}.pt",
                       dataset=True), dest, quiet=quiet)
    return dest


def list_roles() -> list[str]:
    """Every role name published for this model, from the Hub tree listing."""
    url = (f"https://huggingface.co/api/datasets/{AXIS_REPO}/tree/main/"
           f"{AXIS_PREFIX}/role_vectors?limit=1000")
    with _get(url) as resp:
        tree = json.load(resp)
    return sorted(Path(e["path"]).stem for e in tree if e["path"].endswith(".pt"))


def fetch_all_roles() -> list[str]:
    """Download every role vector (~180 MB); used by the whole-cast sweep."""
    roles = list_roles()
    print(f"role vectors: {len(roles)} roles")
    for i, role in enumerate(roles, 1):
        fetch_role(role, quiet=True)
        if i % 50 == 0 or i == len(roles):
            print(f"  {i}/{len(roles)}", flush=True)
    return roles


def fetch_vectors() -> None:
    print("role vectors:")
    for name in VECTORS:
        _download(_url(AXIS_REPO, f"{AXIS_PREFIX}/{name}", dataset=True),
                  DATA / "vectors" / Path(name).name)


def fetch_tokenizer() -> None:
    print("tokenizer:")
    _download(_url(MODEL_REPO, "tokenizer.json"), DATA / "tokenizer.json")


def fetch_lens() -> None:
    print("jacobian lens (~6.6 GB):")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LENS_REPO, LENS_FILE, local_dir=DATA)
    print(f"  {path} ({Path(path).stat().st_size / 1e9:.2f} GB)")


def fetch_unembedding() -> None:
    """Range-fetch ``lm_head.weight`` and the final RMSNorm gain as raw bf16."""
    print("unembedding (range-fetched from the last shard):")
    url = _url(MODEL_REPO, UNEMBED_SHARD)
    header, data_start = safetensors_header(url)
    meta = {}
    for tensor in UNEMBED_TENSORS:
        entry = header[tensor]
        lo, hi = entry["data_offsets"]
        assert entry["dtype"] == "BF16", entry["dtype"]
        dest = DATA / "unembed" / f"{tensor}.bf16.bin"
        _download(url, dest, data_start + lo, data_start + hi - 1)
        meta[tensor] = {"shape": entry["shape"], "dtype": "bfloat16",
                        "file": dest.name}
    (DATA / "unembed" / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fetch_vectors()
    fetch_tokenizer()
    fetch_unembedding()
    fetch_lens()
    print("\nall inputs present in", DATA)


if __name__ == "__main__":
    main()
