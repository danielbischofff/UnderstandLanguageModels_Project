"""Inspect a torch ``.pt`` archive's metadata without reading its tensor data.

``torch.save`` writes an uncompressed (ZIP64) archive: one pickle, ``data.pkl``,
describing the object graph, plus one raw blob per tensor storage. Replaying
that pickle with a stubbed unpickler reveals the dtypes, shapes and keys of a
multi-gigabyte checkpoint from its first few kilobytes -- from a local path or,
via HTTP range requests, straight from the Hub.

Usage::

    python src/inspect_lens.py [PATH_OR_URL]
"""

from __future__ import annotations

import io
import pickle
import struct
import sys
import urllib.request

LENS_URL = (
    "https://huggingface.co/neuronpedia/jacobian-lens/resolve/main/"
    "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
)
_PROBE_BYTES = 1 << 20


def read_head(src: str, n: int = _PROBE_BYTES) -> bytes:
    """Return the first ``n`` bytes of a local file or an HTTP resource."""
    if not src.startswith("http"):
        with open(src, "rb") as fh:
            return fh.read(n)
    req = urllib.request.Request(
        src, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "curl/8"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def zip_entries(buf: bytes) -> list[tuple[str, int, int]]:
    """Scan local file headers in ``buf``.

    Returns ``(name, data_start, data_end)`` per entry. ZIP64 archives leave the
    local-header sizes zeroed, so each entry's extent is taken from the start of
    the next header; trailing alignment padding is harmless for both consumers
    below (pickle stops at its STOP opcode, tensor blobs are read by shape).
    """
    found: list[tuple[str, int, int]] = []
    i = 0
    while (i := buf.find(b"PK\x03\x04", i)) >= 0 and i + 30 <= len(buf):
        name_len, extra_len = struct.unpack("<HH", buf[i + 26 : i + 30])
        name_bytes = buf[i + 30 : i + 30 + name_len]
        i += 4
        # Guard against the signature occurring by chance inside tensor data.
        if not (0 < name_len < 512) or not name_bytes.isascii():
            continue
        found.append((name_bytes.decode(), i - 4, i - 4 + 30 + name_len + extra_len))

    return [
        (name, start, found[j + 1][1] if j + 1 < len(found) else len(buf))
        for j, (name, _hdr, start) in enumerate(found)
    ]


class _Storage:
    """Stand-in for a torch storage recorded during the stubbed unpickle."""

    def __init__(self, dtype: str, key: str, numel: int) -> None:
        self.dtype, self.key, self.numel = dtype, key, numel


def _rebuild_tensor(storage: _Storage, offset: int, size, stride, *_a, **_k) -> dict:
    return {
        "dtype": storage.dtype.replace("Storage", "").split(".")[-1],
        "key": storage.key,
        "shape": list(size),
        "offset": offset,
        "numel": storage.numel,
    }


class _StubUnpickler(pickle.Unpickler):
    """Rebuilds the object graph with tensors reduced to plain descriptors."""

    def find_class(self, module: str, name: str):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            return dict
        return lambda *a, **k: None

    def persistent_load(self, pid):
        _tag, storage_type, key, _location, numel = pid
        return _Storage(str(storage_type), str(key), int(numel))


def describe(src: str) -> dict:
    """Unpickle the metadata of the ``.pt`` archive at ``src``."""
    head = read_head(src)
    entries = zip_entries(head)
    pkl = next((e for e in entries if e[0].endswith("data.pkl")), None)
    if pkl is None:
        raise ValueError("data.pkl not found in the leading bytes of the archive")
    return _StubUnpickler(io.BytesIO(head[pkl[1] : pkl[2]])).load()


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else LENS_URL
    obj = describe(src)
    print(f"source: {src}\ntop-level keys: {sorted(obj)}\n")
    for key, value in obj.items():
        if isinstance(value, dict) and value and all(
            isinstance(v, dict) and "shape" in v for v in value.values()
        ):
            layers = sorted(value)
            first = value[layers[0]]
            storages = {value[l]["key"] for l in layers}
            print(
                f"{key}: {len(layers)} tensors, indices {layers[0]}..{layers[-1]} "
                f"(contiguous={layers == list(range(layers[0], layers[-1] + 1))})\n"
                f"    shape={first['shape']} dtype={first['dtype']}\n"
                f"    distinct storages={len(storages)}"
            )
        elif isinstance(value, list):
            print(f"{key}: list[{len(value)}] {value[:3]}..{value[-3:]}")
        else:
            print(f"{key}: {value!r}")


if __name__ == "__main__":
    main()
