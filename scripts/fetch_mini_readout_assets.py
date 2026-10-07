"""Fetch only pinned mini-batch readout assets (~4.9 GB), not Qwen weights.

Run on an internet-connected login node. Existing files must match the known
checksums; partial downloads are never used as complete files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import struct
import urllib.request

MODEL_REV = '9216db5781bf21249d130ec9da846c4624c16137'
LENS_REV = 'b25d72a96b79c8e309d6625955a98751da47e67a'
LENS_FILE = 'qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt'
LENS_SHA = '0af5ccf2e1c251bedc202607a7ca214f3b9f60980bab73d67a3d0b887db213d8'
TOKENIZER_SHA = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
# Filled from the verified pinned model byte-range downloads used locally.
TENSOR_SHA = {'lm_head.weight': '3a26fe1453afb031b4557449b25f99dc4dcffb5b93c3b6f8fac1451b72f9a9af', 'model.norm.weight': '2dc809bebe6ec8603ac5cd8643e33bec937c69faa4edb7749cbf71b81c955e04'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def request(url, start=None, end=None):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    headers = {} if start is None else {'Range': f'bytes={start}-{end}'}
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180, context=context)
    if start is not None and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/')):
        response.close()
        raise ValueError('Server did not honor requested byte range')
    return response


def download(url, target, checksum, start=None, end=None):
    if target.exists():
        if digest(target) != checksum:
            raise ValueError(f'Existing asset differs: {target}; select a fresh --data directory')
        print('Verified', target, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    with request(url, start, end) as response, partial.open('wb') as handle:
        while block := response.read(8 * 1024**2):
            handle.write(block)
    if digest(partial) != checksum:
        raise ValueError(f'Download checksum mismatch: {partial}')
    partial.replace(target)
    print('Downloaded and verified', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('runtime/models/j-lens/qwen3-32b'))
    args = parser.parse_args()
    root = args.data
    base = f'https://huggingface.co/Qwen/Qwen3-32B/resolve/{MODEL_REV}/'
    download(base + 'tokenizer.json', root / 'tokenizer.json', TOKENIZER_SHA)
    shard = base + 'model-00017-of-00017.safetensors'
    with request(shard, 0, 7) as response:
        n = struct.unpack('<Q', response.read())[0]
    with request(shard, 8, 7 + n) as response:
        header = json.load(response)
    meta = {}
    for name in ('lm_head.weight', 'model.norm.weight'):
        info = header[name]
        if info['dtype'] != 'BF16':
            raise ValueError(f'Unexpected dtype for {name}')
        a, b = info['data_offsets']
        target = root / 'unembed' / (name + '.bf16.bin')
        download(shard, target, TENSOR_SHA[name], 8 + n + a, 8 + n + b - 1)
        meta[name] = {'shape': info['shape'], 'dtype': 'bfloat16', 'file': target.name}
    (root / 'unembed/meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    download(f'https://huggingface.co/neuronpedia/jacobian-lens/resolve/{LENS_REV}/{LENS_FILE}', root / LENS_FILE, LENS_SHA)
    (root / 'download-provenance.json').write_text(json.dumps({
        'model_revision': MODEL_REV, 'lens_revision': LENS_REV,
        'tokenizer_sha256': TOKENIZER_SHA}, indent=2) + '\n')


if __name__ == '__main__':
    main()
