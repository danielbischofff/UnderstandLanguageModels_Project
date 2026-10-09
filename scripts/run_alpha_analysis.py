#!/usr/bin/env python3
"""Run the alpha readout and dashboards directly on a workstation (no Slurm)."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=root/'runtime/models/j-lens/qwen3-32b')
    parser.add_argument('--readouts',type=Path,default=root/'outputs/analysis/mini-alpha-readout')
    parser.add_argument('--reports',type=Path,default=root/'results/mini-alpha-readout')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--benchmark-jobs',type=int,help='Only compute this many new trial/layer jobs, then stop without building reports')
    args=parser.parse_args()
    index=json.loads((args.run/'index.json').read_text())
    steering=[t['metadata']['steering'] for t in index['trials'] if t['metadata']['steering'] is not None]
    if not steering or any(s.get('method')!='additive' for s in steering):
        parser.error('The supplied folder is not an additive alpha experiment')
    if args.threads<1 or (args.benchmark_jobs is not None and args.benchmark_jobs<1):
        parser.error('threads and benchmark-jobs must be positive')
    audit=root/'data/word_tokenization/word_tokenization.json'
    if not audit.exists():
        audit.parent.mkdir(parents=True,exist_ok=True)
        audit.write_bytes(gzip.decompress((root/'results/mini-batch-cluster-readout/word-tokenization.json.gz').read_bytes()))
    assets=['tokenizer.json','download-provenance.json','unembed/meta.json','unembed/lm_head.weight.bf16.bin',
            'unembed/model.norm.weight.bf16.bin','qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt']
    missing=[name for name in assets if not (args.data/name).is_file()]
    if missing:
        parser.error('Missing readout assets: '+', '.join(missing)+'. Run scripts/fetch_mini_readout_assets.py --data '+str(args.data))
    env=os.environ.copy()
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
    readout=[sys.executable,'-u','-m','quantitive_analysis.local_readout','--run',str(args.run.resolve()),
             '--data',str(args.data.resolve()),'--output',str(args.readouts.resolve()),'--threads',str(args.threads),
             '--layers','29','32','35','62']
    if args.benchmark_jobs is not None:
        readout+=['--limit',str(args.benchmark_jobs)]
    subprocess.run(readout,cwd=root,env=env,check=True)
    if args.benchmark_jobs is not None:
        print('Benchmark finished; rerun without --benchmark-jobs to resume all jobs.',flush=True)
        return
    subprocess.run([sys.executable,'-u','scripts/build_alpha_analysis.py','--run',str(args.run.resolve()),
                    '--readouts',str(args.readouts.resolve()),'--output',str(args.reports.resolve())],cwd=root,env=env,check=True)
    subprocess.run([sys.executable,'-u','scripts/build_mini_batch_token_html.py','--run',str(args.run.resolve()),
                    '--readouts',str(args.readouts.resolve()),'--comparisons',str(args.reports.resolve()/'comparisons.json'),
                    '--output',str(args.reports.resolve()/'token-comparison.html')],cwd=root,env=env,check=True)
    overview=args.reports/'index.html'
    overview.write_text(overview.read_text().replace('The token-level page is built separately by the cluster job after these tables.',
                       '<a href="token-comparison.html">Open token trajectories and answer pairs →</a>'))
    print('Finished. Open '+str(overview.resolve()),flush=True)


if __name__=='__main__':
    main()
