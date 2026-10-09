#!/usr/bin/env python3
"""Rebuild all compact analysis pages from the shared, completed score data."""
import gzip
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'results/mini-batch-cluster-readout'
    steps = [
        ['scripts/aggregate_mini_batch.py', '--questions', str(folder / 'questions.json')],
        ['scripts/aggregate_overall_mini_batch.py'],
        ['scripts/compare_overall_medians.py'],
        ['scripts/build_question_comparison_html.py'],
        ['scripts/build_mini_batch_html.py', str(folder), str(folder / 'index.html')],
    ]
    for step in steps:
        subprocess.run([sys.executable, *step], cwd=root, check=True)
    for archive, target in [(folder / 'token-comparison.html.gz', folder / 'token-comparison.html'),
                            (folder / 'word-tokenization.json.gz', root / 'data/word_tokenization/word_tokenization.json')]:
        if not target.exists() and archive.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(archive, 'rb') as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
    print(f'Open in your browser: {folder / "index.html"}')
    print('Uses completed readout scores; no model inference or GPU is required.')


if __name__ == '__main__':
    main()
