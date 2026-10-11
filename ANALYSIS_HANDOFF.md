# Run the mini-batch evaluation

Branch: `analysis/mini-batch-evaluation-20261009`.

## Rebuild the completed analysis (no GPU)

Python 3.11 or newer is required. From a new checkout:

```bash
git clone --branch analysis/mini-batch-evaluation-20261009 --single-branch https://github.com/danielbischofff/UnderstandLanguageModels_Project.git
cd UnderstandLanguageModels_Project
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python scripts/run_mini_batch_analysis.py
```

Open `results/mini-batch-cluster-readout/index.html` in a browser. On a cluster,
copy that results directory to your own computer to view the interactive HTML.
Repository access is required if GitHub requests authentication.

This rebuilds the question-level, overall mean, experimental median, question
comparison, and overview tables using the completed scores supplied here.
It also unpacks the existing token-comparison HTML and tokenization audit.
It does not regenerate model responses or recompute J-Lens from activations.
The compressed token page keeps the checkout smaller; no individual added file
exceeds GitHub's 100 MB limit. The token plot remains a saved readout snapshot.

The dataset contains 396 model trials, 12 questions, eight steered personas,
two bandwidths, two steering schemes, and four readout layers. Configurations
remain separate. Bootstrap intervals resample question IDs jointly across
personas; their interpretation is exploratory for this small selected set.
The median comparison retains the original mean and distinguishes the median
of 96 pairs from the median of eight persona means.

## Recompute alpha J-Lens directly on cgpool1910

The verified input folder is
`/home/bischoffd/Documents/UnderstandLanguageModels_Project/results`:
396 additive trials, α = 1.5 and 4.0, twelve questions, and all 396 activation
exports present. This workstation has `tmux` and no Slurm. After installing the
readout dependencies and downloading the pinned assets, use:

```bash
.venv-analysis/bin/python scripts/run_alpha_analysis.py \
  --run /home/bischoffd/Documents/UnderstandLanguageModels_Project/results \
  --reports /home/bischoffd/Documents/UnderstandLanguageModels_Project/results/mini-alpha-analysis \
  --threads 4
```

Run it inside `tmux` for a persistent session. `--benchmark-jobs 2` provides a
short resumable timing check before the full run. The direct launcher restores
the audit automatically and performs readout, aggregation and all dashboards.

## Optional: recompute J-Lens readouts with Slurm

### Alpha results already stored on the cluster

Use the original alpha run folder containing `index.json`, `provenance/manifest.json`,
and each trial's activation tensors. The folder's absolute path must be supplied;
the cgpool1910 path is verified above. For a different cluster, supply its local
copy of the data. Run these commands in your collaborator's cluster account:

```bash
git clone --branch analysis/mini-batch-evaluation-20261009 --single-branch https://github.com/danielbischofff/UnderstandLanguageModels_Project.git
cd UnderstandLanguageModels_Project
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -e . -r requirements-analysis.txt
# Restore the shared tokenization audit without rerunning the band analysis:
.venv-analysis/bin/python -c 'import gzip,pathlib; p=pathlib.Path("data/word_tokenization/word_tokenization.json"); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(gzip.decompress(pathlib.Path("results/mini-batch-cluster-readout/word-tokenization.json.gz").read_bytes()))'
# Download the pinned readout assets on a node with internet access:
.venv-analysis/bin/python scripts/fetch_mini_readout_assets.py
mkdir -p runtime/logs
sbatch slurm/mini-alpha-analysis.sbatch /ABSOLUTE/PATH/TO/ALPHA_RESULTS
```

For an existing checkout, use `git fetch origin` and
`git switch analysis/mini-batch-evaluation-20261009` instead of cloning.
Add required cluster account/partition flags to `sbatch`. If the assets are
already available, set `ULM_READOUT_ASSETS=/absolute/path/to/qwen3-32b` instead
of downloading them. The default outputs are:

- `outputs/analysis/mini-alpha-readout`: resumable per-trial exact ranks and cosines.
- `results/mini-alpha-readout/index.html`: alpha analysis overview and sortable table.
- Separate per-question, per-persona, overall mean, and median JSON summaries.

This recomputes J-Lens from the stored alpha activations. It does not generate
new model answers. Alpha values are correctly labelled α, with no bandwidth β
interpretation. Results are paired with controls from that alpha run, and output
directories are separate from the band experiment. Reports are generated only
after all expected trial/layer jobs are present. Both generation-time captures
and complete-sequence replays are supported; all recorded response positions
are included. If rerunning older readouts after a code update, select a fresh
readout directory when provenance validation requests it.

### Band results

This additionally needs the original run directory, including its provenance
activation tensors, at `outputs/mini_batch_exp_results`. Those large files are
not in GitHub. The pinned model assets are approximately 4.9 GB. Run from the
repository root, after completing the setup above:

```bash
.venv-analysis/bin/python -m pip install -e .
.venv-analysis/bin/python scripts/fetch_mini_readout_assets.py
mkdir -p runtime/logs
UV_PROJECT_ENVIRONMENT="$PWD/.venv-analysis" sbatch slurm/mini-batch-readout.sbatch
```

Add your cluster's required account or partition flags to `sbatch`. The job is
CPU-based, requests eight cores, 32 GB RAM and twelve hours, and resumes completed
trial/layer jobs. Fetch assets on a node with internet access before submission.
This produces fresh per-trial readouts and comparison summaries under
`outputs/analysis/mini-batch-cluster-readout`; it does not replace the shared
completed score snapshot automatically.

## Validation

```bash
.venv-analysis/bin/python -m pip install pytest
.venv-analysis/bin/python -m pytest -q tests/test_question_aggregation.py tests/test_overall_aggregation.py tests/test_overall_medians.py
```

The HTML behavior checks also use Node.js:

```bash
node tests/overall_aggregation_html.cjs
node tests/overall_median_html.cjs
node tests/overview_sorting.cjs
node tests/question_aggregation_html.cjs
node tests/question_comparison_html.cjs
```
