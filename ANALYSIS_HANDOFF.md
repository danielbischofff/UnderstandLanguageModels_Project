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

## Optional: recompute J-Lens readouts on a cluster

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
