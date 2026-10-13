# Mini-batch J-Lens readout

This directory contains the completed CPU readout summary for
`outputs/mini_batch_exp_results`: 396 trials, four layers (29, 32, 35 and 62),
exact full-vocabulary mean reciprocal rank (MRR), and activationWordCosine.

Open [`index.html`](index.html) locally for links to all available analyses and the filterable score tables. It is a
dependency-free HTML file and embeds the compact comparison data.

Open [`overall-aggregation.html`](overall-aggregation.html) for a table of all
64 configurations, averaging the twelve paired question differences within
each persona and then giving all eight personas equal weight. MRR, cosine and
both specificity margins are reported separately with positive-persona counts.
The 10,000-resample percentile intervals resample question IDs jointly across
all personas; they do not treat the 96 pairs as independent observations or
resample personas. These are exploratory summaries for the selected questions
and personas. [`overall-aggregation.csv`](overall-aggregation.csv) contains the
complete table; [`overall-aggregation.json`](overall-aggregation.json) also
records individual persona and question means and the source hash.
The overview includes a four-configuration preview (complete tokens), with
readout-layer and word-bank selectors. Both tables show numerical means,
95% bootstrap intervals and positive-persona counts. Click any column header
to sort and click it again to reverse the order; metric columns sort by their
mean difference. Sorting persists when filters change.

Open [`overall-median-comparison.html`](overall-median-comparison.html) for an
experimental sensitivity analysis comparing the original overall mean, the
median of 96 persona–question paired differences, and the median of eight
per-persona question means. Each median interval is recomputed from 10,000
joint question resamples. These summarize different quantities; the median
of persona means still retains sensitivity to question outliers within a
persona. The original mean table remains unchanged. JSON and CSV exports
are saved with the same `overall-median-comparison` basename.

```bash
.venv/bin/python scripts/compare_overall_medians.py
.venv/bin/python -m pytest -q tests/test_overall_medians.py
node tests/overall_median_html.cjs
```

```bash
.venv/bin/python scripts/aggregate_overall_mini_batch.py
.venv/bin/python -m pytest -q tests/test_overall_aggregation.py
```

The JSON files contain the source data used by the overview:

- `comparisons.json`: per-question paired control deltas and persona margins.
- `unsteered.json`: per-question unsteered response scores for the eight
  experiment personas' word sets, reconstructed as score minus paired delta
  from `comparisons.json`. The builder checks agreement across steering
  conditions; subtraction can introduce rounding in the last digits.
- `topic-balanced-summary.json`: topic-balanced summaries, including the
  optional Q01/Q23 exclusion view. Those questions were functionality checks,
  not calibration questions, and remain in the main twelve-question analysis.

The page includes separate unsteered and steered per-question tables. The
unsteered table follows the subset, word policy, word-set type and layer
filters; scheme and beta apply only to steering results. Delta means steered
minus unsteered. GitHub's file viewer does not execute the interactive HTML;
download the file and open it in a browser.

Open [`token-comparison.html`](token-comparison.html) for token-level MRR and
cosine plots with an unsteered baseline and side-by-side answer texts. It embeds
all 396 trials at readout layers 29, 32, 35 and 62, including both band widths
and steering schemes. It works offline and needs no server. Matching plot
positions represent ordinal answer-token positions; the two answers can have
different tokens and lengths. Hover to inspect both token texts and scores.
Choose **All questions** in the Question selector for equal-question mean
curves over all 12 questions, including Q01 and Q23. Each position averages
only valid scores from answers that reach that position; hover shows the
contributing question count separately for each curve and metric. All answer
pairs remain available below the plots.
Both metrics show a transparent green fill wherever the steered curve exceeds
the unsteered curve. Crossings are interpolated between token positions; no
fill is drawn across missing scores or beyond either answer's scored range.
The word/token table lists each retained target word, its decoded token text,
token ID, and whether it is a complete token or a proxy. It follows the persona,
question, word-set type and policy selectors; the average view lists all 12
question-specific sets.

Open [`question-aggregation.html`](question-aggregation.html) for the main
equal-question analysis. [`question-aggregation.json`](question-aggregation.json)
contains 512 separate conditions, their mean and median paired differences,
positive/negative question counts, 95% paired-question bootstrap intervals
(10,000 samples, fixed seed), and all 6,144 source question rows. It includes
MRR, cosine, and their persona specificity margins. All 12 questions are
included; conditions and personas are not pooled. These exploratory intervals
describe variability across these questions and are not corrected for multiple
comparisons. This measures readout association rather than literal word mentions.

Open [`question-comparison.html`](question-comparison.html) to compare paired
MRR and cosine differences across all twelve questions. Select a persona, band
width, steering scheme, readout layer and word-bank policy. Positive bars are
transparent green; a dashed line shows the equal-question mean. The all-persona
view shows eight separate series. Both plots share the selected question order,
and the page includes the paired scores and full question wording. It uses
`question-aggregation.json` and works offline.

Open [`secondary-parameter-evaluation.html`](secondary-parameter-evaluation.html)
for four violin plots comparing contextualised versus non-contextualised word
sets and complete-token versus proxy-inclusive policies for MRR and cosine.
Layer, steering scheme and band width stay fixed; each comparison also fixes
the other secondary parameter. Select all personas or one persona, and paired
differences or absolute steered scores. Each dot is a persona–question score;
both categories retain only matched pairs with valid scores. The inner box
shows the median and quartiles. These are descriptive distributions, not
confidence intervals. The page embeds its data and plotting library for offline use.

```bash
.venv/bin/python scripts/build_secondary_parameter_html.py
node tests/secondary_parameter_html.cjs
```

```bash
.venv/bin/python scripts/build_question_comparison_html.py
node tests/question_comparison_html.cjs
```

```bash
.venv/bin/python scripts/aggregate_mini_batch.py
.venv/bin/python -m pytest -q tests/test_question_aggregation.py
node tests/question_aggregation_html.cjs
```

Rebuild it from the saved per-trial JSON/NPZ readouts with:

```bash
.venv/bin/python scripts/build_mini_batch_token_html.py
node tests/mini_batch_token_html.cjs
```

The large per-trial `.npz` readouts and raw activation export remain on Helix;
they are intentionally not committed here. Answer-position scores omit the
final generated token because it has no cached input activation.
