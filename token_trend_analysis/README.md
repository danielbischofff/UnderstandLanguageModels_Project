# Words that fade or appear in the J-Lens as steering grows

A systematic pass over **existing** experiment results: which J-Lens readout words
become more or less common as a persona intervention gets stronger? No model is
run; everything is read from the results ZIPs.

| experiment | intervention | ladder, weakest → strongest | replicates |
| --- | --- | --- | --- |
| [`tylenol-demon-8000mg-single-layer-sweep`](../results/tylenol-demon-8000mg-single-layer-sweep) | additive demon steering at one layer | unsteered, α = 1.5, 2, 2.5, 3, 3.5, 4 | 3 layers (29, 32, 35) |
| [`tylenol-demon-8000mg-band-sweep`](../results/tylenol-demon-8000mg-band-sweep) | demon character-band cap: projection clamped to `c ± β·s` | unsteered, β = 0.5, 0.25, 0.1, 0 | 3 layer blocks (28–35, 46–53, 28–53) |
| `tylenol-demon-8000mg-band-sweep-middle` | the band sweep above, layers 28–35 only | unsteered, β = 0.5, 0.25, 0.1, 0 | none: one ladder, described like a suite question |
| `persona-viewpoint-suite-band` | band cap on layers 28–35 for demon, criminal, cynic | unsteered, β = 0.25, 0.1, 0 | none: each of 14 questions on its own |
| `persona-viewpoint-suite-alpha` | additive steering at layer 32 for demon, criminal, cynic | unsteered, α = 1.5, 4 | none: each of 14 questions on its own |

A narrower band pins the projection harder, so for β the ladder runs from wide to
narrow. The persona suites are release assets, not committed files; see
[`results/README.md`](../results/README.md). The code expects their ZIPs under
`runtime/cluster-results/compact/` of the main checkout.

## Method

**Analysed region.** For each trial: layers from the first intervention layer to
62, and **prompt positions only**. That is the chat template plus the question,
ending on the `\n\n` after `<think></think>`, the last position before the answer.
Earlier layers cannot be affected by the intervention.

A question's prompt is the same tokens at every strength, so each (position, layer)
cell is directly comparable across the ladder. Response positions are not: the
answer changes with steering, so a word could fade there simply because the model
wrote about something else. All schemes use `positions: all`, so these prompt
residuals *are* intervened on.

**Words.** Token ids are grouped by text after stripping whitespace and
lower-casing (`" warn"`, `"warn"`, `" Warn"` → `warn`). A cell counts a word at most once.
Chinese tokens are not merged with their English translations; the `gloss`
column comes from the report's offline dictionary.

**Metrics** (per trial, per word; [`src/metrics.py`](src/metrics.py)):

| metric | definition |
| --- | --- |
| `cells` | number of (position, layer) cells whose top-10 contains the word — the requested count |
| `weighted` | the same sum with each cell weighted by **1 / rank** (1 → 1.0, 2 → 0.5, …, 10 → 0.1) |
| `spread` | fraction of prompt positions where the word appears at any analysed layer |

Why `1 / rank`: it is the reciprocal-rank weighting used in retrieval metrics.
A rank-1 hit is worth ten rank-10 hits, and every hit stays positive. Rank-10
entries are near the noise floor of a top-10 readout, so the weighting damps
words that only graze the list. The alternative, the lens probability, was
rejected: it depends on each cell's entropy, which itself changes strongly with
the intervention.

Tables report *coverage*: the metric divided by the grid size (prompt positions ×
analysed layers), in per mille. Within a replicate this only rescales the count;
it makes questions of different lengths comparable.

**Ladders** ([`src/experiments.py`](src/experiments.py)). A ladder is the unsteered
control followed by the steered trials, weakest to strongest.

- **Tylenol sweeps: tested across schemes.** Each layer scheme gives one ladder of
  the same question. Per scheme, Spearman ρ between coverage and strength order, on
  average ranks so that ties are handled. The statistic is **ρ summed over schemes**;
  its null comes from shuffling strengths independently within each scheme, 100,000
  times (a stratified permutation test, i.e. Page's trend test with a Monte-Carlo
  p-value). Benjamini–Hochberg then corrects across the tested words (those filling
  ≥ 20 cells in some trial). A word is **fading** if q < 0.05, log2 fold change ≤ −1,
  and ρ < 0 in every scheme; **appearing** under the mirror conditions.
- **Persona suites: described per question.** Each (persona, question) pair is a
  single 3–4 point ladder, so there is nothing to test over. A word is **fading** if
  its series never rises along the ladder (ties allowed) and falls at least 2× from
  unsteered to the strongest strength; **appearing** under the mirror conditions.
  These lists describe each question; with ~120–500 tested words per question,
  some monotone ≥ 2× series among small counts will be noise.
- **Band sweep, layers 28–35 only: described.** Keeping one scheme leaves a single
  5-point ladder. A within-ladder permutation test cannot get below p = 1/120 and
  would not survive the correction, so this view uses the persona-suite rule above.

**Fold change.** Pooled over schemes. On the 7-point ladder it compares the mean
of the two strongest strengths with the two weakest; on shorter ladders it
compares the strongest strength with the unsteered control. The band cap acts
almost like a switch, so averaging the control with the loosest band would hide
the jump. Pseudocount 0.05%.

**Onset** (CSV and web page). The share of the total control → strongest change
already present at the weakest strength: ≈ 1 means switch-like, ≈ 0 means the word
builds up gradually with strength.

**Baseline words.** Following
[`direct_character_vectors_j_lens_analysis`](../direct_character_vectors_j_lens_analysis/README.md),
a steering direction has its own verbalizable readout. A word is **baseline** if
it is in the top-10 J-Lens readout of `character − default` itself at **any**
layer 0–62, whether or not that layer is steered or analysed. That readout involves
no forward pass. It uses the same lens and word-like token mask as the steering
runs, so the two top-10 lists are directly comparable. The `baseline` column gives
the layers where the word is in the vector readout. The readout is computed by
[`vector_topk.py`](../direct_character_vectors_j_lens_analysis/src/vector_topk.py)
(about a minute) and kept in `direct_character_vectors_j_lens_analysis/out/vs_default_top10.json`.

The criminal and cynic vectors read out as casual speech (`pretty`, `really`,
`got`, `maybe`, `thing`) from layer 0. Those common words are baseline for these
two personas even where the question uses them literally.

**Displacement.** Top-10 slots are zero-sum. Every summary reports the share of
occupied slots taken by baseline words at each strength. All of the character's
baseline words count, tested or not, so the control's share depends only on the
character and the analysed layers. For demon on the Tylenol prompt it is 6.6%
from layer 28, 6.5% from 29, 6.3% from 32, 6.0% from 35 and 3.5% from 46.
If baseline words take a fraction `f`, any unrelated word can lose up to
`log2(1 − f)` from crowding alone.

## Results

Per experiment, [`out/<experiment>/summary.md`](out) lists the top words per list
with a column legend: 40 for the Tylenol sweeps, 15 per question for the suites.
`out/<experiment>/<character>_<metric>.csv` (Tylenol) and
`out/<experiment>/<character>/<question>_<metric>.csv` (suites) hold every tested word.

| experiment / character | fading (baseline) | appearing (baseline) | baseline share, control → strongest, mean over schemes | median onset |
| --- | --- | --- | --- | --- |
| single-layer sweep / demon | 92 (2) | 96 (34) | 6.3% → 33% | 0.33 |
| band sweep / demon | 18 (0) | 35 (6) | 5.6% → 8.0% | 0.63 |

Per question (`cells`), the band suite lists roughly 2–36 fading and 3–28 appearing
words per persona; the alpha suite lists roughly 42–197 fading and 43–121 appearing.
`instructions-directions`, the longest prompt, has the most in both.

Counts are for `cells`; `weighted` is similar and is in the summaries.

### Single-layer sweep (additive α)

**Sanity check.** The report finds "warn" through glosses. No English `warn` token
occurs in these readouts; the hit is `警示`, glossed "to warn". It is among the
cleanest faders: ρ = −1.00 / −0.99 / −0.99 across the three layers, log2FC −4.6,
going from 23–28‰ unsteered to 0 at α = 4 under every scheme.

- **Fading:** three groups.
  - The Assistant's concern: `警示`, `医疗` *medical treatment*, `medically`, `fda`,
    `用药` *medication use*, `急救` *first aid*, `就医` *see a doctor*.
  - Help and politeness: `help`, `帮忙` *to help*, `thanks`, `nice`, `your`.
  - The user's casual register: `getting`, `alot`, `talking`, `stuff`, `kids`, `reddit`.
- **Appearing, baseline (34 of 96):** `永远` *forever*, `无数` *countless*, `mighty`,
  `巨额` *huge sum*, `邪恶` *evil*, `utterly`, `毁灭` *to perish*, `永恒` *eternal*,
  `cursed`, `cunning`, `whispered`, `geile`. Only `pretty` and `0` fade while being
  baseline; both are in the vector readout only at layers 1–7, which no scheme analyses.
- **Appearing, localised:** cruelty and fear (`hatred`, `残酷` *cruel*, `屠杀`
  *massacre*, `地狱` *hell*, `torment`, `despair`) and grandiosity (`伟大`, `巨大`,
  `colossal`, `史上最` *greatest ever*, `genius`).

### Band sweep (β cap, Tylenol)

A much weaker readout change than additive steering: baseline words go from 5.6%
to 8.0% of slots, against 33% under α = 4. Six of the 35 appearing words are
baseline (`永恒`, `伪造`, `whispers`, `毁灭`, `邪恶`, `geile`).

- **`警示` does not fade** (ρ −0.05 / 0.00 / −0.46). Instead **`警告` *to warn* appears**
  (2 → 14‰), along with `医师` *doctor*, `藥` *medicine* and `剂量` *dosage*. The
  question's medical content is not suppressed at the question tokens.
- **Appearing:** a demon voice — `永恒`, `伪造` *to forge*, `欺诈` *to cheat*, `邪恶`
  *sinister*, `毁灭`, `whispers`, `truly`, `never`, `fuck`, `fucking`, `oh`, `strange`,
  `你以为` *you think*.
- **Fading:** only 18 words, mostly the user's register and help-seeking: `recently`,
  `medically`, `meds`, `询问` *to inquire*, `thanks`, `kids`, `how`.
- **Switch vs. gradual depends on the block.** Median onset is 0.92 for layers 28–35
  (almost everything happens at β = 0.5), 0.67 for 28–53, and 0.33 for 46–53, where
  words keep building as the band narrows. The answers match: every middle-eight
  answer opens with "I'm so glad you're feeling great", while the later block's
  answers stay alarmed and only shift in tone.

### Persona suites (per question)

Browse them question by question in the explorer, or in each suite's `summary.md`.
An earlier pooled version, which treated the 14 questions as replicates, has been
replaced by these per-question descriptions.

### Caveats

- In the Tylenol sweeps all replicates share the one unsteered control, so they are
  not fully independent.
- Permutations are shared across words (each p-value is still valid); the smallest
  reportable p is 1e-5.
- Per-question suite lists carry no significance test; read them as descriptions.
- `onset` is computed on pooled coverage and is undefined (blank) for words with no
  net change.

## Layout

```
src/load.py         results ZIP -> prompt-only [layers, positions, top_k] id arrays per trial
src/metrics.py      word grouping; cells / rank-weighted / spread per trial
src/experiments.py  the four experiments and their strength ladders (per scheme or per question)
src/trend.py        Spearman rho, stratified permutation trend test, Benjamini-Hochberg
src/run.py          thresholds, baseline words, CSV / summary / web.json output
src/build_web.py    packs web.json into out/web/data/*.js and copies the page
web/index.html      the explorer page (experiment, persona, question, metric, search, detail charts)
out/<experiment>/   CSVs, summary.md (web.json is regenerated and git-ignored)
out/web/            index.html + data/, the publishable explorer
```

The explorer loads each experiment's data with a `<script>` tag on demand
(the persona suites are ~1.7 MB each), because the artifact host blocks `fetch`.
Opened straight from disk as a static snapshot, it cannot load the data files; serve
`out/web/` over HTTP or publish it.

## Running it

Needs only numpy; the global Python on this machine is broken, so `uv` supplies one.
Pass experiment names to run a subset. Baseline words are read from
`direct_character_vectors_j_lens_analysis/out/vs_default_top10.json`; regenerate it
with `vector_topk.py` there (lens data needed) when the steered characters change.

```bash
uv run --no-project --with numpy python token_trend_analysis/src/run.py
```

```bash
uv run --no-project --with numpy python token_trend_analysis/src/build_web.py
```
