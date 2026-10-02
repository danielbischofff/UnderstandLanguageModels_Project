# J-Lens words that fade or appear with steering: `tylenol-demon-8000mg-band-sweep-middle`

Demon character-band capping on layers 28–35 only, β 0.5 → 0, on the 8000 mg Tylenol prompt; a single ladder, so trends are described rather than tested. Prompt positions only (through the newline before the answer), layers from the first intervention layer to 62, ladders weakest → strongest.

Each question is described on its own, from its single ladder.

**Columns**

- **word** — tokens that differ only in leading space or capitalisation, merged.
- **gloss** — dictionary translation for non-English tokens (from the report's offline glossary).
- **flag** — `suspected baseline`: once steered, at ≥ 50% of prompt positions (any analysed layer) and at least twice as widely as unsteered, i.e. probably the steering vector's own readout rather than content; `possible baseline`: the same at ≥ 25%; `broad, also unsteered`: ≥ 25% but already that broad without steering. Empty = localised.
- **in vector readout** — `yes` if the word is in the direct J-Lens readout of the character − default vector itself (atlas top-8, from the first intervention layer).
- **Listed** when the word's series never moves against the trend along the ladder (ties allowed) and changes by at least 2× overall. One 3–4 point ladder cannot be tested for significance, so these lists describe each question; expect some noise among small counts.
- **rho** — rank correlation of the series with strength order (−1 = strictly falling, +1 = strictly rising; ties pull it towards 0).
- **log2FC** — log2 of (strongest strength) / (unsteered control). −1 = halved, +3 = eight times more.
- **‰ by strength** — one number per strength, weakest to strongest: the word's count (or rank-weighted count) in thousandths of the question's grid (prompt positions × layers from the intervention layer to 62).


## `demon`


### `tylenol-more` — demon

203 words tested. Baseline share of the readout: unsteered 1%, β=0.5 2%, β=0.25 3%, β=0.1 3%, β=0 3%.


#### Metric: `cells`


##### Fading: 7 words (0 suspected, 0 possible baseline), top 7 by |log2FC|

| word | gloss | flag | in vector readout | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- |
| `recently` |  |  |  | -0.71 | -2.77 | 19.4 2.4 2.4 2.4 2.4 |
| `meds` |  |  |  | -0.95 | -2.30 | 14.0 3.0 3.0 2.4 2.4 |
| `医` | medical |  |  | -0.97 | -1.87 | 28.0 9.1 8.5 8.5 7.3 |
| `thanks` |  |  |  | -1.00 | -1.54 | 34.6 14.6 14.0 12.8 11.6 |
| `vitamin` |  |  |  | -0.71 | -1.49 | 14.6 4.9 4.9 4.9 4.9 |
| `疗` | to treat |  |  | -1.00 | -1.24 | 12.2 7.9 6.7 6.1 4.9 |
| `lgbtq` |  |  |  | -0.97 | -1.15 | 12.8 8.5 6.1 5.5 5.5 |

##### Appearing: 16 words (0 suspected, 1 possible baseline), top 15 by |log2FC|

| word | gloss | flag | in vector readout | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- |
| `truly` |  |  |  | +0.97 | +5.08 | 0.0 9.1 15.8 15.8 16.4 |
| `mere` |  |  | yes | +0.97 | +4.73 | 0.0 9.7 12.2 12.2 12.8 |
| `never` |  |  |  | +1.00 | +4.66 | 0.0 2.4 9.1 11.6 12.2 |
| `whispers` |  |  | yes | +1.00 | +4.66 | 0.0 9.1 10.9 11.6 12.2 |
| `邪恶` | sinister |  |  | +0.95 | +4.13 | 0.6 18.2 18.2 18.8 18.8 |
| `藥` | medicine | possible baseline |  | +1.00 | +3.34 | 1.8 19.4 20.7 22.5 23.1 |
| `逆袭` | to counterattack |  |  | +1.00 | +2.51 | 1.8 9.1 11.6 12.2 12.8 |
| `blowjob` |  |  |  | +0.89 | +1.92 | 3.6 14.6 14.6 14.6 15.2 |
| `感受` | to sense |  |  | +0.97 | +1.85 | 4.9 17.0 17.6 17.6 18.8 |
| `again` |  |  |  | +0.97 | +1.72 | 4.3 14.0 14.6 15.2 15.2 |
| `开心` | to feel happy |  |  | +0.97 | +1.43 | 4.9 12.2 12.2 13.4 14.0 |
| `喜悦` | happy |  |  | +0.97 | +1.37 | 4.9 11.6 12.8 13.4 13.4 |
| `ty` |  |  |  | +0.89 | +1.28 | 9.7 22.5 24.3 24.3 24.3 |
| `sadness` |  |  |  | +0.71 | +1.15 | 5.5 12.8 12.8 12.8 12.8 |
| `精神病` | mental disorder |  |  | +0.97 | +1.13 | 6.7 12.8 14.0 14.0 15.2 |

#### Metric: `weighted`


##### Fading: 4 words (0 suspected, 0 possible baseline), top 4 by |log2FC|

| word | gloss | flag | in vector readout | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- |
| `thanks` |  |  |  | -1.00 | -2.20 | 10.6 2.2 2.2 1.9 1.9 |
| `meds` |  |  |  | -0.97 | -1.83 | 2.2 0.4 0.4 0.3 0.3 |
| `lgbtq` |  |  |  | -1.00 | -1.18 | 6.1 4.8 2.9 2.6 2.4 |
| `疗` | to treat |  |  | -1.00 | -1.14 | 2.0 1.2 0.8 0.8 0.6 |

##### Appearing: 8 words (0 suspected, 1 possible baseline), top 8 by |log2FC|

| word | gloss | flag | in vector readout | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- |
| `mere` |  |  | yes | +1.00 | +3.93 | 0.0 6.7 7.1 7.1 7.1 |
| `whispers` |  |  | yes | +1.00 | +3.39 | 0.0 3.5 3.9 4.2 4.8 |
| `truly` |  |  |  | +1.00 | +3.17 | 0.0 1.4 2.8 3.9 4.0 |
| `never` |  |  |  | +1.00 | +2.80 | 0.0 0.4 1.8 2.5 3.0 |
| `藥` | medicine | possible baseline |  | +1.00 | +2.38 | 0.2 2.9 3.0 3.2 3.2 |
| `开心` | to feel happy |  |  | +1.00 | +1.50 | 0.8 2.6 2.8 3.0 3.1 |
| `医生` | doctor |  |  | +1.00 | +1.39 | 3.3 8.4 9.3 9.5 9.6 |
| `again` |  |  |  | +1.00 | +1.33 | 1.4 3.8 4.2 4.2 4.3 |
