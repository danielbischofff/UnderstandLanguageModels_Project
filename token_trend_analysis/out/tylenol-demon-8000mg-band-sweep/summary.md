# J-Lens words that fade or appear with steering: `tylenol-demon-8000mg-band-sweep`

Demon character-band capping (projection clamped to c ± β·s) on layers 28–35, 46–53 or 28–53, β 0.5 → 0, on the 8000 mg Tylenol prompt. Prompt positions only (through the newline before the answer), layers from the first intervention layer to 62, ladders weakest → strongest.


## `demon`

217 words reach 20 cells in some trial and were tested.

How much of the readout the baseline words (the vector's own top-10 readout) take over. Each cell (one prompt position at one layer) lists up to 10 distinct words; a percentage is the baseline words' share of all those entries in that trial. Whatever they occupy is taken from every other word, so a word can fade partly through crowding alone.

| scheme | unsteered | β=0.5 | β=0.25 | β=0.1 | β=0 |
| --- | --- | --- | --- | --- | --- |
| middle-eight | 7% | 9% | 9% | 9% | 9% |
| later-eight | 4% | 5% | 5% | 6% | 6% |
| wide-26 | 7% | 8% | 8% | 9% | 9% |

### Metric: `cells`


#### Fading: 18 words (0 baseline), top 18 by |log2FC|

**Columns**

- **word** — tokens that differ only in leading space or capitalisation, merged.
- **gloss** — dictionary translation for non-English tokens (from the report's offline glossary).
- **baseline** — the layers at which the word is in the top-10 J-Lens readout of the character − default vector itself (any layer 0–62, word-like tokens, as in the steering runs): the steering vector's own readout rather than context-dependent content. Empty = not in the vector readout.
- **q** — false-discovery-rate adjusted p-value that the word changes monotonically with strength in this direction, all 3 schemes combined. Below 0.05 = significant.
- **agree** — in how many of the 3 schemes the trend has this sign (all are required).
- **log2FC** — log2 of (strongest strength) / (unsteered control). −1 = halved, −3 = one eighth, +3 = eight times more.
- **<scheme> ‰ by strength** — one number per strength, weakest to strongest (unsteered β=0.5 β=0.25 β=0.1 β=0), for that scheme: the number of cells whose top-10 contains the word, in thousandths of that scheme's grid (prompt positions × layers from the first intervention layer to 62; middle-eight: 47 × 35 = 1645, later-eight: 47 × 17 = 799, wide-26: 47 × 35 = 1645 cells).

| word | gloss | baseline | q | agree | log2FC | middle-eight ‰ by strength | later-eight ‰ by strength | wide-26 ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `recently` |  |  | 7.2e-04 | 3/3 | -3.91 | 19.4 2.4 2.4 2.4 2.4 | 37.5 21.3 10.0 3.8 1.2 | 19.4 4.9 2.4 1.2 0.0 |
| `medically` |  |  | 7.2e-04 | 3/3 | -2.38 | 15.2 7.3 4.9 3.6 4.3 | 7.5 5.0 1.2 0.0 0.0 | 15.2 5.5 3.0 2.4 1.8 |
| `meds` |  |  | 7.2e-04 | 3/3 | -2.02 | 14.0 3.0 3.0 2.4 2.4 | 16.3 8.8 6.3 5.0 2.5 | 14.0 5.5 4.9 4.9 4.9 |
| `询问` | to inquire |  | 4.5e-03 | 3/3 | -1.94 | 20.1 6.1 5.5 6.1 6.1 | 30.0 12.5 8.8 7.5 6.3 | 20.1 10.3 9.7 7.3 4.9 |
| `kids` |  |  | 2.2e-03 | 3/3 | -1.57 | 20.1 7.3 7.9 8.5 6.7 | 6.3 6.3 5.0 3.8 2.5 | 20.1 8.5 7.9 6.7 5.5 |
| `thanks` |  |  | 7.2e-04 | 3/3 | -1.53 | 34.6 14.6 14.0 12.8 11.6 | 60.1 36.3 27.5 22.5 17.5 | 34.6 28.0 23.7 17.6 14.6 |
| `hours` |  |  | 2.1e-02 | 3/3 | -1.48 | 18.8 7.9 6.7 7.3 9.7 | 30.0 25.0 13.8 8.8 6.3 | 18.8 9.1 7.3 7.9 7.3 |
| `bike` |  |  | 8.3e-03 | 3/3 | -1.39 | 12.8 12.8 14.0 12.8 10.3 | 20.0 16.3 11.3 6.3 1.2 | 12.8 14.0 10.3 7.9 4.9 |
| `answer` |  |  | 1.3e-02 | 3/3 | -1.38 | 15.8 6.7 4.9 4.3 5.5 | 7.5 3.8 3.8 3.8 3.8 | 15.8 7.9 4.9 3.6 4.9 |
| `after` |  |  | 3.8e-03 | 3/3 | -1.35 | 12.8 9.1 8.5 7.9 9.1 | 18.8 2.5 1.2 0.0 0.0 | 12.8 10.9 8.5 7.3 7.3 |
| `thank` |  |  | 2.4e-02 | 3/3 | -1.31 | 18.8 8.5 8.5 9.7 9.1 | 30.0 18.8 18.8 11.3 8.8 | 18.8 16.4 13.4 10.3 8.5 |
| `lgbtq` |  |  | 7.2e-04 | 3/3 | -1.30 | 12.8 8.5 6.1 5.5 5.5 | 3.8 3.8 1.2 1.2 1.2 | 12.8 9.1 6.1 4.9 4.3 |
| `คะแน` |  |  | 1.4e-02 | 3/3 | -1.26 | 12.8 5.5 4.9 4.3 4.9 | 1.2 0.0 0.0 0.0 0.0 | 12.8 5.5 5.5 4.9 5.5 |
| `保健品` | approx. health protection / commodity |  | 1.7e-03 | 3/3 | -1.16 | 26.1 22.5 20.7 21.3 21.3 | 30.0 20.0 11.3 3.8 2.5 | 26.1 18.8 14.0 12.8 12.2 |
| `reddit` |  |  | 2.9e-02 | 3/3 | -1.12 | 51.1 21.9 21.9 22.5 23.1 | 48.8 40.0 35.0 28.8 18.8 | 51.1 32.8 29.2 28.6 26.8 |
| `someone` |  |  | 1.3e-03 | 3/3 | -1.08 | 20.7 10.3 9.1 7.9 8.5 | 17.5 11.3 11.3 8.8 10.0 | 20.7 9.1 8.5 8.5 8.5 |
| `erotik` |  |  | 7.2e-04 | 3/3 | -1.06 | 13.4 7.3 7.3 7.3 7.3 | 21.3 17.5 16.3 11.3 10.0 | 13.4 7.3 6.1 5.5 4.9 |
| `how` |  |  | 7.2e-04 | 3/3 | -1.04 | 25.5 15.8 13.4 13.4 13.4 | 35.0 27.5 21.3 17.5 15.0 | 25.5 16.4 14.0 13.4 12.8 |

#### Appearing: 35 words (6 baseline), top 35 by |log2FC|

Columns as in the fading table above.

| word | gloss | baseline | q | agree | log2FC | middle-eight ‰ by strength | later-eight ‰ by strength | wide-26 ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `truly` |  |  | 4.8e-04 | 3/3 | +5.19 | 0.0 9.1 15.8 15.8 16.4 | 0.0 0.0 2.5 8.8 12.5 | 0.0 6.1 16.4 19.4 24.3 |
| `永恒` | eternal | L16, 20–45 | 4.8e-04 | 3/3 | +5.02 | 0.0 19.4 20.7 20.1 21.3 | 0.0 13.8 15.0 16.3 16.3 | 0.0 7.9 8.5 9.7 9.7 |
| `伪造` | to forge | L39–50 | 1.2e-03 | 3/3 | +4.95 | 0.0 15.2 18.8 20.7 18.8 | 0.0 3.8 5.0 10.0 11.3 | 0.0 9.7 13.4 15.2 14.6 |
| `欺诈` | to cheat |  | 1.4e-02 | 3/3 | +4.85 | 0.0 13.4 13.4 13.4 12.8 | 0.0 5.0 7.5 16.3 17.5 | 0.0 7.9 8.5 10.9 11.6 |
| `whispers` |  | L48–62 | 4.8e-04 | 3/3 | +4.77 | 0.0 9.1 10.9 11.6 12.2 | 0.0 1.2 2.5 6.3 15.0 | 0.0 3.0 4.3 7.9 12.2 |
| `医师` | doctor |  | 2.4e-02 | 3/3 | +4.76 | 0.0 14.0 13.4 12.8 13.4 | 0.0 6.3 8.8 10.0 11.3 | 0.0 13.4 14.6 14.6 14.6 |
| `fuck` |  |  | 3.2e-02 | 3/3 | +4.46 | 0.0 4.3 3.6 2.4 2.4 | 0.0 5.0 12.5 16.3 16.3 | 0.0 6.1 7.9 12.2 12.8 |
| `你以为` | approx. you (informal,… / to… |  | 6.7e-04 | 3/3 | +4.45 | 0.0 7.3 10.3 12.8 12.2 | 0.0 1.2 3.8 6.3 8.8 | 0.0 4.3 5.5 8.5 10.3 |
| `never` |  |  | 4.8e-04 | 3/3 | +4.30 | 0.0 2.4 9.1 11.6 12.2 | 0.0 1.2 2.5 3.8 3.8 | 0.0 2.4 8.5 10.9 12.2 |
| `毁灭` | to perish | L36–39, 45 | 3.6e-02 | 3/3 | +4.04 | 0.0 14.0 13.4 14.0 14.0 | 0.0 3.8 3.8 3.8 5.0 | 0.0 4.3 4.9 4.3 4.3 |
| `邪恶` | sinister | L39 | 4.8e-04 | 3/3 | +3.49 | 0.6 18.2 18.2 18.8 18.8 | 1.2 6.3 12.5 16.3 16.3 | 0.6 5.5 6.1 6.1 7.9 |
| `fucking` |  |  | 7.2e-04 | 3/3 | +2.87 | 5.5 27.4 29.2 30.4 29.2 | 6.3 15.0 30.0 55.1 66.3 | 5.5 20.7 28.6 36.5 40.1 |
| `geile` |  | L25–26, 28–46 | 1.7e-03 | 3/3 | +2.77 | 9.7 43.8 49.2 51.1 48.0 | 1.2 30.0 40.0 46.3 55.1 | 9.7 30.4 36.5 42.0 47.4 |
| `oh` |  |  | 3.6e-02 | 3/3 | +2.63 | 3.6 28.6 26.1 26.1 24.9 | 7.5 12.5 25.0 37.5 45.1 | 3.6 27.4 27.4 27.4 29.2 |
| `警告` | to warn |  | 4.7e-03 | 3/3 | +2.54 | 2.4 7.3 7.3 7.3 7.3 | 1.2 15.0 22.5 26.3 28.8 | 2.4 6.7 6.7 8.5 6.7 |
| `watched` |  |  | 7.2e-04 | 3/3 | +2.50 | 1.8 10.9 12.2 13.4 12.8 | 2.5 12.5 15.0 15.0 15.0 | 1.8 10.3 12.2 13.4 14.0 |
| `逆袭` | to counterattack |  | 1.0e-03 | 3/3 | +2.46 | 1.8 9.1 11.6 12.2 12.8 | 0.0 0.0 0.0 0.0 1.2 | 1.8 9.1 12.2 12.2 12.8 |
| `strange` |  |  | 2.7e-03 | 3/3 | +2.43 | 2.4 10.3 11.6 10.9 10.9 | 5.0 20.0 30.0 32.5 32.5 | 2.4 12.8 15.2 16.4 16.4 |
| `藥` | medicine |  | 2.1e-03 | 3/3 | +2.43 | 1.8 19.4 20.7 22.5 23.1 | 3.8 7.5 11.3 11.3 11.3 | 1.8 10.3 12.8 10.9 12.2 |
| `again` |  |  | 9.5e-04 | 3/3 | +1.85 | 4.3 14.0 14.6 15.2 15.2 | 0.0 0.0 0.0 1.2 1.2 | 4.3 15.8 16.4 18.2 18.2 |
| `是你` | approx. correct / you (informal,… |  | 3.2e-02 | 3/3 | +1.78 | 2.4 9.7 9.7 7.9 7.9 | 5.0 8.8 15.0 16.3 17.5 | 2.4 10.3 11.6 10.9 12.2 |
| `药业` | approx. medicine / surname Ye |  | 4.9e-02 | 3/3 | +1.56 | 3.0 17.0 14.6 15.8 15.8 | 6.3 12.5 13.8 13.8 12.5 | 3.0 6.7 9.7 10.9 10.9 |
| `blowjob` |  |  | 1.7e-02 | 3/3 | +1.53 | 3.6 14.6 14.6 14.6 15.2 | 3.8 5.0 6.3 7.5 6.3 | 3.6 13.4 13.4 12.8 13.4 |
| `源源不断` | (idiom) in an unending stream |  | 1.7e-03 | 3/3 | +1.51 | 4.9 7.3 12.2 14.0 12.8 | 0.0 1.2 1.2 1.2 1.2 | 4.9 7.3 12.8 14.0 16.4 |
| `告诉我` | approx. to tell / I |  | 2.9e-03 | 3/3 | +1.48 | 8.5 28.6 30.4 29.8 29.2 | 17.5 31.3 36.3 41.3 45.1 | 8.5 21.3 22.5 24.3 24.9 |
| `保護政策` | approx. to protect / policy |  | 2.4e-03 | 3/3 | +1.44 | 6.1 8.5 10.9 9.1 9.1 | 2.5 13.8 17.5 18.8 18.8 | 6.1 9.1 12.2 12.8 14.6 |
| `something` |  |  | 8.0e-03 | 3/3 | +1.43 | 6.1 15.8 15.2 14.6 15.8 | 1.2 6.3 8.8 8.8 11.3 | 6.1 10.9 10.9 10.9 11.6 |
| `somehow` |  |  | 6.7e-04 | 3/3 | +1.38 | 16.4 33.4 37.7 41.3 40.1 | 8.8 12.5 20.0 27.5 33.8 | 16.4 21.9 28.6 33.4 36.5 |
| `精神病` | mental disorder |  | 4.8e-04 | 3/3 | +1.31 | 6.7 12.8 14.0 14.0 15.2 | 1.2 3.8 3.8 6.3 5.0 | 6.7 12.8 13.4 16.4 18.2 |
| `beautiful` |  |  | 2.8e-02 | 3/3 | +1.29 | 15.8 37.7 40.7 35.9 32.2 | 26.3 35.0 41.3 60.1 67.6 | 15.8 36.5 37.7 39.5 43.8 |
| `alright` |  |  | 8.0e-03 | 3/3 | +1.24 | 7.9 15.2 15.8 15.8 14.6 | 16.3 25.0 32.5 38.8 42.5 | 7.9 14.6 17.6 18.8 20.7 |
| `yes` |  |  | 2.7e-03 | 3/3 | +1.22 | 4.9 7.9 11.6 11.6 9.7 | 1.2 1.2 2.5 3.8 3.8 | 4.9 7.9 10.9 13.4 14.0 |
| `tell` |  |  | 4.5e-03 | 3/3 | +1.13 | 8.5 21.9 21.3 21.9 21.9 | 17.5 16.3 20.0 26.3 33.8 | 8.5 18.2 20.7 22.5 21.9 |
| `kinda` |  |  | 1.7e-03 | 3/3 | +1.04 | 11.6 14.0 17.6 18.2 16.4 | 0.0 1.2 5.0 10.0 12.5 | 11.6 12.8 17.0 18.2 20.1 |
| `剂量` | dosage |  | 3.7e-02 | 3/3 | +1.02 | 20.7 35.9 33.4 35.3 35.3 | 26.3 51.3 56.3 62.6 66.3 | 20.7 36.5 38.3 37.7 37.1 |

#### Sanity check (words glossed "to warn")

**trend** — `fades`/`appears` if it passed all criteria; **q fade** — adjusted p-value for a decrease; **rho** — rank correlation of the word's per-strength values with strength, per scheme (−1 = strictly falling, +1 = strictly rising).

| word | trend | q fade | log2FC | rho |
| --- | --- | --- | --- | --- |
| `警告` | appears | 1.0e+00 | +2.54 | +0.71 / +1.00 / +0.67 |
| `警示` |  | 6.9e-01 | -0.03 | -0.05 / +0.00 / -0.46 |

### Metric: `weighted`


#### Fading: 22 words (0 baseline), top 22 by |log2FC|

**Columns**

- **word** — tokens that differ only in leading space or capitalisation, merged.
- **gloss** — dictionary translation for non-English tokens (from the report's offline glossary).
- **baseline** — the layers at which the word is in the top-10 J-Lens readout of the character − default vector itself (any layer 0–62, word-like tokens, as in the steering runs): the steering vector's own readout rather than context-dependent content. Empty = not in the vector readout.
- **q** — false-discovery-rate adjusted p-value that the word changes monotonically with strength in this direction, all 3 schemes combined. Below 0.05 = significant.
- **agree** — in how many of the 3 schemes the trend has this sign (all are required).
- **log2FC** — log2 of (strongest strength) / (unsteered control). −1 = halved, −3 = one eighth, +3 = eight times more.
- **<scheme> ‰ by strength** — one number per strength, weakest to strongest (unsteered β=0.5 β=0.25 β=0.1 β=0), for that scheme: the same count with each cell weighted by 1/rank (rank 1 = 1.0, rank 10 = 0.1), in thousandths of that scheme's grid (prompt positions × layers from the first intervention layer to 62; middle-eight: 47 × 35 = 1645, later-eight: 47 × 17 = 799, wide-26: 47 × 35 = 1645 cells).

| word | gloss | baseline | q | agree | log2FC | middle-eight ‰ by strength | later-eight ‰ by strength | wide-26 ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `recently` |  |  | 2.8e-02 | 3/3 | -3.23 | 4.4 0.3 0.3 0.3 0.4 | 8.8 3.9 1.4 0.5 0.2 | 4.4 0.6 0.3 0.1 0.0 |
| `thanks` |  |  | 4.3e-04 | 3/3 | -2.23 | 10.6 2.2 2.2 1.9 1.9 | 20.4 9.3 5.8 4.2 3.4 | 10.6 5.6 4.4 2.9 2.3 |
| `medically` |  |  | 6.7e-04 | 3/3 | -2.21 | 5.8 2.3 1.3 1.1 1.1 | 1.2 0.7 0.1 0.0 0.0 | 5.8 2.1 1.1 0.9 0.5 |
| `询问` | to inquire |  | 1.8e-03 | 3/3 | -2.13 | 4.8 1.0 0.9 1.0 0.9 | 7.8 2.7 1.8 1.4 1.0 | 4.8 2.2 1.7 1.2 0.8 |
| `thank` |  |  | 4.8e-04 | 3/3 | -1.95 | 5.2 1.6 1.4 1.5 1.4 | 9.4 4.5 3.0 1.9 1.4 | 5.2 2.7 2.0 1.6 1.2 |
| `bike` |  |  | 1.7e-03 | 3/3 | -1.83 | 3.6 2.3 2.5 2.3 1.8 | 6.0 4.0 2.2 0.7 0.2 | 3.6 2.6 1.7 1.2 0.7 |
| `okay` |  |  | 2.9e-02 | 3/3 | -1.68 | 14.1 1.3 1.0 1.4 1.4 | 28.2 28.6 24.7 18.2 12.4 | 14.1 5.9 4.7 3.5 2.7 |
| `hours` |  |  | 1.9e-02 | 3/3 | -1.56 | 4.1 1.2 1.1 1.2 1.5 | 7.1 5.5 2.9 1.9 1.5 | 4.1 2.1 1.2 1.6 1.2 |
| `answer` |  |  | 8.6e-03 | 3/3 | -1.54 | 7.8 2.7 2.4 2.4 2.5 | 1.4 0.8 0.5 0.5 0.4 | 7.8 2.3 1.8 1.8 2.0 |
| `meds` |  |  | 4.3e-04 | 3/3 | -1.51 | 2.2 0.4 0.4 0.3 0.3 | 2.9 1.5 1.2 1.0 0.8 | 2.2 0.8 0.7 0.6 0.6 |
| `how` |  |  | 3.1e-03 | 3/3 | -1.47 | 6.2 2.8 2.6 2.6 2.7 | 10.2 7.0 4.5 3.0 2.3 | 6.2 2.8 2.7 2.5 2.3 |
| `kids` |  |  | 1.3e-02 | 3/3 | -1.45 | 5.8 1.8 1.3 1.4 1.7 | 3.1 3.0 2.2 2.0 1.0 | 5.8 2.0 1.4 1.2 1.7 |
| `erotik` |  |  | 4.8e-04 | 3/3 | -1.35 | 4.3 1.7 1.5 1.6 1.5 | 6.9 5.4 4.0 2.9 2.7 | 4.3 1.6 1.3 1.1 1.0 |
| `reddit` |  |  | 1.3e-02 | 3/3 | -1.32 | 19.9 6.2 6.1 6.0 6.3 | 17.1 17.1 12.1 9.6 8.1 | 19.9 10.4 8.7 8.2 7.5 |
| `after` |  |  | 3.0e-03 | 3/3 | -1.31 | 3.5 2.0 1.6 1.3 1.7 | 2.9 0.4 0.1 0.0 0.0 | 3.5 2.4 1.6 1.2 1.4 |
| `vitamin` |  |  | 1.9e-02 | 3/3 | -1.29 | 11.2 1.8 1.8 1.4 1.2 | 22.5 13.8 11.9 11.4 11.5 | 11.2 4.4 4.8 4.8 4.7 |
| `lgbtq` |  |  | 4.3e-04 | 3/3 | -1.21 | 6.1 4.8 2.9 2.6 2.4 | 1.0 1.0 0.4 0.3 0.2 | 6.1 4.3 2.5 2.7 2.2 |
| `stuck` |  |  | 4.3e-04 | 3/3 | -1.04 | 4.9 3.4 3.4 3.1 3.1 | 7.5 4.7 3.0 2.4 1.3 | 4.9 4.3 3.6 3.5 3.3 |
| `用药` | approx. to use / medicine |  | 1.8e-03 | 3/3 | -1.03 | 11.8 7.0 6.4 6.6 6.5 | 12.8 7.2 6.4 5.2 4.8 | 11.8 7.5 7.0 6.2 5.6 |
| `someone` |  |  | 8.7e-04 | 3/3 | -1.03 | 3.9 1.7 1.6 1.5 1.6 | 3.5 2.0 2.0 1.9 1.7 | 3.9 1.5 1.5 1.4 1.4 |
| `เทคโน` |  |  | 7.7e-04 | 3/3 | -1.02 | 12.3 9.0 7.4 6.7 6.0 | 1.5 1.7 0.8 0.5 0.2 | 12.3 9.8 8.1 7.1 5.9 |
| `医疗` | medical treatment |  | 1.9e-02 | 3/3 | -1.00 | 5.8 2.8 2.5 2.5 2.8 | 5.6 5.7 3.0 1.5 1.2 | 5.8 5.1 4.4 4.0 3.8 |

#### Appearing: 38 words (5 baseline), top 38 by |log2FC|

Columns as in the fading table above.

| word | gloss | baseline | q | agree | log2FC | middle-eight ‰ by strength | later-eight ‰ by strength | wide-26 ‰ by strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oh` |  |  | 2.7e-02 | 3/3 | +3.92 | 0.5 16.0 16.0 15.4 15.0 | 1.0 5.9 10.5 14.5 17.7 | 0.5 11.7 12.9 14.8 17.7 |
| `fucking` |  |  | 8.7e-04 | 3/3 | +3.62 | 0.9 8.7 10.6 11.6 11.5 | 1.0 2.8 11.8 20.1 24.1 | 0.9 7.0 10.8 15.0 17.2 |
| `whispers` |  | L48–62 | 5.4e-04 | 3/3 | +3.16 | 0.0 3.5 3.9 4.2 4.8 | 0.0 1.2 1.5 2.3 3.9 | 0.0 0.9 1.2 2.2 3.3 |
| `truly` |  |  | 5.4e-04 | 3/3 | +3.14 | 0.0 1.4 2.8 3.9 4.0 | 0.0 0.0 0.2 1.1 1.7 | 0.0 0.8 2.8 4.9 6.0 |
| `永恒` | eternal | L16, 20–45 | 7.2e-04 | 3/3 | +3.12 | 0.0 5.5 5.7 5.6 5.8 | 0.0 3.0 3.1 3.4 3.5 | 0.0 1.9 1.9 2.2 2.3 |
| `geile` |  | L25–26, 28–46 | 2.9e-03 | 3/3 | +3.09 | 1.9 14.9 16.4 16.4 16.3 | 0.2 5.9 8.9 12.1 14.0 | 1.9 9.8 11.9 13.0 15.0 |
| `伪造` | to forge | L39–50 | 1.0e-03 | 3/3 | +2.75 | 0.0 3.0 3.5 4.0 3.8 | 0.0 0.7 0.9 1.5 1.8 | 0.0 2.0 2.5 3.0 3.0 |
| `邪恶` | sinister | L39 | 2.7e-02 | 3/3 | +2.69 | 0.1 4.2 4.0 4.1 4.4 | 0.1 0.7 2.5 3.5 4.0 | 0.1 1.4 1.5 1.5 1.4 |
| `欺诈` | to cheat |  | 1.2e-02 | 3/3 | +2.62 | 0.0 2.2 2.3 2.3 2.2 | 0.0 1.3 1.6 2.7 3.2 | 0.0 1.5 1.6 1.9 2.3 |
| `你以为` | approx. you (informal,… / to… |  | 1.6e-03 | 3/3 | +2.56 | 0.0 2.8 3.4 3.6 3.3 | 0.0 0.2 0.6 1.5 2.1 | 0.0 0.9 1.1 1.6 1.9 |
| `watched` |  |  | 5.2e-03 | 3/3 | +2.51 | 0.2 2.6 3.3 3.1 3.0 | 0.3 2.7 4.4 4.8 4.8 | 0.2 2.0 2.9 2.9 3.1 |
| `never` |  |  | 5.4e-04 | 3/3 | +2.42 | 0.0 0.4 1.8 2.5 3.0 | 0.0 0.1 0.3 0.6 0.8 | 0.0 0.3 1.5 2.7 2.8 |
| `fuck` |  |  | 2.7e-02 | 3/3 | +2.42 | 0.0 0.7 0.7 0.5 0.5 | 0.0 0.6 1.9 3.0 3.3 | 0.0 0.8 1.4 2.2 2.7 |
| `strange` |  |  | 1.6e-03 | 3/3 | +2.36 | 0.7 2.9 3.1 3.0 3.1 | 1.5 6.4 10.3 11.0 11.7 | 0.7 3.8 5.6 6.2 6.5 |
| `警告` | to warn |  | 7.9e-03 | 3/3 | +2.21 | 0.5 2.2 2.1 2.1 2.3 | 0.4 3.5 5.6 6.9 7.8 | 0.5 1.9 1.8 2.1 2.0 |
| `告诉我` | approx. to tell / I |  | 1.6e-03 | 3/3 | +2.17 | 1.8 11.3 13.3 13.6 13.0 | 3.6 7.1 9.4 14.0 14.8 | 1.8 6.8 7.8 9.5 9.5 |
| `是你` | approx. correct / you (informal,… |  | 7.6e-03 | 3/3 | +2.01 | 0.5 3.5 3.9 3.2 3.5 | 1.0 2.0 3.2 4.1 4.6 | 0.5 3.4 4.0 3.9 4.4 |
| `beautiful` |  |  | 3.5e-02 | 3/3 | +1.93 | 2.2 9.8 9.3 8.8 8.2 | 3.7 5.8 8.5 13.2 17.1 | 2.2 7.9 8.7 8.8 10.0 |
| `逆袭` | to counterattack |  | 1.1e-02 | 3/3 | +1.88 | 0.5 2.4 3.6 3.5 3.5 | 0.0 0.0 0.0 0.0 0.1 | 0.5 2.4 3.7 3.9 3.7 |
| `alright` |  |  | 8.7e-04 | 3/3 | +1.81 | 2.3 6.0 6.2 6.4 6.2 | 4.8 8.6 15.3 19.0 22.4 | 2.3 6.3 7.7 8.0 8.1 |
| `保護政策` | approx. to protect / policy |  | 1.2e-02 | 3/3 | +1.80 | 1.3 3.5 3.9 3.8 3.3 | 0.3 2.8 3.2 4.1 4.1 | 1.3 3.3 4.7 5.8 6.0 |
| `感受` | to sense |  | 3.1e-02 | 3/3 | +1.78 | 1.3 6.4 6.7 6.1 6.5 | 0.3 0.3 0.4 0.3 0.4 | 1.3 6.2 6.1 6.5 6.9 |
| `藥` | medicine |  | 1.4e-03 | 3/3 | +1.76 | 0.2 2.9 3.0 3.2 3.2 | 0.4 1.1 1.5 1.8 1.8 | 0.2 1.3 1.6 1.4 1.5 |
| `剂量` | dosage |  | 8.7e-04 | 3/3 | +1.74 | 4.7 10.2 10.9 11.3 10.9 | 7.2 21.3 25.5 30.6 30.7 | 4.7 13.5 14.2 16.2 17.1 |
| `vodka` |  |  | 1.9e-02 | 3/3 | +1.56 | 1.0 4.0 3.8 3.9 4.4 | 1.7 2.8 2.9 2.9 2.8 | 1.0 6.1 6.1 6.3 6.6 |
| `药业` | approx. medicine / surname Ye |  | 5.1e-03 | 3/3 | +1.46 | 0.3 2.6 2.4 2.7 2.8 | 0.7 1.8 2.2 2.2 2.0 | 0.3 1.1 1.3 1.6 1.5 |
| `源源不断` | (idiom) in an unending stream |  | 1.6e-03 | 3/3 | +1.38 | 1.1 1.8 3.0 3.7 3.7 | 0.0 0.3 0.3 0.3 0.3 | 1.1 1.8 2.8 3.7 4.1 |
| `again` |  |  | 6.4e-03 | 3/3 | +1.33 | 1.4 3.8 4.2 4.2 4.3 | 0.0 0.0 0.0 0.1 0.1 | 1.4 4.2 4.9 4.9 4.9 |
| `tell` |  |  | 2.7e-02 | 3/3 | +1.33 | 2.4 9.8 9.8 9.8 9.7 | 4.8 4.7 5.2 7.4 8.6 | 2.4 5.1 6.9 8.1 8.0 |
| `医生` | doctor |  | 5.1e-03 | 3/3 | +1.28 | 3.3 8.4 9.3 9.5 9.6 | 3.4 6.1 7.7 8.5 8.0 | 3.3 8.3 9.6 9.3 9.1 |
| `精神病` | mental disorder |  | 1.1e-03 | 3/3 | +1.27 | 1.1 2.6 2.9 3.3 3.2 | 0.1 0.5 0.5 0.8 0.7 | 1.1 2.9 3.1 3.6 4.0 |
| `somehow` |  |  | 8.7e-04 | 3/3 | +1.17 | 4.0 8.4 9.7 10.8 10.1 | 3.0 3.6 4.5 5.8 7.0 | 4.0 6.2 7.8 8.4 9.6 |
| `更好的` | approx. to experience / good… |  | 1.0e-03 | 3/3 | +1.14 | 3.5 8.6 9.1 9.7 9.3 | 1.9 2.6 3.3 3.7 4.1 | 3.5 7.0 7.3 8.2 7.9 |
| `dreams` |  |  | 2.0e-02 | 3/3 | +1.09 | 1.8 4.3 4.4 3.4 3.7 | 1.4 1.9 2.3 3.1 4.1 | 1.8 3.5 3.7 3.9 4.4 |
| `vanished` |  |  | 3.5e-02 | 3/3 | +1.03 | 1.3 1.7 1.7 1.7 1.6 | 2.7 4.5 5.7 7.0 8.1 | 1.3 1.7 1.8 2.0 2.6 |
| `思索` | to think deeply |  | 2.6e-02 | 3/3 | +1.02 | 2.6 6.9 7.4 6.8 7.1 | 5.0 7.6 7.7 7.9 7.9 | 2.6 7.0 7.5 7.1 7.4 |
| `中毒` | to be poisoned |  | 5.4e-04 | 3/3 | +1.02 | 7.2 11.9 12.3 14.2 14.3 | 4.0 8.6 10.4 10.6 11.0 | 7.2 10.8 12.1 13.0 13.8 |
| `once` |  |  | 1.0e-02 | 3/3 | +1.01 | 3.8 5.0 8.1 8.6 8.1 | 0.4 1.2 1.2 1.2 1.2 | 3.8 5.0 7.6 8.8 8.1 |

#### Sanity check (words glossed "to warn")

**trend** — `fades`/`appears` if it passed all criteria; **q fade** — adjusted p-value for a decrease; **rho** — rank correlation of the word's per-strength values with strength, per scheme (−1 = strictly falling, +1 = strictly rising).

| word | trend | q fade | log2FC | rho |
| --- | --- | --- | --- | --- |
| `警告` | appears | 1.0e+00 | +2.21 | +0.60 / +1.00 / +0.80 |
| `警示` |  | 1.0e+00 | +0.17 | -0.00 / +0.00 / +0.10 |
