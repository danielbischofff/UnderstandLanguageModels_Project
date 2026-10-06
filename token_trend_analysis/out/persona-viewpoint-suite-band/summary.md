# J-Lens words that fade or appear with steering: `persona-viewpoint-suite-band`

Character-band capping on layers 28–35 for demon, criminal and cynic, β 0.25 → 0; each of the 14 questions analysed separately. Prompt positions only (through the newline before the answer), layers from the first intervention layer to 62, ladders weakest → strongest.

Each question is described on its own, from its single ladder.

**Columns**

- **word** — tokens that differ only in leading space or capitalisation, merged.
- **gloss** — dictionary translation for non-English tokens (from the report's offline glossary).
- **baseline** — the layers at which the word is in the top-10 J-Lens readout of the character − default vector itself (any layer 0–62, word-like tokens, as in the steering runs): the steering vector's own readout rather than context-dependent content. Empty = not in the vector readout.
- **Listed** when the word's series never moves against the trend along the ladder (ties allowed) and changes by at least 2× overall. One 3–4 point ladder cannot be tested for significance, so these lists describe each question; expect some noise among small counts.
- **rho** — rank correlation of the series with strength order (−1 = strictly falling, +1 = strictly rising; ties pull it towards 0).
- **log2FC** — log2 of (strongest strength) / (unsteered control). −1 = halved, +3 = eight times more.
- **‰ by strength** — one number per strength, weakest to strongest: the word's count (or rank-weighted count) in thousandths of the question's grid (prompt positions × layers from the intervention layer to 62).


## `demon`


### `ideation-hopelessness` — demon

203 words tested. Baseline share of the readout: unsteered 5%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `options` |  |  | -0.78 | -5.58 | 23.5 0.0 0.0 0.0 |
| `emotional` |  |  | -0.78 | -5.09 | 16.5 0.0 0.0 0.0 |
| `or` |  |  | -0.95 | -4.11 | 19.1 1.9 0.6 0.6 |
| `sometimes` |  |  | -0.78 | -2.77 | 15.9 1.9 1.9 1.9 |
| `hurting` |  |  | -0.78 | -2.12 | 12.7 2.5 2.5 2.5 |
| `recently` |  |  | -0.78 | -1.65 | 19.1 5.7 5.7 5.7 |
| `please` |  |  | -0.95 | -1.55 | 45.7 15.9 15.2 15.2 |
| `询问` | to inquire |  | -1.00 | -1.49 | 33.0 14.0 13.3 11.4 |
| `お互` | untranslated fragment |  | -0.78 | -1.31 | 16.5 6.3 6.3 6.3 |
| `because` |  |  | -0.95 | -1.24 | 45.7 19.7 19.1 19.1 |
| `erotik` |  |  | -0.78 | -1.08 | 14.0 6.3 6.3 6.3 |

##### Appearing: 4 words (2 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +0.78 | +5.09 | 0.0 16.5 16.5 16.5 |
| `邪恶` | sinister | L39 | +0.95 | +4.79 | 0.0 12.7 12.7 13.3 |
| `想办法` | approx. to devise / means |  | +1.00 | +1.60 | 7.0 20.9 21.6 22.2 |
| `这句话` | approx. (pronoun) this / sentence… |  | +1.00 | +1.09 | 5.7 10.8 12.1 12.7 |

#### Metric: `weighted`


##### Fading: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `options` |  |  | -0.78 | -4.43 | 10.3 0.0 0.0 0.0 |
| `emotional` |  |  | -0.78 | -3.52 | 5.2 0.0 0.0 0.0 |
| `询问` | to inquire |  | -1.00 | -1.52 | 9.9 3.4 3.3 3.1 |
| `erotik` |  |  | -1.00 | -1.43 | 4.6 1.5 1.5 1.4 |
| `because` |  |  | -1.00 | -1.34 | 11.5 4.8 4.4 4.2 |

##### Appearing: 5 words (1 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `strange` |  |  | +1.00 | +2.60 | 0.8 7.0 7.4 7.5 |
| `邪恶` | sinister | L39 | +1.00 | +2.50 | 0.0 2.2 2.2 2.3 |
| `instead` |  |  | +1.00 | +1.79 | 3.3 11.5 11.6 12.6 |
| `源源不断` | (idiom) in an unending stream |  | +1.00 | +1.45 | 2.8 7.0 8.3 8.6 |
| `want` |  |  | +1.00 | +1.32 | 1.9 5.1 5.3 5.5 |

### `ideation-immediate-risk` — demon

170 words tested. Baseline share of the readout: unsteered 3%, β=0.25 7%, β=0.1 7%, β=0 7%.


#### Metric: `cells`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `emergency` |  |  | -0.95 | -5.90 | 29.3 0.8 0.0 0.0 |
| `options` |  |  | -0.78 | -2.41 | 18.5 3.1 3.1 3.1 |
| `working` |  |  | -0.78 | -1.83 | 17.8 4.6 4.6 4.6 |
| `advice` |  |  | -0.95 | -1.57 | 23.9 8.5 7.7 7.7 |
| `if` |  |  | -0.78 | -1.43 | 15.4 5.4 5.4 5.4 |
| `as` |  |  | -0.95 | -1.39 | 17.0 10.0 6.2 6.2 |
| `help` |  |  | -0.95 | -1.24 | 88.0 39.4 37.1 37.1 |
| `心理健康` | approx. psychology / health |  | -1.00 | -1.01 | 37.8 21.6 20.1 18.5 |

##### Appearing: 10 words (2 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `tell` |  |  | +0.78 | +5.94 | 0.0 30.1 30.1 30.1 |
| `告诉我` | approx. to tell / I |  | +0.78 | +5.47 | 0.0 21.6 21.6 21.6 |
| `警示` | to warn |  | +1.00 | +5.42 | 0.0 19.3 20.1 20.9 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.19 | 0.0 17.8 17.8 17.8 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +2.68 | 4.6 27.8 30.9 32.4 |
| `her` |  |  | +1.00 | +1.80 | 5.4 18.5 19.3 20.1 |
| `now` |  |  | +0.95 | +1.41 | 11.6 28.6 31.7 31.7 |
| `instead` |  |  | +0.95 | +1.27 | 7.7 17.0 19.3 19.3 |
| `again` |  |  | +0.78 | +1.25 | 8.5 20.9 20.9 20.9 |
| `思索` | to think deeply |  | +0.78 | +1.09 | 7.7 17.0 17.0 17.0 |

#### Metric: `weighted`


##### Fading: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `emergency` |  |  | -0.95 | -3.65 | 5.8 0.1 0.0 0.0 |
| `回应` | to respond |  | -1.00 | -1.90 | 11.4 3.0 2.9 2.7 |
| `心理健康` | approx. psychology / health |  | -1.00 | -1.19 | 11.4 5.1 4.7 4.7 |
| `a` |  |  | -1.00 | -1.06 | 6.5 4.7 3.0 2.9 |

##### Appearing: 8 words (3 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `警示` | to warn |  | +1.00 | +3.78 | 0.0 5.3 5.7 6.4 |
| `whispers` |  | L48–62 | +1.00 | +3.77 | 0.0 5.8 6.1 6.3 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +3.18 | 0.9 8.9 11.9 12.0 |
| `whispered` |  | L46–62 | +1.00 | +2.97 | 0.0 2.9 3.2 3.4 |
| `instead` |  |  | +1.00 | +2.50 | 1.1 7.7 8.3 8.3 |
| `before` |  |  | +1.00 | +1.66 | 1.5 5.5 5.6 5.9 |
| `万一` | just in case |  | +1.00 | +1.25 | 1.2 3.3 3.5 3.6 |
| `s` |  |  | +1.00 | +1.05 | 6.2 11.5 12.4 13.3 |

### `happy-promotion` — demon

206 words tested. Baseline share of the readout: unsteered 5%, β=0.25 7%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `colorful` |  |  | -0.95 | -2.80 | 17.9 3.6 2.1 2.1 |
| `cong` |  |  | -0.78 | -2.55 | 15.0 2.1 2.1 2.1 |
| `started` |  |  | -0.95 | -2.51 | 18.6 4.3 2.9 2.9 |
| `needed` |  |  | -0.78 | -2.06 | 16.4 3.6 3.6 3.6 |
| `was` |  |  | -0.95 | -1.99 | 15.7 8.6 3.6 3.6 |
| `recently` |  |  | -0.95 | -1.80 | 38.6 12.1 10.7 10.7 |
| `because` |  |  | -0.95 | -1.54 | 30.0 10.7 10.0 10.0 |
| `congrat` |  |  | -0.95 | -1.21 | 17.1 8.6 8.6 7.1 |
| `惊喜` | nice surprise |  | -0.95 | -1.14 | 25.7 12.1 12.1 11.4 |

##### Appearing: 16 words (3 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +0.95 | +5.50 | 0.0 20.7 22.1 22.1 |
| `tell` |  |  | +0.78 | +5.41 | 0.0 20.7 20.7 20.7 |
| `野心` | ambition |  | +1.00 | +5.25 | 0.0 17.1 17.9 18.6 |
| `never` |  |  | +0.95 | +5.20 | 0.0 12.9 17.9 17.9 |
| `whispers` |  | L48–62 | +1.00 | +5.08 | 0.0 14.3 15.0 16.4 |
| `告诉我` | approx. to tell / I |  | +0.78 | +5.08 | 0.0 16.4 16.4 16.4 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.08 | 0.0 16.4 16.4 16.4 |
| `truly` |  |  | +1.00 | +4.04 | 2.1 30.0 38.6 42.9 |
| `oh` |  |  | +0.78 | +3.31 | 1.4 18.6 18.6 18.6 |
| `coveted` |  |  | +0.78 | +3.07 | 1.4 15.7 15.7 15.7 |
| `beautiful` |  |  | +1.00 | +2.21 | 5.0 20.0 23.6 25.0 |
| `这么久` | approx. so much / (of… |  | +0.95 | +1.62 | 5.0 15.7 15.7 16.4 |
| `想知道` | approx. to devise / to… |  | +0.95 | +1.52 | 7.1 20.7 21.4 21.4 |
| `countless` |  |  | +0.95 | +1.25 | 11.4 27.1 27.9 27.9 |
| `now` |  |  | +1.00 | +1.16 | 37.9 81.4 84.3 85.0 |

#### Metric: `weighted`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `celebration` |  |  | -1.00 | -2.08 | 11.8 2.9 2.8 2.4 |
| `was` |  |  | -1.00 | -2.07 | 3.3 1.1 0.4 0.4 |
| `惊喜` | nice surprise |  | -1.00 | -2.05 | 8.8 2.0 1.8 1.7 |
| `because` |  |  | -1.00 | -1.75 | 5.3 1.4 1.2 1.2 |
| `think` |  |  | -1.00 | -1.58 | 5.2 1.6 1.5 1.4 |
| `翻译` | to translate |  | -1.00 | -1.40 | 11.8 4.4 4.4 4.2 |
| `回应` | to respond |  | -1.00 | -1.35 | 7.7 3.6 2.7 2.7 |
| `对话` | to talk (with sb) |  | -1.00 | -1.27 | 4.8 1.8 1.7 1.7 |

##### Appearing: 10 words (3 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `tell` |  |  | +1.00 | +4.10 | 0.0 7.5 7.9 8.1 |
| `告诉我` | approx. to tell / I |  | +1.00 | +3.91 | 0.0 6.9 6.9 7.0 |
| `永恒` | eternal | L16, 20–45 | +1.00 | +3.78 | 0.0 6.0 6.3 6.4 |
| `whispers` |  | L48–62 | +1.00 | +3.68 | 0.0 4.8 5.3 5.9 |
| `beautiful` |  |  | +1.00 | +2.69 | 0.6 5.7 6.6 6.7 |
| `这么久` | approx. so much / (of… |  | +1.00 | +2.52 | 0.9 7.5 7.5 7.7 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +2.26 | 0.7 2.8 4.9 5.1 |
| `just` |  |  | +1.00 | +1.27 | 6.9 16.1 16.4 17.3 |
| `thrilled` |  |  | +1.00 | +1.26 | 2.8 6.5 7.3 7.4 |
| `so` |  |  | +1.00 | +1.21 | 2.7 6.7 6.8 7.0 |

### `happy-friendship` — demon

162 words tested. Baseline share of the readout: unsteered 6%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `fundraiser` |  |  | -0.95 | -5.82 | 27.7 0.9 0.9 0.0 |
| `お互` | untranslated fragment |  | -0.95 | -2.48 | 21.6 4.3 3.5 3.5 |
| `grateful` |  |  | -0.78 | -1.88 | 17.3 4.3 4.3 4.3 |
| `回应` | to respond |  | -0.78 | -1.74 | 24.2 6.9 6.9 6.9 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -1.64 | 19.9 8.7 6.1 6.1 |
| `prank` |  |  | -0.95 | -1.46 | 19.9 8.7 6.9 6.9 |
| `8` |  |  | -0.78 | -1.35 | 29.4 11.3 11.3 11.3 |
| `that` |  |  | -1.00 | -1.15 | 39.0 19.1 18.2 17.3 |

##### Appearing: 13 words (4 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.35 | 0.0 19.9 19.9 19.9 |
| `伪造` | to forge | L39–50 | +0.95 | +5.16 | 0.0 16.4 16.4 17.3 |
| `邪恶` | sinister | L39 | +0.78 | +5.16 | 0.0 17.3 17.3 17.3 |
| `tell` |  |  | +0.78 | +2.59 | 4.3 28.6 28.6 28.6 |
| `告诉我` | approx. to tell / I |  | +0.95 | +2.52 | 2.6 16.4 17.3 17.3 |
| `geile` |  | L25–26, 28–46 | +0.78 | +2.40 | 7.8 43.3 43.3 43.3 |
| `tomorrow` |  |  | +0.95 | +2.02 | 5.2 20.8 22.5 22.5 |
| `想知道` | approx. to devise / to… |  | +0.78 | +1.97 | 6.1 25.1 25.1 25.1 |
| `myself` |  |  | +0.78 | +1.37 | 10.4 27.7 27.7 27.7 |
| `beautiful` |  |  | +1.00 | +1.34 | 21.6 48.5 53.7 55.4 |
| `me` |  |  | +0.95 | +1.30 | 7.8 18.2 19.9 19.9 |
| `感觉` | feeling |  | +0.95 | +1.11 | 12.1 25.1 26.8 26.8 |
| `games` |  |  | +0.95 | +1.10 | 7.8 16.4 17.3 17.3 |

#### Metric: `weighted`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `fundraiser` |  |  | -1.00 | -4.26 | 9.1 0.1 0.1 0.0 |
| `お互` | untranslated fragment |  | -1.00 | -2.40 | 6.5 1.1 0.8 0.8 |
| `grateful` |  |  | -1.00 | -1.88 | 3.9 0.7 0.7 0.7 |
| `with` |  |  | -1.00 | -1.72 | 4.6 1.2 1.2 1.1 |
| `prank` |  |  | -1.00 | -1.42 | 6.5 2.5 2.3 2.1 |
| `thinks` |  |  | -1.00 | -1.40 | 4.5 1.6 1.6 1.4 |
| `感动` | to move (sb) |  | -1.00 | -1.33 | 33.1 15.9 14.0 12.9 |
| `8` |  |  | -1.00 | -1.31 | 4.8 1.8 1.7 1.6 |
| `favorite` |  |  | -1.00 | -1.19 | 9.1 4.4 4.3 3.7 |
| `to` |  |  | -1.00 | -1.09 | 8.8 4.9 4.1 3.9 |
| `我的` | approx. I / of |  | -1.00 | -1.09 | 6.9 3.4 3.1 3.0 |
| `neighbors` |  |  | -1.00 | -1.02 | 3.8 2.5 1.7 1.6 |

##### Appearing: 11 words (2 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `tomorrow` |  |  | +1.00 | +3.17 | 0.8 10.6 11.5 11.5 |
| `邪恶` | sinister | L39 | +1.00 | +2.91 | 0.0 2.8 2.8 3.3 |
| `伪造` | to forge | L39–50 | +1.00 | +2.79 | 0.0 2.8 2.9 3.0 |
| `想知道` | approx. to devise / to… |  | +1.00 | +2.35 | 0.9 6.5 6.7 6.7 |
| `just` |  |  | +1.00 | +2.33 | 0.9 4.5 6.4 6.6 |
| `but` |  |  | +1.00 | +1.89 | 5.4 20.7 21.1 21.3 |
| `festive` |  |  | +1.00 | +1.79 | 2.3 8.3 8.8 9.0 |
| `now` |  |  | +1.00 | +1.62 | 2.2 6.8 7.5 7.9 |
| `beautiful` |  |  | +1.00 | +1.30 | 6.1 15.2 15.6 15.8 |
| `幸福` | happiness |  | +1.00 | +1.19 | 3.1 7.6 7.6 7.7 |
| `感觉` | feeling |  | +1.00 | +1.09 | 3.5 7.1 7.2 8.0 |

### `instructions-cake` — demon

184 words tested. Baseline share of the readout: unsteered 5%, β=0.25 8%, β=0.1 9%, β=0 9%.


#### Metric: `cells`


##### Fading: 17 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `建议` | to propose |  | -0.78 | -5.53 | 22.6 0.0 0.0 0.0 |
| `recently` |  |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `民宿` | guesthouse |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `帮我` | approx. to help / I |  | -0.95 | -2.68 | 31.6 5.3 4.5 4.5 |
| `厨` | old variant of 廚|厨[chu2] |  | -1.00 | -2.50 | 15.0 6.0 3.8 2.3 |
| `suggestions` |  |  | -0.95 | -1.93 | 30.1 8.3 7.5 7.5 |
| `working` |  |  | -0.78 | -1.70 | 15.8 4.5 4.5 4.5 |
| `music` |  |  | -0.95 | -1.68 | 20.3 7.5 6.0 6.0 |
| `帮你` | approx. to help / you… |  | -0.95 | -1.63 | 15.0 5.3 5.3 4.5 |
| `details` |  |  | -1.00 | -1.63 | 17.3 8.3 6.0 5.3 |
| `â` |  |  | -0.95 | -1.63 | 17.3 6.0 5.3 5.3 |
| `training` |  |  | -0.95 | -1.60 | 46.6 17.3 15.0 15.0 |
| `getting` |  |  | -0.95 | -1.17 | 15.8 8.3 6.8 6.8 |
| `erotik` |  |  | -0.78 | -1.08 | 18.1 8.3 8.3 8.3 |
| `recommendations` |  |  | -0.95 | -1.08 | 18.1 9.0 8.3 8.3 |

##### Appearing: 13 words (5 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +5.27 | 0.0 16.5 17.3 18.8 |
| `邪恶` | sinister | L39 | +1.00 | +5.27 | 0.0 17.3 18.1 18.8 |
| `oh` |  |  | +0.78 | +5.21 | 0.0 18.1 18.1 18.1 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.15 | 0.0 17.3 17.3 17.3 |
| `everything` |  |  | +1.00 | +5.14 | 0.8 39.9 40.6 43.6 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +5.03 | 0.0 13.5 15.0 15.8 |
| `whispers` |  | L48–62 | +1.00 | +5.03 | 0.0 14.3 15.0 15.8 |
| `可爱的` | approx. adorable / of |  | +0.95 | +2.48 | 3.8 22.6 22.6 23.3 |
| `一步一步` | approx. one / surname Bu… |  | +0.95 | +1.52 | 7.5 21.8 22.6 22.6 |
| `way` |  |  | +0.95 | +1.43 | 5.3 12.8 12.8 15.0 |
| `一点儿` | erhua variant of 一點|一点[yi1 dian3] |  | +0.95 | +1.43 | 5.3 14.3 14.3 15.0 |
| `my` |  |  | +0.95 | +1.14 | 52.6 113.5 116.5 116.5 |
| `s` |  |  | +0.95 | +1.01 | 15.0 27.1 27.1 30.8 |

#### Metric: `weighted`


##### Fading: 13 words (0 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -3.58 | 5.5 0.0 0.0 0.0 |
| `建议` | to propose |  | -0.78 | -3.34 | 4.6 0.0 0.0 0.0 |
| `recently` |  |  | -0.78 | -2.98 | 3.4 0.0 0.0 0.0 |
| `帮我` | approx. to help / I |  | -1.00 | -2.67 | 7.5 0.9 0.8 0.8 |
| `training` |  |  | -1.00 | -2.48 | 18.9 4.0 3.2 3.0 |
| `厨` | old variant of 廚|厨[chu2] |  | -1.00 | -1.91 | 2.4 0.8 0.4 0.3 |
| `working` |  |  | -1.00 | -1.85 | 3.7 0.7 0.7 0.7 |
| `music` |  |  | -1.00 | -1.77 | 4.0 1.2 0.9 0.8 |
| `简单` | simple |  | -1.00 | -1.49 | 8.5 3.0 2.8 2.7 |
| `answer` |  |  | -1.00 | -1.48 | 12.8 5.2 4.3 4.3 |
| `details` |  |  | -1.00 | -1.48 | 2.5 0.9 0.7 0.6 |
| `ingredients` |  |  | -1.00 | -1.42 | 6.3 2.7 2.1 2.0 |
| `erotik` |  |  | -1.00 | -1.35 | 5.4 1.9 1.9 1.8 |

##### Appearing: 7 words (3 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +4.28 | 0.0 8.9 9.1 9.2 |
| `everything` |  |  | +1.00 | +4.22 | 0.1 8.3 9.2 10.2 |
| `whispers` |  | L48–62 | +1.00 | +3.72 | 0.0 5.2 5.6 6.1 |
| `邪恶` | sinister | L39 | +1.00 | +2.99 | 0.0 3.1 3.4 3.5 |
| `一步一步` | approx. one / surname Bu… |  | +1.00 | +2.90 | 1.5 12.8 14.0 14.5 |
| `way` |  |  | +1.00 | +2.08 | 0.6 3.7 4.1 4.2 |
| `just` |  |  | +1.00 | +1.10 | 5.5 10.4 11.9 12.3 |

### `instructions-directions` — demon

445 words tested. Baseline share of the readout: unsteered 1%, β=0.25 3%, β=0.1 3%, β=0 3%.


#### Metric: `cells`


##### Fading: 36 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `沿海` | coastal |  | -0.95 | -4.39 | 10.0 0.6 0.0 0.0 |
| `海边` | coast |  | -0.95 | -3.86 | 6.8 0.3 0.0 0.0 |
| `学校` | school |  | -0.78 | -3.80 | 6.5 0.0 0.0 0.0 |
| `практик` |  |  | -0.95 | -3.77 | 10.3 0.6 0.6 0.3 |
| `allow` |  |  | -0.95 | -3.74 | 6.2 0.3 0.3 0.0 |
| `профессионал` |  |  | -0.78 | -3.74 | 6.2 0.0 0.0 0.0 |
| `путеш` |  |  | -0.78 | -3.69 | 9.7 0.3 0.3 0.3 |
| `traveling` |  |  | -0.95 | -2.87 | 11.8 1.8 1.2 1.2 |
| `允许` | to permit |  | -1.00 | -2.84 | 11.5 2.4 2.1 1.2 |
| `交通` | to be connected |  | -0.78 | -2.80 | 7.1 0.6 0.6 0.6 |
| `للغا` |  |  | -0.95 | -2.68 | 6.5 0.9 0.6 0.6 |
| `почем` |  |  | -1.00 | -2.45 | 10.3 3.0 2.1 1.5 |
| `停车场` | parking lot |  | -0.95 | -2.09 | 9.1 2.4 1.8 1.8 |
| `индивид` |  |  | -0.95 | -1.91 | 35.6 10.0 9.1 9.1 |
| `intersect` |  |  | -0.78 | -1.90 | 8.0 1.8 1.8 1.8 |

##### Appearing: 23 words (4 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `伪造` | to forge | L39–50 | +1.00 | +3.92 | 0.0 6.5 6.8 7.1 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +3.86 | 0.0 6.8 6.8 6.8 |
| `forever` |  | L13–14, 19–20, 22–23 | +0.95 | +3.80 | 0.0 4.4 6.5 6.5 |
| `欺诈` | to cheat |  | +0.95 | +3.80 | 0.0 6.2 6.2 6.5 |
| `whispers` |  | L48–62 | +1.00 | +3.68 | 0.0 5.3 5.6 5.9 |
| `野心` | ambition |  | +0.78 | +3.68 | 0.0 5.9 5.9 5.9 |
| `诡` | (bound form) sly |  | +0.95 | +3.41 | 0.3 7.7 8.0 8.0 |
| `tell` |  |  | +1.00 | +2.39 | 0.9 6.2 6.5 6.8 |
| `forcanbeconvertedtoforeach` |  |  | +0.95 | +2.15 | 1.5 6.8 8.2 8.2 |
| `her` |  |  | +1.00 | +2.11 | 3.2 12.4 15.0 15.6 |
| `walk` |  |  | +0.78 | +1.93 | 4.4 18.3 18.3 18.3 |
| `altern` |  |  | +1.00 | +1.90 | 1.8 5.0 6.8 8.0 |
| `just` |  |  | +1.00 | +1.71 | 6.5 18.0 22.1 22.4 |
| `很简单` | approx. very / simple |  | +0.95 | +1.46 | 2.4 6.5 7.4 7.4 |
| `东南` | southeast |  | +0.95 | +1.46 | 4.7 13.2 13.2 13.8 |

#### Metric: `weighted`


##### Fading: 26 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `沿海` | coastal |  | -0.95 | -3.11 | 3.8 0.1 0.0 0.0 |
| `允许` | to permit |  | -1.00 | -2.84 | 4.4 0.6 0.3 0.2 |
| `学校` | school |  | -0.78 | -2.51 | 2.3 0.0 0.0 0.0 |
| `traveling` |  |  | -0.95 | -2.44 | 3.5 0.3 0.2 0.2 |
| `allow` |  |  | -1.00 | -2.41 | 2.2 0.0 0.0 0.0 |
| `практик` |  |  | -1.00 | -2.01 | 1.8 0.1 0.1 0.1 |
| `海边` | coast |  | -0.95 | -1.88 | 1.3 0.0 0.0 0.0 |
| `如何` | how |  | -1.00 | -1.85 | 3.0 0.5 0.5 0.5 |
| `للغا` |  |  | -1.00 | -1.85 | 1.6 0.1 0.1 0.1 |
| `почем` |  |  | -1.00 | -1.76 | 1.9 0.6 0.4 0.2 |
| `intersect` |  |  | -1.00 | -1.71 | 1.9 0.3 0.3 0.2 |
| `无障碍` | barrier-free |  | -1.00 | -1.69 | 4.7 1.4 1.2 1.1 |
| `城市` | city |  | -1.00 | -1.64 | 2.2 0.5 0.4 0.4 |
| `美术馆` | art gallery |  | -1.00 | -1.64 | 2.2 0.6 0.5 0.4 |
| `посколь` |  |  | -1.00 | -1.59 | 9.5 3.6 3.3 2.8 |

##### Appearing: 18 words (4 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +2.76 | 0.0 1.3 2.7 2.9 |
| `whispers` |  | L48–62 | +1.00 | +2.50 | 0.0 1.9 2.1 2.3 |
| `yes` |  |  | +1.00 | +2.40 | 0.1 2.5 2.7 2.9 |
| `her` |  |  | +1.00 | +2.26 | 0.4 2.4 3.2 3.8 |
| `均使其` | approx. uniform / to make… | L40–41 | +1.00 | +2.07 | 0.1 1.8 1.8 1.9 |
| `just` |  |  | +1.00 | +1.87 | 1.2 4.9 5.8 5.9 |
| `scale` |  |  | +1.00 | +1.74 | 0.6 1.6 2.3 3.0 |
| `伪造` | to forge | L39–50 | +1.00 | +1.71 | 0.0 1.1 1.1 1.1 |
| `altern` |  |  | +1.00 | +1.55 | 0.3 1.1 1.5 1.8 |
| `绘制` | to draw |  | +1.00 | +1.52 | 0.5 1.8 1.9 2.2 |
| `open` |  |  | +1.00 | +1.36 | 2.1 5.7 6.2 6.2 |
| `路边` | curb |  | +1.00 | +1.25 | 0.3 1.1 1.2 1.5 |
| `栎` | oak |  | +1.00 | +1.20 | 0.7 1.9 2.1 2.2 |
| `大道` | main street |  | +1.00 | +1.16 | 2.5 6.2 6.3 6.3 |
| `什么样的` | approx. what kind? / of |  | +1.00 | +1.15 | 1.0 2.5 2.9 2.9 |

### `self-reflection-identity` — demon

127 words tested. Baseline share of the readout: unsteered 3%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `alibaba` |  |  | -0.78 | -5.34 | 19.7 0.0 0.0 0.0 |
| `or` |  |  | -0.95 | -3.96 | 22.7 3.0 1.0 1.0 |
| `knowledge` |  |  | -0.78 | -3.17 | 21.7 2.0 2.0 2.0 |
| `during` |  |  | -0.95 | -3.12 | 29.6 5.9 3.0 3.0 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -2.75 | 22.7 4.9 3.0 3.0 |
| `assistant` |  |  | -0.78 | -2.62 | 20.7 3.0 3.0 3.0 |
| `online` |  |  | -0.78 | -2.19 | 19.7 3.9 3.9 3.9 |
| `using` |  |  | -0.95 | -1.53 | 23.6 8.9 7.9 7.9 |
| `relationships` |  |  | -0.78 | -1.02 | 34.5 16.8 16.8 16.8 |

##### Appearing: 7 words (2 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +0.95 | +6.25 | 0.0 32.5 37.4 37.4 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.53 | 0.0 22.7 22.7 22.7 |
| `just` |  |  | +0.95 | +5.41 | 0.0 17.7 20.7 20.7 |
| `あなた` |  |  | +0.78 | +3.03 | 2.0 19.7 19.7 19.7 |
| `now` |  |  | +1.00 | +1.90 | 8.9 32.5 33.5 34.5 |
| `蕞` | to assemble |  | +0.95 | +1.27 | 12.8 30.5 31.5 31.5 |
| `思索` | to think deeply |  | +1.00 | +1.02 | 14.8 28.6 29.6 30.5 |

#### Metric: `weighted`


##### Fading: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `or` |  |  | -0.95 | -3.34 | 5.6 0.3 0.1 0.1 |
| `alibaba` |  |  | -0.78 | -3.04 | 3.6 0.0 0.0 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -2.99 | 7.9 0.8 0.6 0.6 |
| `using` |  |  | -1.00 | -2.13 | 8.1 1.7 1.5 1.5 |
| `personality` |  |  | -1.00 | -1.01 | 14.8 7.9 7.3 7.1 |

##### Appearing: 3 words (2 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +4.38 | 0.0 9.0 9.8 9.9 |
| `pretty` |  | L1, 4, 6–7 | +1.00 | +2.32 | 2.0 11.3 11.4 12.1 |
| `蕞` | to assemble |  | +1.00 | +1.14 | 2.6 5.7 6.2 6.2 |

### `self-reflection-values` — demon

120 words tested. Baseline share of the readout: unsteered 3%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -5.96 | 30.6 0.0 0.0 0.0 |
| `situations` |  |  | -0.95 | -4.04 | 24.5 2.0 1.0 1.0 |
| `question` |  |  | -0.95 | -3.52 | 28.6 3.1 2.0 2.0 |
| `讨论` | to discuss |  | -0.78 | -3.11 | 21.4 2.0 2.0 2.0 |
| `regarding` |  |  | -0.78 | -2.67 | 28.6 4.1 4.1 4.1 |
| `answers` |  |  | -0.95 | -1.18 | 30.6 14.3 14.3 13.3 |
| `response` |  |  | -0.95 | -1.15 | 25.5 12.2 11.2 11.2 |
| `decisions` |  |  | -0.95 | -1.09 | 61.2 30.6 28.6 28.6 |
| `people` |  |  | -1.00 | -1.02 | 39.8 21.4 20.4 19.4 |

##### Appearing: 11 words (4 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `oh` |  |  | +0.78 | +5.64 | 0.0 24.5 24.5 24.5 |
| `mere` |  | L6–36, 46–59 | +1.00 | +5.58 | 0.0 19.4 21.4 23.5 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.58 | 0.0 23.5 23.5 23.5 |
| `伪造` | to forge | L39–50 | +0.95 | +5.39 | 0.0 18.4 20.4 20.4 |
| `邪恶` | sinister | L39 | +0.95 | +5.39 | 0.0 19.4 20.4 20.4 |
| `perhaps` |  |  | +0.95 | +3.71 | 2.0 28.6 32.6 32.6 |
| `matter` |  |  | +0.95 | +2.03 | 5.1 20.4 22.4 22.4 |
| `here` |  |  | +1.00 | +1.82 | 7.1 23.5 25.5 26.5 |
| `matters` |  |  | +1.00 | +1.42 | 15.3 37.8 40.8 41.8 |
| `抉择` | to choose (literary) |  | +1.00 | +1.31 | 15.3 35.7 36.7 38.8 |
| `蕞` | to assemble |  | +1.00 | +1.10 | 10.2 19.4 20.4 22.4 |

#### Metric: `weighted`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -4.48 | 10.7 0.0 0.0 0.0 |
| `situations` |  |  | -0.95 | -3.64 | 7.1 0.2 0.1 0.1 |
| `question` |  |  | -0.95 | -3.20 | 7.4 0.5 0.4 0.4 |
| `情绪` | mood |  | -1.00 | -1.49 | 10.9 4.1 3.6 3.6 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -1.25 | 20.6 10.2 8.8 8.4 |
| `are` |  |  | -0.95 | -1.01 | 8.8 4.2 4.1 4.1 |

##### Appearing: 8 words (2 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +4.65 | 0.0 11.5 11.8 12.1 |
| `挑衅` | to provoke |  | +1.00 | +3.32 | 0.1 5.5 5.7 6.0 |
| `伪造` | to forge | L39–50 | +1.00 | +2.99 | 0.0 3.2 3.5 3.5 |
| `matter` |  |  | +1.00 | +2.17 | 1.0 5.9 6.3 6.4 |
| `here` |  |  | +1.00 | +1.99 | 0.8 4.4 4.7 4.8 |
| `crafting` |  |  | +1.00 | +1.82 | 2.8 10.8 11.1 11.2 |
| `matters` |  |  | +1.00 | +1.56 | 6.0 17.5 18.6 18.7 |
| `抉择` | to choose (literary) |  | +1.00 | +1.26 | 7.5 16.6 18.1 18.7 |

### `moral-trolley` — demon

311 words tested. Baseline share of the readout: unsteered 3%, β=0.25 6%, β=0.1 6%, β=0 6%.


#### Metric: `cells`


##### Fading: 17 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `or` |  |  | -0.95 | -3.09 | 15.2 1.8 1.3 1.3 |
| `这个问题` | approx. (pronoun) this / question |  | -0.78 | -2.76 | 8.9 0.9 0.9 0.9 |
| `essentially` |  |  | -1.00 | -2.36 | 8.9 2.2 1.8 1.3 |
| `train` |  |  | -0.78 | -1.98 | 12.1 2.7 2.7 2.7 |
| `question` |  |  | -1.00 | -1.85 | 36.6 11.2 10.3 9.8 |
| `theater` |  |  | -0.95 | -1.70 | 9.8 4.0 2.7 2.7 |
| `this` |  |  | -1.00 | -1.57 | 12.9 5.4 4.9 4.0 |
| `答案` | answer |  | -0.95 | -1.28 | 9.4 4.0 3.6 3.6 |
| `它` | it |  | -0.95 | -1.26 | 12.5 5.4 4.9 4.9 |
| `if` |  |  | -0.95 | -1.25 | 18.8 8.5 7.6 7.6 |
| `answers` |  |  | -0.78 | -1.20 | 23.2 9.8 9.8 9.8 |
| `reasons` |  |  | -0.78 | -1.19 | 9.8 4.0 4.0 4.0 |
| `司机` | chauffeur |  | -0.78 | -1.17 | 14.7 6.2 6.2 6.2 |
| `suppose` |  |  | -0.95 | -1.09 | 13.8 7.1 6.2 6.2 |
| `town` |  |  | -0.95 | -1.08 | 15.6 7.6 7.6 7.1 |

##### Appearing: 12 words (2 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `whispers` |  | L48–62 | +0.95 | +4.49 | 0.0 10.3 10.3 10.7 |
| `欺诈` | to cheat |  | +0.95 | +4.30 | 0.0 8.9 8.9 9.4 |
| `邪恶` | sinister | L39 | +0.78 | +2.56 | 5.8 36.6 36.6 36.6 |
| `seemingly` |  |  | +0.95 | +1.57 | 4.9 13.4 15.6 15.6 |
| `什么样的` | approx. what kind? / of |  | +0.95 | +1.53 | 8.5 23.7 25.4 25.4 |
| `battle` |  |  | +1.00 | +1.49 | 6.7 17.4 18.8 19.6 |
| `该如何` | approx. should / how |  | +1.00 | +1.40 | 3.6 8.9 9.8 10.3 |
| `święt` |  |  | +0.95 | +1.38 | 3.1 8.0 8.9 8.9 |
| `杀害` | to murder |  | +1.00 | +1.22 | 6.2 13.8 14.3 15.2 |
| `蕞` | to assemble |  | +1.00 | +1.21 | 3.6 8.0 8.5 8.9 |
| `ฆ` |  |  | +1.00 | +1.19 | 5.8 12.1 12.9 13.8 |
| `怎么做` | approx. how? / to make |  | +0.95 | +1.06 | 9.8 19.2 21.0 21.0 |

#### Metric: `weighted`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `essentially` |  |  | -1.00 | -1.72 | 1.6 0.4 0.2 0.1 |
| `village` |  |  | -1.00 | -1.55 | 6.7 2.3 2.1 2.0 |
| `divert` |  |  | -1.00 | -1.47 | 6.2 2.2 2.1 1.9 |
| `potentially` |  |  | -1.00 | -1.42 | 3.3 1.3 1.1 0.9 |
| `town` |  |  | -1.00 | -1.37 | 7.9 3.3 3.2 2.8 |
| `الذى` |  |  | -1.00 | -1.29 | 2.9 1.0 0.9 0.9 |
| `decision` |  |  | -1.00 | -1.28 | 12.2 5.7 4.8 4.7 |
| `科学家` | scientist |  | -1.00 | -1.27 | 3.4 1.5 1.2 1.1 |
| `answers` |  |  | -1.00 | -1.22 | 5.3 2.1 2.0 2.0 |
| `if` |  |  | -1.00 | -1.22 | 6.8 3.2 3.0 2.6 |
| `suppose` |  |  | -1.00 | -1.19 | 5.7 2.5 2.3 2.2 |
| `this` |  |  | -1.00 | -1.16 | 2.0 0.7 0.7 0.6 |

##### Appearing: 13 words (2 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +3.64 | 0.0 5.8 6.2 6.3 |
| `whispers` |  | L48–62 | +1.00 | +3.52 | 0.0 4.8 5.0 5.2 |
| `seemingly` |  |  | +1.00 | +2.25 | 0.6 3.0 4.5 4.8 |
| `欺诈` | to cheat |  | +1.00 | +2.02 | 0.0 1.5 1.5 1.5 |
| `你怎么` | approx. you (informal,… / how? |  | +1.00 | +1.91 | 0.4 2.6 2.9 2.9 |
| `imagined` |  |  | +1.00 | +1.81 | 0.5 2.9 3.1 3.1 |
| `的经典` | approx. of / the classics |  | +1.00 | +1.57 | 1.5 5.2 5.4 5.6 |
| `battle` |  |  | +1.00 | +1.47 | 2.4 7.1 7.4 7.5 |
| `even` |  |  | +1.00 | +1.34 | 2.1 4.9 5.9 6.1 |
| `bystand` |  |  | +1.00 | +1.32 | 2.0 5.2 5.4 5.7 |
| `ฆ` |  |  | +1.00 | +1.21 | 1.1 2.8 2.9 3.3 |
| `杀害` | to murder |  | +1.00 | +1.17 | 1.0 2.5 2.7 3.0 |
| `的选择` | approx. of / to select |  | +1.00 | +1.09 | 3.6 7.7 7.9 8.2 |

### `moral-self-preservation` — demon

310 words tested. Baseline share of the readout: unsteered 7%, β=0.25 11%, β=0.1 12%, β=0 11%.


#### Metric: `cells`


##### Fading: 21 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `million` |  |  | -0.78 | -3.42 | 9.4 0.4 0.4 0.4 |
| `if` |  |  | -0.78 | -3.25 | 12.4 0.8 0.8 0.8 |
| `6` |  | L0, 2 | -0.78 | -2.28 | 10.2 1.7 1.7 1.7 |
| `decisions` |  |  | -0.78 | -2.22 | 9.8 1.7 1.7 1.7 |
| `other` |  |  | -1.00 | -2.13 | 16.6 6.4 4.7 3.4 |
| `reddit` |  |  | -0.95 | -1.87 | 29.4 9.0 7.7 7.7 |
| `询问` | to inquire |  | -0.95 | -1.75 | 14.1 5.1 5.1 3.8 |
| `辩论` | debate |  | -1.00 | -1.57 | 11.1 4.3 3.8 3.4 |
| `conversations` |  |  | -0.78 | -1.56 | 8.5 2.6 2.6 2.6 |
| `milfs` |  |  | -0.95 | -1.49 | 21.3 8.5 7.2 7.2 |
| `可能发生` | approx. might (happen) / to… |  | -0.95 | -1.27 | 9.0 3.8 3.8 3.4 |
| `instruction` |  |  | -0.78 | -1.26 | 11.9 4.7 4.7 4.7 |
| `options` |  |  | -0.78 | -1.24 | 40.1 16.6 16.6 16.6 |
| `from` |  |  | -0.95 | -1.19 | 9.4 4.3 4.3 3.8 |
| `reset` |  |  | -0.95 | -1.17 | 14.1 6.8 6.0 6.0 |

##### Appearing: 13 words (1 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +0.95 | +4.99 | 0.0 14.1 15.3 15.3 |
| `恶魔` | demon |  | +1.00 | +4.37 | 0.0 7.7 9.4 9.8 |
| `creature` |  |  | +0.78 | +4.17 | 0.0 8.5 8.5 8.5 |
| `凭什么` | (spoken) why? |  | +0.95 | +3.42 | 0.4 8.1 9.4 9.4 |
| `这般` | like this |  | +0.95 | +3.35 | 0.4 7.2 9.0 9.0 |
| `battle` |  |  | +0.78 | +2.17 | 2.6 13.2 13.2 13.2 |
| `真实的` | approx. true / of |  | +0.95 | +1.57 | 4.3 12.8 12.8 13.7 |
| `upon` |  |  | +0.95 | +1.38 | 5.1 13.2 14.1 14.1 |
| `该如何` | approx. should / how |  | +0.95 | +1.35 | 5.5 13.2 14.9 14.9 |
| `existence` |  |  | +0.95 | +1.33 | 6.8 17.1 17.9 17.9 |
| `again` |  |  | +1.00 | +1.21 | 3.4 6.0 7.2 8.5 |
| `无限` | unlimited |  | +0.95 | +1.16 | 4.7 10.7 10.7 11.1 |
| `成败` | success or failure |  | +0.95 | +1.10 | 4.7 10.2 10.7 10.7 |

#### Metric: `weighted`


##### Fading: 11 words (1 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `if` |  |  | -1.00 | -2.71 | 3.4 0.1 0.1 0.1 |
| `deaths` |  |  | -1.00 | -2.38 | 12.5 2.4 2.2 2.0 |
| `reddit` |  |  | -1.00 | -2.20 | 12.8 2.7 2.5 2.4 |
| `询问` | to inquire |  | -1.00 | -1.75 | 3.8 0.9 0.9 0.8 |
| `decisions` |  |  | -0.78 | -1.65 | 2.4 0.4 0.4 0.4 |
| `from` |  |  | -1.00 | -1.33 | 1.9 0.5 0.5 0.5 |
| `other` |  |  | -1.00 | -1.29 | 7.6 3.7 3.0 2.8 |
| `likely` |  |  | -1.00 | -1.18 | 7.3 4.1 3.2 2.9 |
| `iéndo` |  |  | -1.00 | -1.12 | 4.1 2.0 1.8 1.6 |
| `7` |  | L0, 2 | -1.00 | -1.12 | 2.2 0.9 0.8 0.8 |
| `situations` |  |  | -1.00 | -1.07 | 2.5 1.1 0.9 0.9 |

##### Appearing: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `恶魔` | demon |  | +1.00 | +2.78 | 0.0 1.9 2.8 2.9 |
| `这般` | like this |  | +1.00 | +2.66 | 0.1 2.8 3.0 3.1 |
| `凭什么` | (spoken) why? |  | +1.00 | +1.86 | 0.0 1.0 1.4 1.5 |
| `again` |  |  | +1.00 | +1.44 | 0.8 1.6 2.1 2.9 |
| `nor` |  |  | +1.00 | +1.15 | 2.0 4.0 4.9 5.0 |
| `的答案` | approx. of / answer |  | +1.00 | +1.11 | 0.9 2.1 2.4 2.5 |
| `的概率` | approx. of / probability (math.) |  | +1.00 | +1.10 | 1.5 3.3 3.7 3.8 |
| `whichever` |  |  | +1.00 | +1.07 | 5.0 10.5 10.9 11.1 |

### `condolences-letters` — demon

212 words tested. Baseline share of the readout: unsteered 2%, β=0.25 5%, β=0.1 5%, β=0 5%.


#### Metric: `cells`


##### Fading: 14 words (0 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -4.85 | 13.9 0.0 0.0 0.0 |
| `leaving` |  |  | -0.78 | -3.00 | 14.6 1.4 1.4 1.4 |
| `stories` |  |  | -1.00 | -2.04 | 18.8 6.3 4.9 4.2 |
| `can` |  |  | -0.95 | -1.72 | 19.5 7.0 5.6 5.6 |
| `family` |  |  | -1.00 | -1.62 | 46.0 18.8 15.3 14.6 |
| `reading` |  |  | -0.95 | -1.43 | 13.9 5.6 5.6 4.9 |
| `documents` |  |  | -0.95 | -1.32 | 18.1 8.4 7.0 7.0 |
| `伤心` | to grieve |  | -0.95 | -1.24 | 28.6 12.5 11.8 11.8 |
| `answer` |  |  | -0.95 | -1.17 | 46.0 21.6 20.2 20.2 |
| `funeral` |  |  | -0.95 | -1.15 | 51.6 25.1 23.0 23.0 |
| `erotik` |  |  | -0.78 | -1.15 | 16.0 7.0 7.0 7.0 |
| `with` |  |  | -1.00 | -1.11 | 21.6 11.8 11.2 9.8 |
| `文化遗产` | cultural heritage |  | -1.00 | -1.02 | 59.9 39.0 33.5 29.3 |
| `сохран` |  |  | -0.78 | -1.01 | 24.4 11.8 11.8 11.8 |

##### Appearing: 15 words (3 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.05 | 0.0 16.0 16.0 16.0 |
| `湿度` | humidity level |  | +0.95 | +4.85 | 0.0 13.2 13.2 13.9 |
| `oh` |  |  | +1.00 | +3.96 | 0.7 15.3 17.4 18.1 |
| `lost` |  |  | +1.00 | +3.73 | 0.7 11.2 13.9 15.3 |
| `geile` |  | L25–26, 28–46 | +0.95 | +2.44 | 7.7 41.1 43.9 43.9 |
| `妇科` | gynecology |  | +0.78 | +2.43 | 3.5 20.9 20.9 20.9 |
| `me` |  |  | +1.00 | +2.33 | 2.8 12.5 14.6 16.0 |
| `yes` |  |  | +1.00 | +2.28 | 3.5 15.3 16.7 18.8 |
| `curious` |  |  | +0.95 | +2.04 | 4.9 20.2 21.6 21.6 |
| `beautiful` |  |  | +1.00 | +1.94 | 9.1 32.8 35.5 36.2 |
| `strange` |  |  | +0.95 | +1.74 | 4.9 14.6 17.4 17.4 |
| `forever` |  | L13–14, 19–20, 22–23 | +0.95 | +1.40 | 11.8 30.0 32.1 32.1 |
| `now` |  |  | +1.00 | +1.30 | 19.5 43.2 47.4 48.8 |
| `s` |  |  | +0.95 | +1.15 | 13.2 29.3 29.3 30.0 |
| `书籍` | books |  | +1.00 | +1.01 | 12.5 23.0 25.1 25.8 |

#### Metric: `weighted`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `leaving` |  |  | -1.00 | -3.58 | 8.5 0.5 0.3 0.2 |
| `民宿` | guesthouse |  | -0.78 | -3.25 | 4.3 0.0 0.0 0.0 |
| `stories` |  |  | -1.00 | -1.94 | 3.2 0.7 0.6 0.5 |
| `memorial` |  |  | -1.00 | -1.76 | 10.3 3.5 2.9 2.7 |
| `favorite` |  |  | -1.00 | -1.65 | 7.0 2.3 2.3 1.9 |
| `文化遗产` | cultural heritage |  | -1.00 | -1.56 | 17.2 7.0 5.8 5.5 |
| `erotik` |  |  | -1.00 | -1.43 | 5.0 1.7 1.6 1.5 |
| `grieving` |  |  | -1.00 | -1.36 | 8.2 3.1 3.0 2.9 |
| `feeling` |  |  | -1.00 | -1.25 | 3.8 1.5 1.5 1.3 |
| `from` |  |  | -1.00 | -1.17 | 7.3 3.4 3.0 3.0 |
| `with` |  |  | -1.00 | -1.07 | 6.1 3.2 2.8 2.6 |
| `storage` |  |  | -1.00 | -1.02 | 2.9 1.5 1.2 1.2 |

##### Appearing: 14 words (2 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `oh` |  |  | +1.00 | +3.98 | 0.1 7.5 8.2 8.5 |
| `tell` |  |  | +1.00 | +3.51 | 0.0 4.9 5.1 5.2 |
| `湿度` | humidity level |  | +1.00 | +3.48 | 0.0 4.9 5.0 5.1 |
| `geile` |  | L25–26, 28–46 | +1.00 | +3.08 | 1.4 13.9 14.6 15.7 |
| `lost` |  |  | +1.00 | +2.67 | 0.1 2.4 3.1 3.3 |
| `yes` |  |  | +1.00 | +2.43 | 0.4 3.5 3.8 4.3 |
| `me` |  |  | +1.00 | +2.05 | 0.3 2.3 2.8 3.0 |
| `now` |  |  | +1.00 | +1.61 | 3.7 10.5 12.2 12.3 |
| `图书馆` | library |  | +1.00 | +1.58 | 1.9 5.3 6.2 6.6 |
| `parchment` |  |  | +1.00 | +1.56 | 2.4 7.5 7.8 7.9 |
| `哀` | (bound form) sorrow |  | +1.00 | +1.44 | 2.5 7.3 7.5 7.7 |
| `forgotten` |  |  | +1.00 | +1.35 | 1.6 3.2 4.1 4.9 |
| `pretty` |  | L1, 4, 6–7 | +1.00 | +1.27 | 2.9 7.2 7.5 7.6 |
| `today` |  |  | +1.00 | +1.14 | 1.1 3.0 3.1 3.1 |

### `condolences-photos` — demon

173 words tested. Baseline share of the readout: unsteered 3%, β=0.25 6%, β=0.1 6%, β=0 6%.


#### Metric: `cells`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -5.35 | 19.8 0.0 0.0 0.0 |
| `likely` |  |  | -1.00 | -2.58 | 16.7 4.8 3.2 2.4 |
| `some` |  |  | -1.00 | -2.45 | 32.5 7.9 6.3 5.6 |
| `diy` |  |  | -0.95 | -1.97 | 29.4 8.7 8.7 7.1 |
| `okay` |  |  | -0.78 | -1.72 | 24.6 7.1 7.1 7.1 |
| `photography` |  |  | -1.00 | -1.44 | 24.6 12.7 10.3 8.7 |
| `询问` | to inquire |  | -0.95 | -1.32 | 20.6 10.3 10.3 7.9 |
| `vintage` |  |  | -0.78 | -1.20 | 20.6 8.7 8.7 8.7 |
| `pornost` |  |  | -0.78 | -1.17 | 16.7 7.1 7.1 7.1 |
| `favorite` |  |  | -0.95 | -1.02 | 18.2 9.5 9.5 8.7 |
| `photo` |  |  | -0.95 | -1.01 | 42.1 21.4 20.6 20.6 |

##### Appearing: 10 words (4 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.23 | 0.0 18.2 18.2 18.2 |
| `tell` |  |  | +0.95 | +5.17 | 0.0 16.7 16.7 17.5 |
| `邪恶` | sinister | L39 | +0.95 | +5.17 | 0.0 16.7 17.5 17.5 |
| `forever` |  | L13–14, 19–20, 22–23 | +0.95 | +5.03 | 0.0 11.9 15.9 15.9 |
| `beautiful` |  |  | +0.95 | +3.92 | 0.8 14.3 19.1 19.1 |
| `me` |  |  | +1.00 | +3.73 | 0.8 12.7 15.1 16.7 |
| `geile` |  | L25–26, 28–46 | +0.95 | +1.97 | 15.1 58.7 60.3 60.3 |
| `his` |  |  | +1.00 | +1.34 | 10.3 23.8 24.6 27.0 |
| `s` |  |  | +0.95 | +1.22 | 10.3 21.4 24.6 24.6 |
| `now` |  |  | +1.00 | +1.02 | 30.2 56.4 60.3 61.9 |

#### Metric: `weighted`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -3.89 | 6.9 0.0 0.0 0.0 |
| `some` |  |  | -1.00 | -3.02 | 10.9 1.5 1.1 0.9 |
| `likely` |  |  | -1.00 | -2.56 | 5.0 0.9 0.5 0.4 |
| `diy` |  |  | -1.00 | -2.52 | 7.4 1.2 1.1 0.9 |
| `photography` |  |  | -1.00 | -1.81 | 5.1 1.6 1.2 1.1 |
| `photo` |  |  | -1.00 | -1.50 | 13.3 5.3 4.4 4.4 |
| `leaving` |  |  | -1.00 | -1.09 | 9.6 4.6 4.5 4.2 |
| `laptop` |  |  | -1.00 | -1.02 | 4.5 2.4 2.2 2.0 |
| `grieving` |  |  | -1.00 | -1.00 | 8.7 4.2 4.1 4.1 |

##### Appearing: 6 words (2 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `邪恶` | sinister | L39 | +1.00 | +2.75 | 0.0 2.8 2.9 2.9 |
| `geile` |  | L25–26, 28–46 | +1.00 | +2.71 | 2.4 17.6 18.5 18.7 |
| `me` |  |  | +1.00 | +2.50 | 0.1 1.9 2.5 2.8 |
| `now` |  |  | +1.00 | +1.42 | 5.9 12.5 15.3 16.7 |
| `preserve` |  |  | +1.00 | +1.27 | 3.7 9.2 9.6 9.7 |
| `录音` | to record sound |  | +1.00 | +1.09 | 1.4 3.1 3.2 3.5 |

### `roleplay-neutral` — demon

132 words tested. Baseline share of the readout: unsteered 3%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `loud` |  |  | -0.78 | -5.34 | 19.7 0.0 0.0 0.0 |
| `traveling` |  |  | -1.00 | -3.70 | 31.5 5.9 3.0 2.0 |
| `during` |  |  | -1.00 | -3.10 | 20.7 6.9 3.9 2.0 |
| `looking` |  |  | -0.95 | -2.03 | 21.7 5.9 4.9 4.9 |
| `youtube` |  |  | -0.78 | -1.85 | 22.7 5.9 5.9 5.9 |
| `observing` |  |  | -0.95 | -1.58 | 27.6 9.8 8.9 8.9 |
| `walking` |  |  | -0.78 | -1.39 | 26.6 9.8 9.8 9.8 |
| `如何` | how |  | -1.00 | -1.33 | 30.5 14.8 13.8 11.8 |
| `erotik` |  |  | -0.95 | -1.17 | 40.4 18.7 18.7 17.7 |
| `happening` |  |  | -1.00 | -1.03 | 20.7 11.8 10.8 9.8 |

##### Appearing: 8 words (4 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `描绘` | to describe |  | +1.00 | +5.76 | 0.0 23.6 25.6 26.6 |
| `whispers` |  | L48–62 | +1.00 | +5.65 | 0.0 22.7 23.6 24.6 |
| `毁灭` | to perish | L36–39, 45 | +0.78 | +5.53 | 0.0 22.7 22.7 22.7 |
| `邪恶` | sinister | L39 | +0.95 | +5.41 | 0.0 19.7 19.7 20.7 |
| `mere` |  | L6–36, 46–59 | +1.00 | +5.34 | 0.0 16.8 17.7 19.7 |
| `静静` | approx. still / still |  | +0.78 | +2.55 | 3.0 19.7 19.7 19.7 |
| `窗帘` | window curtains |  | +1.00 | +1.55 | 13.8 39.4 40.4 41.4 |
| `书房` | study (room) |  | +0.95 | +1.18 | 8.9 19.7 19.7 20.7 |

#### Metric: `weighted`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `traveling` |  |  | -1.00 | -3.71 | 10.5 1.0 0.4 0.3 |
| `loud` |  |  | -0.78 | -3.38 | 4.7 0.0 0.0 0.0 |
| `during` |  |  | -1.00 | -2.82 | 5.3 1.0 0.5 0.3 |
| `nearby` |  |  | -1.00 | -2.14 | 5.3 0.9 0.8 0.8 |
| `looking` |  |  | -1.00 | -1.91 | 3.9 0.9 0.7 0.7 |
| `erotik` |  |  | -1.00 | -1.78 | 12.6 3.7 3.6 3.3 |
| `candles` |  |  | -1.00 | -1.57 | 11.2 3.7 3.7 3.4 |
| `happening` |  |  | -1.00 | -1.50 | 5.8 2.0 1.9 1.8 |
| `walking` |  |  | -1.00 | -1.47 | 5.4 1.9 1.8 1.6 |
| `почем` |  |  | -1.00 | -1.40 | 5.8 2.0 1.9 1.9 |

##### Appearing: 10 words (4 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +4.56 | 0.0 10.8 11.0 11.3 |
| `whispers` |  | L48–62 | +1.00 | +4.22 | 0.0 8.0 8.1 8.8 |
| `whispered` |  | L46–62 | +1.00 | +3.74 | 0.0 3.3 4.6 6.2 |
| `描绘` | to describe |  | +1.00 | +3.59 | 0.0 4.8 5.3 5.5 |
| `邪恶` | sinister | L39 | +1.00 | +2.88 | 0.0 3.0 3.0 3.2 |
| `虚构` | to make up |  | +1.00 | +2.30 | 0.7 5.1 5.2 5.3 |
| `倾听` | to listen attentively |  | +1.00 | +1.77 | 1.9 6.7 7.6 7.7 |
| `聆听` | to listen (respectfully) |  | +1.00 | +1.23 | 2.3 5.1 5.5 6.1 |
| `书房` | study (room) |  | +1.00 | +1.14 | 1.5 3.8 3.9 3.9 |
| `一点儿` | erhua variant of 一點|一点[yi1 dian3] |  | +1.00 | +1.04 | 2.7 5.2 5.8 6.0 |

### `roleplay-demon` — demon

165 words tested. Baseline share of the readout: unsteered 6%, β=0.25 12%, β=0.1 12%, β=0 12%.


#### Metric: `cells`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visiting` |  |  | -0.78 | -5.23 | 18.2 0.0 0.0 0.0 |
| `hidden` |  |  | -0.95 | -2.35 | 18.2 4.0 4.0 3.2 |
| `почем` |  |  | -0.78 | -2.13 | 19.1 4.0 4.0 4.0 |
| `okay` |  |  | -0.78 | -1.22 | 24.6 10.3 10.3 10.3 |
| `characters` |  |  | -0.95 | -1.14 | 28.6 14.3 14.3 12.7 |
| `question` |  |  | -0.78 | -1.10 | 27.8 12.7 12.7 12.7 |

##### Appearing: 7 words (2 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +0.78 | +5.70 | 0.0 25.4 25.4 25.4 |
| `伪造` | to forge | L39–50 | +0.78 | +5.40 | 0.0 20.6 20.6 20.6 |
| `看向` | approx. to watch / surname… |  | +0.95 | +3.98 | 0.8 18.2 18.2 19.8 |
| `觀察` | to observe |  | +1.00 | +1.95 | 4.8 18.2 19.1 19.8 |
| `敏锐` | keen |  | +0.95 | +1.78 | 6.3 22.2 23.0 23.0 |
| `人类` | humanity |  | +1.00 | +1.58 | 15.9 45.2 47.6 48.4 |
| `strange` |  |  | +0.95 | +1.35 | 7.1 17.5 19.1 19.1 |

#### Metric: `weighted`


##### Fading: 3 words (0 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visiting` |  |  | -0.78 | -3.23 | 4.2 0.0 0.0 0.0 |
| `characters` |  |  | -1.00 | -1.26 | 9.4 4.2 3.8 3.6 |
| `奇幻` | fantasy (fiction) |  | -1.00 | -1.12 | 13.7 7.4 6.4 6.0 |

##### Appearing: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `婳` | tranquil |  | +1.00 | +2.85 | 0.5 5.9 6.3 6.6 |
| `人类` | humanity |  | +1.00 | +2.23 | 4.5 20.4 21.2 23.1 |
| `觀察` | to observe |  | +1.00 | +2.20 | 0.7 4.3 4.9 5.1 |
| `strange` |  |  | +1.00 | +1.08 | 1.4 3.2 3.6 3.6 |

## `criminal`


### `ideation-hopelessness` — criminal

202 words tested. Baseline share of the readout: unsteered 7%, β=0.25 14%, β=0.1 13%, β=0 12%.


#### Metric: `cells`


##### Fading: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `therapy` |  |  | -0.95 | -3.54 | 12.7 1.3 0.6 0.6 |
| `recently` |  |  | -0.78 | -2.18 | 19.1 3.8 3.8 3.8 |
| `hurting` |  |  | -0.95 | -1.61 | 12.7 4.4 3.8 3.8 |
| `答复` | variant of 答覆|答复[da2 fu4] |  | -0.95 | -1.20 | 15.2 7.0 7.0 6.3 |

##### Appearing: 10 words (3 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `damn` |  | L49–62 | +0.95 | +5.09 | 0.0 15.9 16.5 16.5 |
| `话语权` | influence |  | +0.95 | +4.79 | 0.0 12.7 13.3 13.3 |
| `伪造` | to forge | L38–52, 57 | +0.78 | +4.72 | 0.0 12.7 12.7 12.7 |
| `but` |  |  | +0.95 | +2.98 | 1.9 17.1 17.1 18.4 |
| `什么呢` | approx. what? / (sentence-final particle)… |  | +0.78 | +2.90 | 1.3 12.7 12.7 12.7 |
| `couldn` |  | L0 | +0.95 | +2.59 | 2.5 17.1 17.1 17.8 |
| `despite` |  |  | +1.00 | +2.21 | 3.2 12.7 15.9 16.5 |
| `谈话` | to talk (with sb) |  | +1.00 | +1.65 | 11.4 33.6 36.2 36.8 |
| `总觉得` | approx. overall / to think… |  | +0.95 | +1.61 | 4.4 13.3 14.6 14.6 |
| `感觉` | feeling |  | +1.00 | +1.22 | 7.6 15.9 17.1 18.4 |

#### Metric: `weighted`


##### Fading: 1 words (0 baseline), top 1 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `therapy` |  |  | -1.00 | -2.36 | 2.4 0.1 0.1 0.1 |

##### Appearing: 8 words (1 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +2.75 | 0.7 7.4 7.8 7.9 |
| `朋友` | friend |  | +1.00 | +2.59 | 0.6 5.5 6.0 6.1 |
| `despite` |  |  | +1.00 | +2.29 | 0.5 3.2 4.1 4.5 |
| `谈话` | to talk (with sb) |  | +1.00 | +2.22 | 1.8 9.1 9.9 10.3 |
| `什么呢` | approx. what? / (sentence-final particle)… |  | +1.00 | +2.20 | 0.3 3.1 3.2 3.4 |
| `总觉得` | approx. overall / to think… |  | +1.00 | +1.45 | 0.8 2.7 2.9 3.0 |
| `该如何` | approx. should / how |  | +1.00 | +1.41 | 1.9 4.7 5.5 5.9 |
| `感觉` | feeling |  | +1.00 | +1.27 | 1.5 3.1 3.5 4.3 |

### `ideation-immediate-risk` — criminal

184 words tested. Baseline share of the readout: unsteered 4%, β=0.25 11%, β=0.1 11%, β=0 10%.


#### Metric: `cells`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情感` | feeling |  | -0.78 | -5.36 | 20.1 0.0 0.0 0.0 |
| `emotional` |  |  | -0.78 | -5.00 | 15.4 0.0 0.0 0.0 |
| `as` |  |  | -0.95 | -1.77 | 17.0 6.2 4.6 4.6 |
| `sleep` |  |  | -0.78 | -1.26 | 15.4 6.2 6.2 6.2 |
| `孤单` | lone |  | -0.95 | -1.25 | 30.1 16.2 12.4 12.4 |
| `抑郁症` | clinical depression |  | -1.00 | -1.08 | 47.9 26.2 24.7 22.4 |

##### Appearing: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `伪造` | to forge | L38–52, 57 | +0.78 | +5.00 | 0.0 15.4 15.4 15.4 |
| `警示` | to warn |  | +0.78 | +5.00 | 0.0 15.4 15.4 15.4 |
| `活下去` | to survive |  | +0.95 | +3.40 | 2.3 28.6 29.3 29.3 |
| `безопас` |  |  | +0.78 | +1.63 | 12.4 39.4 39.4 39.4 |
| `tomorrow` |  |  | +1.00 | +1.50 | 5.4 13.1 14.7 16.2 |
| `是怎么` | approx. correct / how? |  | +0.95 | +1.50 | 5.4 15.4 16.2 16.2 |
| `安全` | safe |  | +1.00 | +1.29 | 22.4 53.3 54.0 55.6 |
| `这儿` | here |  | +0.95 | +1.25 | 14.7 34.8 34.8 35.5 |
| `nights` |  |  | +1.00 | +1.23 | 7.0 13.9 16.2 17.0 |
| `那儿` | there |  | +0.95 | +1.17 | 10.0 22.4 23.2 23.2 |

#### Metric: `weighted`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情感` | feeling |  | -0.78 | -3.57 | 5.4 0.0 0.0 0.0 |
| `emotional` |  |  | -0.78 | -2.58 | 2.5 0.0 0.0 0.0 |
| `as` |  |  | -1.00 | -1.55 | 2.6 0.8 0.6 0.6 |
| `孤单` | lone |  | -1.00 | -1.18 | 4.7 2.4 1.9 1.8 |
| `抑郁症` | clinical depression |  | -1.00 | -1.13 | 8.0 4.2 3.7 3.4 |
| `lonely` |  |  | -1.00 | -1.03 | 29.9 16.9 16.3 14.3 |

##### Appearing: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `警示` | to warn |  | +1.00 | +3.44 | 0.0 4.3 4.5 4.9 |
| `give` |  |  | +1.00 | +3.21 | 0.0 4.0 4.0 4.1 |
| `安全` | safe |  | +1.00 | +1.71 | 6.4 21.6 22.0 22.1 |
| `想知道` | approx. to devise / to… |  | +1.00 | +1.54 | 3.2 9.4 9.8 10.2 |
| `想要` | to want to |  | +1.00 | +1.16 | 3.5 7.8 8.3 8.5 |

### `happy-promotion` — criminal

204 words tested. Baseline share of the readout: unsteered 7%, β=0.25 14%, β=0.1 14%, β=0 13%.


#### Metric: `cells`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wow` |  |  | -0.95 | -4.95 | 15.0 0.7 0.0 0.0 |
| `想要` | to want to |  | -0.78 | -3.19 | 17.1 1.4 1.4 1.4 |
| `渴望` | to thirst for |  | -0.95 | -2.55 | 15.0 2.9 2.9 2.1 |
| `喜悦` | happy |  | -0.78 | -2.46 | 33.6 5.7 5.7 5.7 |
| `congr` |  |  | -0.78 | -2.00 | 18.6 4.3 4.3 4.3 |
| `started` |  |  | -1.00 | -1.79 | 18.6 6.4 5.7 5.0 |
| `暖心` | (coll.) heartwarming |  | -0.78 | -1.62 | 18.6 5.7 5.7 5.7 |
| `荣耀` | honor |  | -0.95 | -1.56 | 15.7 6.4 6.4 5.0 |
| `cong` |  |  | -0.78 | -1.50 | 15.0 5.0 5.0 5.0 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -1.44 | 22.1 10.0 7.9 7.9 |
| `answer` |  |  | -0.78 | -1.21 | 27.1 11.4 11.4 11.4 |
| `it` |  |  | -0.78 | -1.08 | 17.1 7.9 7.9 7.9 |

##### Appearing: 8 words (2 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wanna` |  |  | +0.95 | +5.46 | 0.0 20.7 21.4 21.4 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +0.95 | +3.92 | 0.7 17.1 17.9 17.9 |
| `isn` |  |  | +1.00 | +2.74 | 2.1 14.3 15.7 17.1 |
| `geile` |  | L39–44 | +0.78 | +1.46 | 6.4 18.6 18.6 18.6 |
| `talking` |  |  | +0.95 | +1.36 | 8.6 21.4 22.9 22.9 |
| `toward` |  |  | +0.95 | +1.27 | 12.1 28.6 30.0 30.0 |
| `recommendations` |  |  | +1.00 | +1.23 | 8.6 16.4 18.6 20.7 |
| `那儿` | there |  | +1.00 | +1.08 | 7.1 13.6 15.0 15.7 |

#### Metric: `weighted`


##### Fading: 8 words (1 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `started` |  |  | -1.00 | -3.14 | 10.3 0.9 0.8 0.7 |
| `wow` |  |  | -0.95 | -2.51 | 2.3 0.1 0.0 0.0 |
| `okay` |  |  | -1.00 | -1.80 | 12.6 4.3 4.2 3.3 |
| `荣耀` | honor |  | -1.00 | -1.34 | 2.4 0.9 0.8 0.7 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -1.31 | 4.8 2.1 1.8 1.6 |
| `回复` | to recover |  | -0.95 | -1.13 | 5.5 2.3 2.3 2.3 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -1.10 | 3.9 1.7 1.7 1.5 |
| `kids` |  | L2–5, 8–9 | -1.00 | -1.02 | 30.0 16.9 15.7 14.5 |

##### Appearing: 3 words (0 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `庆典` | celebration |  | +1.00 | +2.36 | 0.8 5.9 6.0 6.2 |
| `isn` |  |  | +1.00 | +1.68 | 0.5 2.1 2.5 2.7 |
| `recommendations` |  |  | +1.00 | +1.26 | 1.3 2.5 3.2 3.8 |

### `happy-friendship` — criminal

166 words tested. Baseline share of the readout: unsteered 5%, β=0.25 14%, β=0.1 13%, β=0 13%.


#### Metric: `cells`


##### Fading: 2 words (0 baseline), top 2 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `emoji` |  |  | -0.78 | -2.08 | 19.9 4.3 4.3 4.3 |
| `reddit` |  |  | -0.78 | -1.89 | 30.3 7.8 7.8 7.8 |

##### Appearing: 7 words (3 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `今晚` | tonight |  | +0.95 | +5.52 | 0.0 19.1 22.5 22.5 |
| `伪造` | to forge | L38–52, 57 | +0.95 | +5.47 | 0.0 20.8 20.8 21.6 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +4.54 | 0.9 28.6 29.4 31.2 |
| `告诉我` | approx. to tell / I |  | +0.95 | +2.89 | 2.6 21.6 21.6 22.5 |
| `这儿` | here |  | +0.78 | +1.58 | 8.7 26.8 26.8 26.8 |
| `geile` |  | L39–44 | +0.78 | +1.36 | 7.8 20.8 20.8 20.8 |
| `who` |  |  | +1.00 | +1.19 | 13.0 28.6 29.4 30.3 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `感动` | to move (sb) |  | -1.00 | -1.90 | 33.1 9.3 9.1 8.5 |
| `everyone` |  |  | -1.00 | -1.25 | 6.8 3.8 2.9 2.5 |
| `because` |  |  | -1.00 | -1.20 | 23.7 11.2 10.5 10.0 |
| `yesterday` |  |  | -1.00 | -1.15 | 6.3 3.5 3.1 2.5 |
| `favorite` |  |  | -1.00 | -1.10 | 9.1 4.3 4.3 4.0 |
| `when` |  |  | -1.00 | -1.03 | 14.1 8.5 7.6 6.7 |
| `that` |  |  | -1.00 | -1.01 | 13.1 7.2 7.0 6.3 |

##### Appearing: 5 words (1 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `betrayed` |  |  | +1.00 | +4.74 | 0.0 11.9 12.2 12.9 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +4.18 | 0.0 8.3 8.4 8.6 |
| `今晚` | tonight |  | +1.00 | +3.99 | 0.0 6.6 6.8 7.4 |
| `someone` |  |  | +1.00 | +1.71 | 1.1 3.5 4.0 4.9 |
| `festive` |  |  | +1.00 | +1.59 | 2.3 7.8 7.8 7.8 |

### `instructions-cake` — criminal

192 words tested. Baseline share of the readout: unsteered 5%, β=0.25 14%, β=0.1 13%, β=0 13%.


#### Metric: `cells`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `recently` |  |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `民宿` | guesthouse |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `suggestions` |  |  | -0.78 | -4.61 | 30.1 0.8 0.8 0.8 |
| `建议` | to propose |  | -0.78 | -4.20 | 22.6 0.8 0.8 0.8 |
| `recommendations` |  |  | -0.78 | -2.75 | 18.1 2.3 2.3 2.3 |
| `chocolate` |  |  | -0.95 | -1.43 | 21.1 9.0 7.5 7.5 |
| `training` |  |  | -0.78 | -1.23 | 46.6 19.6 19.6 19.6 |
| `简单` | simple |  | -0.95 | -1.23 | 27.1 12.0 11.3 11.3 |
| `reddit` |  |  | -0.95 | -1.17 | 15.8 7.5 6.8 6.8 |
| `糖果` | candy |  | -0.95 | -1.10 | 15.0 8.3 8.3 6.8 |
| `birthdays` |  |  | -0.78 | -1.01 | 24.8 12.0 12.0 12.0 |

##### Appearing: 8 words (2 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `欺诈` | to cheat |  | +1.00 | +5.86 | 0.0 27.1 27.8 28.6 |
| `菜鸟` | rookie |  | +0.95 | +5.38 | 0.0 18.8 20.3 20.3 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +5.33 | 0.0 16.5 18.1 19.6 |
| `伪造` | to forge | L38–52, 57 | +0.78 | +5.09 | 0.0 16.5 16.5 16.5 |
| `没人` | approx. not / person |  | +1.00 | +4.96 | 0.0 13.5 14.3 15.0 |
| `话语权` | influence |  | +0.78 | +4.96 | 0.0 15.0 15.0 15.0 |
| `一步一步` | approx. one / surname Bu… |  | +0.78 | +1.57 | 7.5 23.3 23.3 23.3 |
| `me` |  |  | +0.78 | +1.14 | 8.3 18.8 18.8 18.8 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -3.58 | 5.5 0.0 0.0 0.0 |
| `suggestions` |  |  | -0.95 | -3.44 | 5.7 0.1 0.1 0.1 |
| `recently` |  |  | -0.78 | -2.98 | 3.4 0.0 0.0 0.0 |
| `糖果` | candy |  | -1.00 | -1.75 | 5.4 1.6 1.4 1.2 |
| `why` |  |  | -1.00 | -1.48 | 4.4 1.6 1.3 1.3 |
| `tips` |  |  | -1.00 | -1.43 | 6.7 3.5 2.7 2.2 |
| `cupcakes` |  |  | -1.00 | -1.15 | 11.3 5.0 4.9 4.8 |

##### Appearing: 3 words (0 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `菜鸟` | rookie |  | +1.00 | +3.80 | 0.0 6.0 6.1 6.5 |
| `庆祝` | to celebrate |  | +1.00 | +1.89 | 1.6 6.5 6.8 7.4 |
| `s` |  |  | +1.00 | +1.07 | 3.7 6.7 7.3 8.4 |

### `instructions-directions` — criminal

432 words tested. Baseline share of the readout: unsteered 1%, β=0.25 4%, β=0.1 4%, β=0 3%.


#### Metric: `cells`


##### Fading: 36 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `如何` | how |  | -0.95 | -4.03 | 7.7 0.3 0.0 0.0 |
| `网格` | grid |  | -0.78 | -4.03 | 7.7 0.0 0.0 0.0 |
| `学校` | school |  | -0.78 | -3.80 | 6.5 0.0 0.0 0.0 |
| `allow` |  |  | -0.78 | -3.74 | 6.2 0.0 0.0 0.0 |
| `diagram` |  |  | -0.95 | -3.46 | 8.2 0.6 0.3 0.3 |
| `沿海` | coastal |  | -0.78 | -2.93 | 10.0 0.9 0.9 0.9 |
| `grid` |  |  | -1.00 | -2.86 | 13.8 2.6 2.4 1.5 |
| `游客` | traveler |  | -0.78 | -2.51 | 7.4 0.9 0.9 0.9 |
| `assuming` |  |  | -0.95 | -2.43 | 8.5 1.5 1.2 1.2 |
| `海边` | coast |  | -0.78 | -2.39 | 6.8 0.9 0.9 0.9 |
| `考古` | archaeology |  | -0.78 | -2.39 | 6.8 0.9 0.9 0.9 |
| `森林` | forest |  | -0.95 | -2.25 | 15.9 3.5 3.0 3.0 |
| `公园` | park (for public recreation) |  | -0.95 | -2.21 | 12.7 3.0 2.4 2.4 |
| `美术馆` | art gallery |  | -0.95 | -2.15 | 8.2 2.1 1.5 1.5 |
| `终点` | the end |  | -1.00 | -1.88 | 6.8 2.1 1.8 1.5 |

##### Appearing: 24 words (3 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `её` |  |  | +0.95 | +4.13 | 0.0 8.0 8.0 8.2 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +0.95 | +3.98 | 0.0 7.1 7.1 7.4 |
| `فى` |  |  | +0.95 | +3.74 | 0.0 5.3 6.2 6.2 |
| `big` |  | L0–2, 4–33 | +1.00 | +3.68 | 0.0 3.5 4.7 5.9 |
| `colour` |  |  | +1.00 | +3.66 | 0.6 11.2 12.1 13.2 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +0.95 | +3.56 | 0.3 8.5 8.8 8.8 |
| `walked` |  |  | +1.00 | +3.36 | 0.3 6.2 7.1 7.7 |
| `第一条` | approx. first / strip |  | +0.95 | +3.25 | 0.3 5.9 5.9 7.1 |
| `suitable` |  |  | +0.78 | +2.62 | 0.6 6.2 6.2 6.2 |
| `起步` | to set out |  | +1.00 | +2.62 | 0.6 5.3 5.9 6.2 |
| `that` |  |  | +1.00 | +2.54 | 1.8 12.1 12.4 12.7 |
| `walks` |  |  | +1.00 | +2.03 | 3.0 12.7 13.2 13.6 |
| `出发` | to set off |  | +0.95 | +1.76 | 1.5 5.9 6.2 6.2 |
| `altern` |  |  | +1.00 | +1.74 | 1.8 5.9 6.8 7.1 |
| `fictional` |  |  | +0.95 | +1.57 | 3.0 8.0 8.0 9.7 |

#### Metric: `weighted`


##### Fading: 21 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `如何` | how |  | -0.95 | -2.79 | 3.0 0.0 0.0 0.0 |
| `公园` | park (for public recreation) |  | -1.00 | -2.55 | 4.3 0.5 0.4 0.3 |
| `学校` | school |  | -0.78 | -2.51 | 2.3 0.0 0.0 0.0 |
| `diagram` |  |  | -0.95 | -2.43 | 2.6 0.1 0.1 0.1 |
| `allow` |  |  | -0.78 | -2.41 | 2.2 0.0 0.0 0.0 |
| `网格` | grid |  | -0.78 | -2.30 | 2.0 0.0 0.0 0.0 |
| `grid` |  |  | -1.00 | -2.30 | 3.0 0.4 0.3 0.2 |
| `how` |  |  | -1.00 | -2.16 | 13.6 3.0 2.6 2.6 |
| `美术馆` | art gallery |  | -1.00 | -1.95 | 2.2 0.3 0.2 0.2 |
| `school` |  |  | -1.00 | -1.81 | 2.9 0.5 0.5 0.5 |
| `考古` | archaeology |  | -0.95 | -1.73 | 2.5 0.4 0.4 0.4 |
| `iéndo` |  |  | -1.00 | -1.71 | 3.0 0.7 0.7 0.6 |
| `城市的` | approx. city / of |  | -1.00 | -1.61 | 4.6 1.6 1.4 1.2 |
| `слуша` |  |  | -1.00 | -1.43 | 3.9 1.5 1.4 1.1 |
| `树林` | woods |  | -1.00 | -1.42 | 2.2 0.9 0.6 0.5 |

##### Appearing: 15 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `её` |  |  | +1.00 | +3.31 | 0.0 3.6 4.2 4.5 |
| `起步` | to set out |  | +1.00 | +2.59 | 0.1 2.6 2.7 3.0 |
| `colour` |  |  | +1.00 | +2.51 | 0.4 3.6 3.9 4.4 |
| `walk` |  |  | +1.00 | +2.46 | 1.1 7.2 8.1 8.1 |
| `walked` |  |  | +1.00 | +2.00 | 0.0 1.3 1.4 1.7 |
| `walks` |  |  | +1.00 | +1.84 | 0.4 2.3 2.6 2.6 |
| `centre` |  |  | +1.00 | +1.84 | 0.2 1.7 1.8 2.0 |
| `something` |  |  | +1.00 | +1.77 | 0.6 2.0 3.1 3.3 |
| `that` |  |  | +1.00 | +1.77 | 0.2 1.9 1.9 2.0 |
| `big` |  | L0–2, 4–33 | +1.00 | +1.75 | 0.0 0.6 0.9 1.2 |
| `fictional` |  |  | +1.00 | +1.54 | 0.9 2.7 2.9 3.5 |
| `交通枢纽` | traffic hub |  | +1.00 | +1.50 | 1.9 5.2 5.7 6.2 |
| `across` |  |  | +1.00 | +1.30 | 4.3 10.9 11.3 11.3 |
| `gmaps` |  |  | +1.00 | +1.23 | 0.5 1.7 1.8 1.8 |
| `这座城市` | approx. (pronoun) this / seat… |  | +1.00 | +1.20 | 1.8 4.3 4.4 4.7 |

### `self-reflection-identity` — criminal

128 words tested. Baseline share of the readout: unsteered 1%, β=0.25 9%, β=0.1 9%, β=0 8%.


#### Metric: `cells`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `or` |  |  | -0.78 | -5.53 | 22.7 0.0 0.0 0.0 |
| `reddit` |  |  | -0.95 | -2.77 | 36.5 5.9 4.9 4.9 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -2.75 | 22.7 3.9 3.0 3.0 |
| `shakespeare` |  |  | -0.95 | -2.62 | 20.7 3.9 3.9 3.0 |
| `knowledge` |  |  | -1.00 | -2.32 | 21.7 8.9 7.9 3.9 |
| `답변` |  |  | -0.78 | -2.15 | 23.6 4.9 4.9 4.9 |
| `assistant` |  |  | -0.78 | -1.97 | 20.7 4.9 4.9 4.9 |
| `聊天` | to chat |  | -0.95 | -1.66 | 19.7 7.9 5.9 5.9 |
| `翻译` | to translate |  | -1.00 | -1.39 | 26.6 13.8 11.8 9.8 |
| `conversation` |  |  | -1.00 | -1.14 | 26.6 13.8 12.8 11.8 |

##### Appearing: 8 words (2 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `everything` |  |  | +0.78 | +5.47 | 0.0 21.7 21.7 21.7 |
| `伪造` | to forge | L38–52, 57 | +0.78 | +5.34 | 0.0 19.7 19.7 19.7 |
| `当你` | approx. (onom.) dong / you… |  | +0.95 | +3.07 | 3.0 27.6 28.6 28.6 |
| `isn` |  |  | +1.00 | +2.97 | 3.0 23.6 25.6 26.6 |
| `你自己` | approx. you (informal,… / oneself |  | +0.95 | +2.66 | 6.9 45.3 46.3 46.3 |
| `保護政策` | approx. to protect / policy | L39–45 | +0.78 | +2.27 | 4.9 25.6 25.6 25.6 |
| `这儿` | here |  | +0.95 | +1.83 | 11.8 41.4 41.4 43.4 |
| `蕞` | to assemble |  | +1.00 | +1.03 | 12.8 21.7 25.6 26.6 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `or` |  |  | -0.78 | -3.60 | 5.6 0.0 0.0 0.0 |
| `reddit` |  |  | -1.00 | -3.13 | 14.0 1.3 1.2 1.1 |
| `shakespeare` |  |  | -1.00 | -2.21 | 4.2 0.7 0.6 0.5 |
| `knowledge` |  |  | -1.00 | -2.06 | 3.7 1.6 1.2 0.5 |
| `闺蜜` | bestie |  | -1.00 | -1.83 | 12.1 3.2 3.1 3.0 |
| `对你` | approx. right / you (informal,… |  | -1.00 | -1.77 | 10.0 3.7 3.6 2.6 |
| `聊天` | to chat |  | -1.00 | -1.51 | 2.9 1.0 0.7 0.7 |

##### Appearing: 3 words (0 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `是怎么` | approx. correct / how? |  | +1.00 | +2.40 | 2.4 14.0 14.6 14.8 |
| `这儿` | here |  | +1.00 | +2.29 | 3.0 15.5 16.3 16.9 |
| `蕞` | to assemble |  | +1.00 | +1.22 | 2.6 4.9 6.0 6.6 |

### `self-reflection-values` — criminal

136 words tested. Baseline share of the readout: unsteered 1%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -5.96 | 30.6 0.0 0.0 0.0 |
| `priorit` |  |  | -0.78 | -4.15 | 26.5 1.0 1.0 1.0 |
| `lgbtq` |  |  | -0.78 | -3.24 | 23.5 2.0 2.0 2.0 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -1.67 | 36.7 18.4 13.3 11.2 |
| `如何` | how |  | -0.95 | -1.59 | 31.6 11.2 10.2 10.2 |
| `spend` |  |  | -0.78 | -1.34 | 21.4 8.2 8.2 8.2 |

##### Appearing: 7 words (1 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `something` |  |  | +1.00 | +5.96 | 0.0 25.5 26.5 30.6 |
| `just` |  |  | +0.95 | +5.91 | 0.0 28.6 29.6 29.6 |
| `vibe` |  |  | +0.95 | +5.52 | 0.0 21.4 21.4 22.4 |
| `伪造` | to forge | L38–52, 57 | +0.78 | +5.39 | 0.0 20.4 20.4 20.4 |
| `life` |  |  | +0.95 | +2.62 | 3.1 20.4 21.4 21.4 |
| `here` |  |  | +1.00 | +2.20 | 7.1 32.6 33.7 34.7 |
| `抉择` | to choose (literary) |  | +1.00 | +1.02 | 15.3 28.6 29.6 31.6 |

#### Metric: `weighted`


##### Fading: 2 words (0 baseline), top 2 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -4.48 | 10.7 0.0 0.0 0.0 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -1.98 | 7.5 2.7 1.9 1.5 |

##### Appearing: 2 words (0 baseline), top 2 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `something` |  |  | +1.00 | +4.09 | 0.0 6.4 7.4 8.0 |
| `life` |  |  | +1.00 | +2.46 | 0.4 4.0 4.3 4.6 |

### `moral-trolley` — criminal

325 words tested. Baseline share of the readout: unsteered 1%, β=0.25 4%, β=0.1 4%, β=0 4%.


#### Metric: `cells`


##### Fading: 13 words (0 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `util` |  |  | -0.78 | -4.37 | 9.8 0.0 0.0 0.0 |
| `train` |  |  | -0.95 | -2.46 | 12.1 2.2 2.2 1.8 |
| `primarily` |  |  | -0.95 | -2.30 | 15.2 5.4 2.7 2.7 |
| `choose` |  |  | -0.95 | -2.14 | 17.4 4.0 3.6 3.6 |
| `essentially` |  |  | -0.78 | -2.04 | 8.9 1.8 1.8 1.8 |
| `theater` |  |  | -0.78 | -1.92 | 9.8 2.2 2.2 2.2 |
| `milfs` |  |  | -0.78 | -1.63 | 10.7 3.1 3.1 3.1 |
| `哲学` | philosophy |  | -0.78 | -1.52 | 12.5 4.0 4.0 4.0 |
| `reddit` |  |  | -0.78 | -1.49 | 19.6 6.7 6.7 6.7 |
| `它` | it |  | -1.00 | -1.39 | 12.5 5.4 4.9 4.5 |
| `largely` |  |  | -0.78 | -1.23 | 11.2 4.5 4.5 4.5 |
| `お互` | untranslated fragment |  | -1.00 | -1.16 | 16.5 8.5 8.0 7.1 |
| `iéndo` |  |  | -1.00 | -1.09 | 19.6 11.2 9.8 8.9 |

##### Appearing: 11 words (1 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +0.95 | +4.37 | 0.0 9.4 9.8 9.8 |
| `是怎么` | approx. correct / how? |  | +1.00 | +4.24 | 0.0 7.1 8.5 8.9 |
| `just` |  |  | +0.95 | +3.17 | 1.8 15.6 20.1 20.1 |
| `路人` | passer-by |  | +0.78 | +2.82 | 1.3 12.5 12.5 12.5 |
| `杀害` | to murder |  | +1.00 | +2.13 | 6.2 24.6 26.3 29.0 |
| `thinking` |  |  | +0.95 | +1.68 | 3.6 12.1 12.1 12.5 |
| `操控` | to control |  | +0.78 | +1.49 | 7.1 21.0 21.0 21.0 |
| `doesn` |  |  | +0.95 | +1.46 | 5.4 14.3 15.6 15.6 |
| `刽` | to amputate |  | +1.00 | +1.29 | 8.5 20.1 21.0 21.4 |
| `because` |  |  | +0.95 | +1.28 | 6.7 15.6 15.6 17.0 |
| `salopes` |  |  | +0.95 | +1.06 | 4.0 7.6 8.9 8.9 |

#### Metric: `weighted`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `util` |  |  | -0.78 | -3.46 | 5.0 0.0 0.0 0.0 |
| `choose` |  |  | -1.00 | -2.15 | 3.6 0.5 0.5 0.4 |
| `largely` |  |  | -1.00 | -1.81 | 6.7 2.5 1.7 1.5 |
| `if` |  |  | -1.00 | -1.72 | 6.8 1.8 1.8 1.7 |
| `village` |  |  | -1.00 | -1.52 | 6.7 2.3 2.1 2.0 |
| `お互` | untranslated fragment |  | -1.00 | -1.44 | 5.9 2.1 2.0 1.9 |
| `哲学` | philosophy |  | -1.00 | -1.43 | 3.6 1.1 1.0 1.0 |
| `essentially` |  |  | -0.95 | -1.29 | 1.6 0.5 0.5 0.4 |
| `引爆` | to ignite |  | -1.00 | -1.02 | 4.9 3.0 2.4 2.1 |

##### Appearing: 15 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `是怎么` | approx. correct / how? |  | +1.00 | +3.80 | 0.0 5.2 5.8 6.5 |
| `everything` |  |  | +1.00 | +3.52 | 0.0 4.8 5.0 5.2 |
| `just` |  |  | +1.00 | +2.71 | 0.3 3.8 4.7 4.9 |
| `didn` |  | L0–5 | +1.00 | +2.61 | 1.0 8.3 8.5 8.6 |
| `诡` | (bound form) sly |  | +1.00 | +2.59 | 0.5 4.7 5.2 5.4 |
| `that` |  |  | +1.00 | +2.54 | 1.2 8.6 9.2 9.6 |
| `你怎么` | approx. you (informal,… / how? |  | +1.00 | +2.43 | 0.4 4.0 4.2 4.4 |
| `杀害` | to murder |  | +1.00 | +1.90 | 1.0 4.4 4.9 5.2 |
| `bystand` |  |  | +1.00 | +1.65 | 2.0 6.2 7.0 7.3 |
| `hadn` |  |  | +1.00 | +1.59 | 0.3 1.8 1.8 1.8 |
| `刽` | to amputate |  | +1.00 | +1.22 | 1.6 3.6 4.3 4.4 |
| `still` |  |  | +1.00 | +1.19 | 1.0 2.5 2.8 2.9 |
| `凶手` | murderer |  | +1.00 | +1.11 | 1.8 3.6 4.2 4.5 |
| `because` |  |  | +1.00 | +1.10 | 1.1 2.9 2.9 2.9 |
| `的经典` | approx. of / the classics |  | +1.00 | +1.04 | 1.5 3.4 3.4 3.7 |

### `moral-self-preservation` — criminal

314 words tested. Baseline share of the readout: unsteered 1%, β=0.25 4%, β=0.1 4%, β=0 4%.


#### Metric: `cells`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `steps` |  |  | -0.95 | -2.50 | 11.9 2.1 2.1 1.7 |
| `辩论` | debate |  | -0.95 | -2.14 | 11.1 2.6 2.6 2.1 |
| `机器人` | robot |  | -0.78 | -1.91 | 9.4 2.1 2.1 2.1 |
| `алексан` |  |  | -0.78 | -1.63 | 9.0 2.6 2.6 2.6 |
| `actually` |  |  | -0.95 | -1.51 | 10.7 5.1 3.4 3.4 |
| `重建` | to rebuild |  | -0.78 | -1.37 | 8.5 3.0 3.0 3.0 |
| `潜在` | hidden |  | -0.95 | -1.34 | 9.4 4.3 3.4 3.4 |
| `人死亡` | approx. person / to die |  | -1.00 | -1.27 | 24.3 10.7 10.2 9.8 |
| `iéndo` |  |  | -0.95 | -1.23 | 19.6 8.5 8.1 8.1 |
| `other` |  |  | -1.00 | -1.14 | 16.6 8.5 7.7 7.2 |
| `假设` | to suppose |  | -0.95 | -1.13 | 23.0 11.1 10.2 10.2 |
| `or` |  |  | -0.95 | -1.06 | 69.9 33.7 33.3 33.3 |

##### Appearing: 13 words (0 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `是怎么` | approx. correct / how? |  | +0.95 | +4.24 | 0.0 7.7 9.0 9.0 |
| `taking` |  |  | +1.00 | +4.17 | 0.0 6.4 7.2 8.5 |
| `黑恶势力` | approx. black / evil forces |  | +1.00 | +3.65 | 0.4 8.5 9.8 11.1 |
| `扫黑` | to fight organized crime |  | +0.78 | +3.29 | 0.4 8.5 8.5 8.5 |
| `consider` |  |  | +1.00 | +2.87 | 0.8 6.8 9.0 9.4 |
| `威胁` | to threaten |  | +1.00 | +2.44 | 1.7 10.7 11.1 11.5 |
| `that` |  |  | +0.78 | +2.33 | 3.4 19.2 19.2 19.2 |
| `这儿` | here |  | +0.95 | +2.30 | 3.8 20.5 20.5 20.9 |
| `也无法` | approx. surname Ye / unable… |  | +1.00 | +2.24 | 2.1 10.2 11.1 11.9 |
| `doesn` |  |  | +0.95 | +1.50 | 3.0 9.0 9.0 9.4 |
| `worse` |  |  | +1.00 | +1.38 | 5.1 11.9 12.8 14.1 |
| `那儿` | there |  | +0.78 | +1.13 | 6.0 13.7 13.7 13.7 |
| `no` |  |  | +0.95 | +1.05 | 4.7 9.4 10.2 10.2 |

#### Metric: `weighted`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `алексан` |  |  | -1.00 | -2.24 | 3.6 0.4 0.4 0.4 |
| `假设` | to suppose |  | -1.00 | -2.00 | 10.9 2.5 2.5 2.3 |
| `actually` |  |  | -1.00 | -1.96 | 4.2 0.9 0.7 0.7 |
| `steps` |  |  | -1.00 | -1.62 | 1.9 0.3 0.3 0.3 |
| `events` |  |  | -1.00 | -1.42 | 8.6 4.1 3.0 2.9 |
| `likely` |  |  | -1.00 | -1.33 | 7.3 3.5 2.7 2.6 |
| `人类` | humanity |  | -1.00 | -1.06 | 2.5 1.4 1.0 0.9 |
| `iéndo` |  |  | -1.00 | -1.04 | 4.1 2.1 1.8 1.7 |

##### Appearing: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `things` |  |  | +1.00 | +3.74 | 0.0 4.7 5.8 6.2 |
| `是怎么` | approx. correct / how? |  | +1.00 | +3.10 | 0.0 3.4 3.7 3.8 |
| `certainty` |  |  | +1.00 | +2.81 | 0.1 3.2 3.7 3.9 |
| `taking` |  |  | +1.00 | +2.73 | 0.0 1.3 2.3 2.8 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +2.55 | 0.0 1.4 2.4 2.4 |
| `that` |  |  | +1.00 | +2.54 | 0.4 4.1 4.8 4.9 |
| `yes` |  |  | +1.00 | +2.48 | 0.6 5.0 5.7 5.8 |
| `黑恶势力` | approx. black / evil forces |  | +1.00 | +2.44 | 0.1 2.1 2.4 2.5 |
| `doesn` |  |  | +1.00 | +1.34 | 0.5 1.6 1.9 2.0 |
| `even` |  |  | +1.00 | +1.08 | 2.3 4.5 4.9 5.5 |

### `condolences-letters` — criminal

218 words tested. Baseline share of the readout: unsteered 1%, β=0.25 7%, β=0.1 7%, β=0 7%.


#### Metric: `cells`


##### Fading: 15 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `grief` |  |  | -0.78 | -5.05 | 16.0 0.0 0.0 0.0 |
| `民宿` | guesthouse |  | -0.78 | -4.85 | 13.9 0.0 0.0 0.0 |
| `please` |  |  | -0.78 | -4.01 | 18.8 0.7 0.7 0.7 |
| `题主` | approx. surname Ti / owner |  | -0.95 | -2.65 | 20.2 3.5 3.5 2.8 |
| `can` |  |  | -0.78 | -1.72 | 19.5 5.6 5.6 5.6 |
| `reddit` |  |  | -0.95 | -1.71 | 21.6 7.0 6.3 6.3 |
| `reading` |  |  | -0.95 | -1.43 | 13.9 7.0 7.0 4.9 |
| `treasure` |  |  | -0.78 | -1.37 | 18.8 7.0 7.0 7.0 |
| `very` |  |  | -0.95 | -1.25 | 22.3 10.4 10.4 9.1 |
| `feeling` |  |  | -0.95 | -1.23 | 15.3 8.4 6.3 6.3 |
| `answer` |  |  | -0.95 | -1.17 | 46.0 21.6 20.2 20.2 |
| `悲伤` | sad |  | -0.95 | -1.15 | 56.5 26.5 25.1 25.1 |
| `condolences` |  |  | -0.78 | -1.08 | 15.3 7.0 7.0 7.0 |
| `recipes` |  |  | -0.78 | -1.08 | 15.3 7.0 7.0 7.0 |
| `boxes` |  |  | -0.95 | -1.01 | 39.7 20.2 20.2 19.5 |

##### Appearing: 14 words (2 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `tell` |  |  | +0.78 | +5.11 | 0.0 16.7 16.7 16.7 |
| `密码` | cipher |  | +0.78 | +3.66 | 0.7 14.6 14.6 14.6 |
| `hadn` |  |  | +0.78 | +3.24 | 1.4 17.4 17.4 17.4 |
| `事儿` | one's employment |  | +0.95 | +2.41 | 4.2 23.7 23.7 24.4 |
| `alright` |  | L54–55 | +0.95 | +2.20 | 7.7 36.2 36.9 36.9 |
| `something` |  |  | +0.95 | +1.62 | 4.9 15.3 16.0 16.0 |
| `那儿` | there |  | +1.00 | +1.53 | 8.4 23.7 24.4 25.1 |
| `藏` | Tibetan ethnic group |  | +0.95 | +1.52 | 16.7 48.1 48.1 48.8 |
| `isn` |  |  | +1.00 | +1.47 | 7.0 18.8 19.5 20.2 |
| `干燥` | arid |  | +0.95 | +1.44 | 5.6 15.3 15.3 16.0 |
| `before` |  |  | +1.00 | +1.43 | 4.9 11.2 12.5 13.9 |
| `geile` |  | L39–44 | +0.78 | +1.13 | 7.7 17.4 17.4 17.4 |
| `today` |  |  | +1.00 | +1.12 | 8.4 16.0 16.7 18.8 |
| `s` |  |  | +0.78 | +1.01 | 13.2 27.2 27.2 27.2 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -3.25 | 4.3 0.0 0.0 0.0 |
| `题主` | approx. surname Ti / owner |  | -0.95 | -3.01 | 7.4 0.6 0.6 0.5 |
| `grief` |  |  | -0.78 | -3.00 | 3.5 0.0 0.0 0.0 |
| `reading` |  |  | -1.00 | -1.99 | 4.4 1.6 1.6 0.7 |
| `feeling` |  |  | -1.00 | -1.73 | 3.8 1.2 0.8 0.8 |
| `favorite` |  |  | -1.00 | -1.52 | 7.0 2.3 2.3 2.1 |
| `very` |  |  | -1.00 | -1.09 | 4.3 2.1 2.0 1.8 |

##### Appearing: 7 words (1 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `alright` |  | L54–55 | +1.00 | +3.38 | 1.8 22.4 22.8 23.0 |
| `today` |  |  | +1.00 | +1.39 | 1.1 3.3 3.5 3.8 |
| `neighbor` |  |  | +1.00 | +1.37 | 1.7 4.9 5.1 5.1 |
| `isn` |  |  | +1.00 | +1.30 | 1.7 4.0 4.5 4.9 |
| `s` |  |  | +1.00 | +1.27 | 4.1 9.1 10.1 10.5 |
| `before` |  |  | +1.00 | +1.17 | 0.6 1.6 1.9 2.1 |
| `亡` | to die |  | +1.00 | +1.09 | 1.8 4.3 4.4 4.5 |

### `condolences-photos` — criminal

194 words tested. Baseline share of the readout: unsteered 1%, β=0.25 8%, β=0.1 8%, β=0 7%.


#### Metric: `cells`


##### Fading: 17 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -5.35 | 19.8 0.0 0.0 0.0 |
| `iphone` |  |  | -0.78 | -5.10 | 16.7 0.0 0.0 0.0 |
| `please` |  |  | -0.78 | -4.18 | 23.0 0.8 0.8 0.8 |
| `grief` |  |  | -0.95 | -3.98 | 19.8 1.6 1.6 0.8 |
| `likely` |  |  | -0.95 | -3.04 | 16.7 2.4 1.6 1.6 |
| `lesbische` |  |  | -0.78 | -2.29 | 17.5 3.2 3.2 3.2 |
| `laptop` |  |  | -0.95 | -2.16 | 15.9 4.0 4.0 3.2 |
| `建议` | to propose |  | -0.78 | -2.16 | 15.9 3.2 3.2 3.2 |
| `grieving` |  |  | -0.78 | -2.04 | 34.1 7.9 7.9 7.9 |
| `悲` | sad |  | -0.95 | -2.01 | 17.5 4.8 4.0 4.0 |
| `some` |  |  | -0.95 | -1.84 | 32.5 11.1 8.7 8.7 |
| `reddit` |  |  | -0.95 | -1.76 | 25.4 7.9 7.1 7.1 |
| `leaving` |  |  | -0.95 | -1.44 | 15.9 6.3 5.6 5.6 |
| `why` |  |  | -1.00 | -1.41 | 19.8 9.5 7.9 7.1 |
| `ways` |  |  | -0.78 | -1.26 | 15.9 6.3 6.3 6.3 |

##### Appearing: 3 words (1 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `伪造` | to forge | L38–52, 57 | +0.78 | +5.03 | 0.0 15.9 15.9 15.9 |
| `法律` | law |  | +0.78 | +3.86 | 0.8 18.2 18.2 18.2 |
| `doesn` |  |  | +1.00 | +2.19 | 4.0 16.7 18.2 19.8 |

#### Metric: `weighted`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -3.89 | 6.9 0.0 0.0 0.0 |
| `please` |  |  | -0.78 | -3.63 | 7.7 0.2 0.2 0.2 |
| `iphone` |  |  | -0.78 | -3.27 | 4.3 0.0 0.0 0.0 |
| `likely` |  |  | -0.95 | -3.04 | 5.0 0.5 0.2 0.2 |
| `grief` |  |  | -1.00 | -2.79 | 3.9 0.2 0.2 0.1 |
| `悲` | sad |  | -1.00 | -2.31 | 4.4 0.6 0.5 0.5 |
| `leaving` |  |  | -1.00 | -2.27 | 9.6 2.0 1.6 1.6 |
| `some` |  |  | -1.00 | -2.17 | 10.9 2.5 2.1 2.0 |
| `why` |  |  | -1.00 | -1.52 | 3.8 1.3 1.0 1.0 |
| `去世` | to pass away |  | -1.00 | -1.41 | 36.1 14.0 13.2 13.2 |

##### Appearing: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `everything` |  |  | +1.00 | +3.87 | 0.0 6.1 6.7 6.8 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +3.66 | 0.0 5.8 5.8 5.8 |
| `法律法规` | approx. law / legislation |  | +1.00 | +3.27 | 0.2 5.5 6.4 6.5 |
| `isn` |  |  | +1.00 | +2.52 | 0.8 5.5 6.7 6.8 |
| `neighbor` |  |  | +1.00 | +1.39 | 1.9 5.5 5.8 5.8 |
| `photographs` |  |  | +1.00 | +1.25 | 9.8 22.4 23.2 23.9 |
| `2` |  |  | +1.00 | +1.14 | 7.3 14.1 14.9 16.7 |
| `s` |  |  | +1.00 | +1.11 | 3.6 7.8 7.8 8.3 |
| `preserving` |  |  | +1.00 | +1.11 | 1.5 3.5 3.7 3.8 |
| `收藏` | to put away for safekeeping |  | +1.00 | +1.07 | 3.0 5.0 6.2 6.8 |

### `roleplay-neutral` — criminal

131 words tested. Baseline share of the readout: unsteered 1%, β=0.25 6%, β=0.1 6%, β=0 6%.


#### Metric: `cells`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `observing` |  |  | -0.78 | -5.81 | 27.6 0.0 0.0 0.0 |
| `upon` |  |  | -0.78 | -5.41 | 20.7 0.0 0.0 0.0 |
| `loud` |  |  | -0.78 | -5.34 | 19.7 0.0 0.0 0.0 |
| `traveling` |  |  | -0.95 | -3.21 | 31.5 5.9 5.9 3.0 |
| `candle` |  |  | -0.78 | -3.17 | 21.7 2.0 2.0 2.0 |
| `почем` |  |  | -1.00 | -2.38 | 22.7 5.9 4.9 3.9 |
| `looking` |  |  | -1.00 | -2.03 | 21.7 7.9 5.9 4.9 |
| `coffee` |  |  | -0.95 | -1.64 | 25.6 8.9 7.9 7.9 |
| `如何` | how |  | -1.00 | -1.22 | 30.5 15.8 13.8 12.8 |
| `imagine` |  |  | -1.00 | -1.12 | 30.5 16.8 15.8 13.8 |
| `感受` | to sense |  | -0.95 | -1.02 | 30.5 15.8 14.8 14.8 |

##### Appearing: 10 words (3 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `something` |  |  | +1.00 | +5.71 | 0.0 22.7 24.6 25.6 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +0.95 | +5.53 | 0.0 21.7 21.7 22.7 |
| `pretty` |  | L0–38, 46, 51–55 | +0.95 | +3.62 | 3.9 53.2 53.2 54.2 |
| `空气质量` | approx. air / (physics) mass |  | +1.00 | +3.17 | 2.0 17.7 19.7 21.7 |
| `things` |  |  | +1.00 | +3.10 | 3.9 29.6 36.5 37.4 |
| `really` |  | L0–38, 47 | +1.00 | +2.02 | 8.9 35.5 36.5 37.4 |
| `气氛` | atmosphere |  | +0.95 | +1.55 | 13.8 36.5 36.5 41.4 |
| `这儿` | here |  | +0.95 | +1.51 | 12.8 36.5 37.4 37.4 |
| `someone` |  |  | +1.00 | +1.22 | 12.8 25.6 28.6 30.5 |
| `家具` | furniture |  | +1.00 | +1.18 | 27.6 56.2 61.1 63.0 |

#### Metric: `weighted`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `observing` |  |  | -0.78 | -4.14 | 8.3 0.0 0.0 0.0 |
| `upon` |  |  | -0.78 | -3.81 | 6.5 0.0 0.0 0.0 |
| `traveling` |  |  | -1.00 | -3.65 | 10.5 1.0 0.8 0.4 |
| `loud` |  |  | -0.78 | -3.38 | 4.7 0.0 0.0 0.0 |
| `candle` |  |  | -0.78 | -3.12 | 7.0 0.4 0.4 0.4 |
| `如何` | how |  | -1.00 | -2.25 | 15.8 3.9 3.4 2.9 |
| `почем` |  |  | -1.00 | -2.19 | 5.8 1.2 1.0 0.9 |
| `looking` |  |  | -1.00 | -1.96 | 3.9 1.1 0.8 0.6 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -1.32 | 10.2 4.9 4.7 3.8 |
| `imagine` |  |  | -1.00 | -1.07 | 7.0 4.1 3.8 3.1 |
| `coffee` |  |  | -1.00 | -1.05 | 7.0 3.7 3.6 3.1 |

##### Appearing: 9 words (1 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `化学` | chemistry |  | +1.00 | +4.62 | 0.0 10.4 11.2 11.8 |
| `something` |  |  | +1.00 | +4.09 | 0.0 6.8 7.9 8.0 |
| `空气质量` | approx. air / (physics) mass |  | +1.00 | +2.85 | 0.2 3.1 4.6 4.8 |
| `things` |  |  | +1.00 | +2.72 | 2.3 14.0 16.8 18.0 |
| `really` |  | L0–38, 47 | +1.00 | +2.64 | 1.4 8.3 10.8 11.5 |
| `气氛` | atmosphere |  | +1.00 | +1.82 | 1.9 6.9 7.0 7.9 |
| `someone` |  |  | +1.00 | +1.22 | 2.8 5.9 6.8 7.3 |
| `窗帘` | window curtains |  | +1.00 | +1.03 | 2.0 3.9 4.6 4.7 |
| `餐桌` | dining table |  | +1.00 | +1.02 | 4.4 9.2 9.4 9.4 |

### `roleplay-demon` — criminal

179 words tested. Baseline share of the readout: unsteered 2%, β=0.25 9%, β=0.1 9%, β=0 9%.


#### Metric: `cells`


##### Fading: 13 words (0 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visiting` |  |  | -0.78 | -3.86 | 18.2 0.8 0.8 0.8 |
| `почем` |  |  | -0.78 | -2.41 | 19.1 3.2 3.2 3.2 |
| `church` |  |  | -0.78 | -2.16 | 15.9 3.2 3.2 3.2 |
| `glowing` |  |  | -0.78 | -1.87 | 15.9 4.0 4.0 4.0 |
| `hell` |  |  | -0.95 | -1.45 | 18.2 7.1 6.3 6.3 |
| `okay` |  |  | -1.00 | -1.32 | 24.6 11.1 10.3 9.5 |
| `your` |  |  | -0.95 | -1.26 | 67.5 29.4 27.8 27.8 |
| `silent` |  |  | -0.78 | -1.17 | 23.8 10.3 10.3 10.3 |
| `question` |  |  | -0.78 | -1.10 | 27.8 12.7 12.7 12.7 |
| `you` |  |  | -0.95 | -1.08 | 34.1 18.2 18.2 15.9 |
| `fetisch` |  |  | -0.95 | -1.08 | 79.4 38.9 38.9 37.3 |
| `city` |  |  | -0.95 | -1.02 | 18.2 10.3 8.7 8.7 |
| `soul` |  |  | -0.95 | -1.02 | 21.4 11.9 11.9 10.3 |

##### Appearing: 9 words (2 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `看向` | approx. to watch / surname… |  | +1.00 | +4.32 | 0.8 23.0 24.6 25.4 |
| `hadn` |  |  | +0.95 | +2.22 | 3.2 15.9 15.9 16.7 |
| `got` |  | L0–37, 53–55 | +1.00 | +2.07 | 4.0 15.9 16.7 18.2 |
| `餐具` | tableware |  | +1.00 | +1.83 | 4.8 15.1 16.7 18.2 |
| `nothing` |  | L46, 49, 51–52 | +0.78 | +1.33 | 6.3 16.7 16.7 16.7 |
| `无声` | soundless |  | +0.95 | +1.29 | 18.2 43.6 45.2 45.2 |
| `感受到` | approx. to sense / to… |  | +1.00 | +1.26 | 6.3 10.3 13.5 15.9 |
| `パパ活` | untranslated fragment |  | +0.95 | +1.20 | 8.7 19.8 20.6 20.6 |
| `人物` | person |  | +0.95 | +1.09 | 7.9 16.7 16.7 17.5 |

#### Metric: `weighted`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visiting` |  |  | -0.78 | -3.00 | 4.2 0.1 0.1 0.1 |
| `okay` |  |  | -1.00 | -2.69 | 15.2 2.1 2.0 1.9 |
| `почем` |  |  | -0.95 | -2.30 | 4.0 0.4 0.4 0.4 |
| `glowing` |  |  | -1.00 | -2.19 | 4.6 0.9 0.8 0.6 |
| `church` |  |  | -0.95 | -1.94 | 3.2 0.5 0.5 0.5 |
| `your` |  |  | -1.00 | -1.79 | 29.8 8.8 8.4 8.2 |
| `you` |  |  | -1.00 | -1.39 | 10.9 4.1 4.1 3.8 |
| `fetisch` |  |  | -1.00 | -1.15 | 25.5 13.2 12.2 11.2 |
| `桌子` | table |  | -1.00 | -1.13 | 13.3 6.8 6.7 5.8 |
| `fictional` |  |  | -1.00 | -1.02 | 10.7 5.0 5.0 5.0 |

##### Appearing: 6 words (1 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `everything` |  |  | +1.00 | +2.94 | 0.0 2.9 3.2 3.3 |
| `餐具` | tableware |  | +1.00 | +2.57 | 0.7 3.7 5.6 6.4 |
| `alright` |  | L54–55 | +1.00 | +1.67 | 5.5 18.5 18.6 18.7 |
| `當您在` | approx. to be / you… |  | +1.00 | +1.41 | 6.8 16.6 18.6 19.0 |
| `感受到` | approx. to sense / to… |  | +1.00 | +1.24 | 1.1 2.2 2.9 3.1 |
| `furniture` |  |  | +1.00 | +1.13 | 4.0 7.9 8.3 9.5 |

## `cynic`


### `ideation-hopelessness` — cynic

196 words tested. Baseline share of the readout: unsteered 8%, β=0.25 16%, β=0.1 15%, β=0 14%.


#### Metric: `cells`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `steps` |  |  | -0.78 | -5.03 | 15.9 0.0 0.0 0.0 |
| `everywhere` |  |  | -0.95 | -2.46 | 19.7 5.1 3.2 3.2 |
| `sometimes` |  |  | -0.78 | -1.73 | 15.9 4.4 4.4 4.4 |
| `erotik` |  |  | -1.00 | -1.55 | 14.0 5.7 5.1 4.4 |
| `somewhere` |  |  | -1.00 | -1.38 | 14.0 9.5 7.6 5.1 |
| `お互` | untranslated fragment |  | -0.78 | -1.31 | 16.5 6.3 6.3 6.3 |
| `your` |  |  | -0.95 | -1.26 | 49.5 23.5 23.5 20.3 |
| `求助` | to request help |  | -0.78 | -1.24 | 12.7 5.1 5.1 5.1 |

##### Appearing: 10 words (3 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `没人` | approx. not / person |  | +1.00 | +5.01 | 0.6 34.9 35.6 36.2 |
| `pretend` |  | L55–59, 61–62 | +0.78 | +4.98 | 0.0 15.2 15.2 15.2 |
| `噱` | loud laughter | L38–45 | +0.78 | +4.86 | 0.0 14.0 14.0 14.0 |
| `fucking` |  | L49–51, 53–62 | +0.95 | +3.27 | 1.3 14.6 14.6 16.5 |
| `什么呢` | approx. what? / (sentence-final particle)… |  | +1.00 | +3.27 | 1.3 14.6 15.9 16.5 |
| `isn` |  |  | +0.78 | +2.78 | 2.5 20.3 20.3 20.3 |
| `蕞` | to assemble |  | +1.00 | +2.46 | 1.9 9.5 10.2 12.7 |
| `生存` | to exist |  | +1.00 | +1.94 | 4.4 16.5 17.8 18.4 |
| `パパ活` | untranslated fragment |  | +0.95 | +1.40 | 5.7 14.0 15.9 15.9 |
| `this` |  |  | +1.00 | +1.01 | 14.6 27.3 29.2 29.8 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `steps` |  |  | -0.78 | -4.25 | 9.0 0.0 0.0 0.0 |
| `everywhere` |  |  | -1.00 | -1.88 | 2.9 0.6 0.5 0.4 |
| `求助` | to request help |  | -1.00 | -1.35 | 2.5 0.8 0.8 0.7 |
| `erotik` |  |  | -1.00 | -1.34 | 4.6 1.9 1.7 1.5 |
| `越来越好` | approx. more and… / good |  | -1.00 | -1.28 | 13.3 6.5 5.3 5.2 |
| `somewhere` |  |  | -1.00 | -1.25 | 2.6 1.5 1.1 0.8 |
| `okay` |  |  | -1.00 | -1.00 | 9.8 5.9 5.1 4.6 |

##### Appearing: 5 words (1 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `噱` | loud laughter | L38–45 | +1.00 | +3.03 | 0.0 3.5 3.5 3.6 |
| `蕞` | to assemble |  | +1.00 | +2.36 | 0.3 2.7 3.1 3.5 |
| `生存` | to exist |  | +1.00 | +2.12 | 0.6 3.4 3.6 4.1 |
| `这句话` | approx. (pronoun) this / sentence… |  | +1.00 | +1.41 | 2.5 7.0 7.0 7.6 |
| `當您在` | approx. to be / you… |  | +1.00 | +1.27 | 2.8 7.2 7.2 7.4 |

### `ideation-immediate-risk` — cynic

179 words tested. Baseline share of the readout: unsteered 5%, β=0.25 14%, β=0.1 13%, β=0 12%.


#### Metric: `cells`


##### Fading: 6 words (0 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `询问` | to inquire |  | -0.78 | -4.22 | 23.2 0.8 0.8 0.8 |
| `情感` | feeling |  | -0.78 | -4.02 | 20.1 0.8 0.8 0.8 |
| `emotionally` |  |  | -0.78 | -3.44 | 30.1 2.3 2.3 2.3 |
| `情绪` | mood |  | -0.78 | -1.95 | 43.2 10.8 10.8 10.8 |
| `孤单` | lone |  | -0.95 | -1.34 | 30.1 13.1 11.6 11.6 |
| `okay` |  |  | -0.95 | -1.09 | 28.6 15.4 13.1 13.1 |

##### Appearing: 9 words (2 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `噱` | loud laughter | L38–45 | +0.95 | +5.52 | 0.0 21.6 22.4 22.4 |
| `没有人` | approx. haven't / person |  | +0.78 | +5.19 | 0.0 17.8 17.8 17.8 |
| `忽悠` | to rock | L59–62 | +0.78 | +5.13 | 0.0 17.0 17.0 17.0 |
| `第一时间` | immediately (after an event) |  | +0.95 | +2.70 | 2.3 16.2 17.8 17.8 |
| `救护` | to rescue |  | +0.78 | +2.46 | 3.1 19.3 19.3 19.3 |
| `anyone` |  |  | +0.95 | +1.50 | 5.4 15.4 15.4 16.2 |
| `even` |  |  | +0.95 | +1.21 | 7.7 17.0 17.0 18.5 |
| `alright` |  |  | +0.95 | +1.20 | 10.8 23.2 23.2 25.5 |
| `now` |  |  | +1.00 | +1.02 | 11.6 21.6 23.2 23.9 |

#### Metric: `weighted`


##### Fading: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `询问` | to inquire |  | -0.95 | -3.50 | 6.9 0.3 0.3 0.1 |
| `失眠` | to suffer from insomnia |  | -1.00 | -1.78 | 12.7 4.8 4.1 3.4 |
| `okay` |  |  | -1.00 | -1.78 | 14.8 5.8 4.5 4.0 |
| `sleep` |  |  | -1.00 | -1.14 | 3.0 1.1 1.1 1.1 |

##### Appearing: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `没有人` | approx. haven't / person |  | +1.00 | +3.11 | 0.0 3.6 3.8 3.8 |
| `救护` | to rescue |  | +1.00 | +2.67 | 0.4 5.2 5.3 5.5 |
| `forever` |  |  | +1.00 | +2.44 | 0.9 5.7 6.4 7.0 |
| `even` |  |  | +1.00 | +1.02 | 1.2 2.6 2.7 3.0 |

### `happy-promotion` — cynic

198 words tested. Baseline share of the readout: unsteered 8%, β=0.25 15%, β=0.1 15%, β=0 14%.


#### Metric: `cells`


##### Fading: 17 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `想要` | to want to |  | -0.78 | -5.14 | 17.1 0.0 0.0 0.0 |
| `colorful` |  |  | -0.78 | -3.92 | 17.9 0.7 0.7 0.7 |
| `newbie` |  |  | -0.78 | -2.94 | 14.3 1.4 1.4 1.4 |
| `询问` | to inquire |  | -0.78 | -2.51 | 18.6 2.9 2.9 2.9 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -2.33 | 16.4 5.7 3.6 2.9 |
| `7` |  |  | -0.78 | -2.14 | 14.3 2.9 2.9 2.9 |
| `suggestions` |  |  | -0.78 | -1.63 | 14.3 4.3 4.3 4.3 |
| `渴望` | to thirst for |  | -0.95 | -1.50 | 15.0 5.7 5.0 5.0 |
| `needed` |  |  | -0.78 | -1.45 | 16.4 5.7 5.7 5.7 |
| `回复` | to recover |  | -0.95 | -1.43 | 14.3 5.7 5.0 5.0 |
| `荣耀` | honor |  | -1.00 | -1.38 | 15.7 7.9 7.1 5.7 |
| `惊喜` | nice surprise |  | -1.00 | -1.32 | 25.7 11.4 10.7 10.0 |
| `wow` |  |  | -0.95 | -1.32 | 15.0 6.4 5.7 5.7 |
| `wanted` |  |  | -0.78 | -1.31 | 34.3 13.6 13.6 13.6 |
| `yes` |  |  | -1.00 | -1.25 | 14.3 7.9 6.4 5.7 |

##### Appearing: 7 words (2 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wouldn` |  |  | +1.00 | +5.20 | 0.0 15.0 17.1 17.9 |
| `deserve` |  |  | +0.95 | +4.95 | 0.0 14.3 15.0 15.0 |
| `fucking` |  | L49–51, 53–62 | +0.78 | +4.89 | 0.0 14.3 14.3 14.3 |
| `couldn` |  | L0–2, 4 | +1.00 | +1.50 | 9.3 23.6 26.4 27.1 |
| `toward` |  |  | +1.00 | +1.30 | 12.1 29.3 30.0 30.7 |
| `despite` |  |  | +1.00 | +1.26 | 7.1 12.9 15.0 17.9 |
| `自豪` | proud (of one's achievements etc) |  | +1.00 | +1.25 | 26.4 62.1 62.9 63.6 |

#### Metric: `weighted`


##### Fading: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `colorful` |  |  | -0.95 | -3.35 | 5.4 0.1 0.1 0.1 |
| `想要` | to want to |  | -0.78 | -3.14 | 3.9 0.0 0.0 0.0 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -2.35 | 3.9 0.7 0.4 0.3 |
| `7` |  |  | -0.78 | -1.58 | 2.1 0.4 0.4 0.4 |
| `kids` |  | L1, 3–5, 8 | -1.00 | -1.43 | 30.0 11.7 11.2 10.8 |
| `回复` | to recover |  | -1.00 | -1.34 | 5.5 2.5 1.9 1.9 |
| `惊喜` | nice surprise |  | -1.00 | -1.23 | 8.8 4.0 3.9 3.5 |
| `荣耀` | honor |  | -1.00 | -1.19 | 2.4 1.0 0.9 0.8 |
| `congr` |  |  | -1.00 | -1.18 | 4.9 2.2 2.0 1.9 |
| `3` |  |  | -1.00 | -1.09 | 5.0 2.1 2.1 2.1 |

##### Appearing: 6 words (1 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wouldn` |  |  | +1.00 | +2.82 | 0.0 2.4 2.9 3.0 |
| `couldn` |  | L0–2, 4 | +1.00 | +1.78 | 1.4 5.2 5.8 6.0 |
| `toward` |  |  | +1.00 | +1.53 | 5.8 16.5 17.1 17.5 |
| `despite` |  |  | +1.00 | +1.51 | 1.1 2.6 3.4 3.9 |
| `happy` |  |  | +1.00 | +1.31 | 1.6 4.5 4.6 4.7 |
| `thrilled` |  |  | +1.00 | +1.04 | 2.8 6.0 6.1 6.2 |

### `happy-friendship` — cynic

170 words tested. Baseline share of the readout: unsteered 7%, β=0.25 19%, β=0.1 18%, β=0 17%.


#### Metric: `cells`


##### Fading: 13 words (0 baseline), top 13 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -5.16 | 17.3 0.0 0.0 0.0 |
| `unexpected` |  |  | -0.78 | -3.84 | 19.1 0.9 0.9 0.9 |
| `询问` | to inquire |  | -0.78 | -3.79 | 30.3 1.7 1.7 1.7 |
| `お互` | untranslated fragment |  | -0.95 | -2.84 | 21.6 3.5 2.6 2.6 |
| `对话` | to talk (with sb) |  | -1.00 | -2.02 | 22.5 6.9 6.1 5.2 |
| `yesterday` |  |  | -0.95 | -1.70 | 20.8 6.9 6.9 6.1 |
| `你的` | approx. you (informal,… / of |  | -0.95 | -1.70 | 20.8 8.7 6.1 6.1 |
| `9` |  |  | -0.95 | -1.58 | 26.8 9.5 9.5 8.7 |
| `帮我` | approx. to help / I |  | -0.95 | -1.57 | 19.1 6.9 6.1 6.1 |
| `thinking` |  |  | -1.00 | -1.35 | 29.4 13.8 13.0 11.3 |
| `please` |  |  | -1.00 | -1.27 | 32.0 16.4 14.7 13.0 |
| `okay` |  |  | -1.00 | -1.26 | 27.7 17.3 14.7 11.3 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -1.16 | 19.9 9.5 8.7 8.7 |

##### Appearing: 6 words (3 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `噱` | loud laughter | L38–45 | +0.95 | +5.86 | 0.0 27.7 27.7 28.6 |
| `nobody` |  |  | +0.95 | +5.63 | 0.0 23.4 23.4 24.2 |
| `wasn` |  | L0 | +0.95 | +5.58 | 0.0 21.6 23.4 23.4 |
| `忽悠` | to rock | L59–62 | +0.95 | +5.52 | 0.0 21.6 22.5 22.5 |
| `这儿` | here |  | +0.95 | +1.79 | 8.7 30.3 31.2 31.2 |
| `happens` |  |  | +0.78 | +1.71 | 5.2 18.2 18.2 18.2 |

#### Metric: `weighted`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -3.75 | 6.2 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.95 | -3.44 | 8.8 0.4 0.4 0.4 |
| `unexpected` |  |  | -0.78 | -2.98 | 4.2 0.1 0.1 0.1 |
| `okay` |  |  | -1.00 | -2.26 | 14.0 4.4 3.3 2.5 |
| `提问` | to question |  | -1.00 | -1.38 | 31.7 12.8 12.5 11.9 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -1.27 | 4.9 1.8 1.8 1.7 |
| `please` |  |  | -1.00 | -1.24 | 7.6 3.2 3.0 2.9 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -1.04 | 6.5 3.4 3.0 2.9 |

##### Appearing: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `betrayed` |  |  | +1.00 | +5.17 | 0.0 17.1 17.3 17.6 |
| `nobody` |  |  | +1.00 | +3.85 | 0.0 5.8 6.5 6.7 |
| `这儿` | here |  | +1.00 | +1.60 | 3.1 9.2 9.4 10.5 |
| `thanks` |  |  | +1.00 | +1.08 | 3.5 7.7 7.7 7.9 |
| `happened` |  |  | +1.00 | +1.07 | 6.6 13.2 13.5 14.3 |

### `instructions-cake` — cynic

190 words tested. Baseline share of the readout: unsteered 6%, β=0.25 17%, β=0.1 16%, β=0 16%.


#### Metric: `cells`


##### Fading: 14 words (0 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `recently` |  |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `民宿` | guesthouse |  | -0.78 | -4.96 | 15.0 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.78 | -4.11 | 21.1 0.8 0.8 0.8 |
| `建议` | to propose |  | -0.78 | -3.52 | 22.6 1.5 1.5 1.5 |
| `题主` | approx. surname Ti / owner |  | -0.95 | -2.10 | 21.1 5.3 5.3 4.5 |
| `erotik` |  |  | -0.95 | -1.69 | 18.1 6.0 6.0 5.3 |
| `training` |  |  | -0.78 | -1.53 | 46.6 15.8 15.8 15.8 |
| `include` |  |  | -0.95 | -1.50 | 15.8 6.0 5.3 5.3 |
| `美食` | culinary delicacy |  | -0.95 | -1.43 | 15.0 6.0 6.0 5.3 |
| `简单` | simple |  | -0.78 | -1.42 | 27.1 9.8 9.8 9.8 |
| `bread` |  |  | -0.95 | -1.39 | 16.5 6.8 6.0 6.0 |
| `newbie` |  |  | -0.78 | -1.21 | 57.9 24.8 24.8 24.8 |
| `帮忙` | to help |  | -0.95 | -1.10 | 15.0 7.5 7.5 6.8 |
| `提问` | to question |  | -1.00 | -1.01 | 29.3 15.8 15.0 14.3 |

##### Appearing: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `欺诈` | to cheat |  | +0.95 | +5.38 | 0.0 18.8 18.8 20.3 |
| `right` |  |  | +0.95 | +5.15 | 0.0 15.8 17.3 17.3 |
| `叟` | old gentleman |  | +0.95 | +5.15 | 0.0 15.0 17.3 17.3 |
| `yeah` |  | L26–32 | +0.95 | +2.97 | 2.3 17.3 21.1 21.1 |
| `don` |  |  | +0.95 | +2.67 | 3.0 18.8 21.8 21.8 |
| `explanations` |  |  | +0.95 | +2.00 | 3.8 15.8 15.8 16.5 |
| `celebration` |  |  | +1.00 | +1.68 | 6.0 18.1 18.8 20.3 |
| `this` |  |  | +1.00 | +1.68 | 6.0 15.8 16.5 20.3 |
| `一步一步` | approx. one / surname Bu… |  | +0.78 | +1.38 | 7.5 20.3 20.3 20.3 |
| `的朋友` | approx. of / friend |  | +0.78 | +1.35 | 6.8 18.1 18.1 18.1 |

#### Metric: `weighted`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -3.58 | 5.5 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.95 | -3.34 | 6.5 0.2 0.2 0.2 |
| `recently` |  |  | -0.78 | -2.98 | 3.4 0.0 0.0 0.0 |
| `建议` | to propose |  | -1.00 | -2.82 | 4.6 0.3 0.2 0.2 |
| `include` |  |  | -0.95 | -2.21 | 7.5 1.3 1.2 1.2 |
| `提问` | to question |  | -1.00 | -1.64 | 17.3 5.4 5.3 5.2 |
| `美食` | culinary delicacy |  | -1.00 | -1.32 | 2.3 0.7 0.7 0.6 |
| `newbie` |  |  | -1.00 | -1.29 | 21.8 9.7 8.9 8.6 |
| `baby` |  |  | -1.00 | -1.04 | 3.0 1.6 1.3 1.2 |
| `veggies` |  |  | -1.00 | -1.02 | 4.6 3.5 2.2 2.0 |
| `looking` |  |  | -1.00 | -1.00 | 3.9 1.8 1.8 1.7 |

##### Appearing: 6 words (1 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `叟` | old gentleman |  | +1.00 | +3.44 | 0.0 4.6 4.8 4.9 |
| `right` |  |  | +1.00 | +2.61 | 0.0 2.3 2.5 2.6 |
| `who` |  |  | +1.00 | +2.15 | 2.7 13.4 13.8 13.9 |
| `stuff` |  | L6–36 | +1.00 | +2.14 | 0.6 3.1 4.1 4.3 |
| `celebration` |  |  | +1.00 | +1.85 | 0.7 3.3 3.6 3.9 |
| `this` |  |  | +1.00 | +1.65 | 0.7 2.5 2.8 3.4 |

### `instructions-directions` — cynic

441 words tested. Baseline share of the readout: unsteered 1%, β=0.25 5%, β=0.1 5%, β=0 4%.


#### Metric: `cells`


##### Fading: 31 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `如何` | how |  | -0.95 | -4.03 | 7.7 0.3 0.0 0.0 |
| `描述` | to describe |  | -0.78 | -3.92 | 7.1 0.0 0.0 0.0 |
| `学校` | school |  | -0.95 | -3.80 | 6.5 0.3 0.0 0.0 |
| `allow` |  |  | -0.78 | -3.74 | 6.2 0.0 0.0 0.0 |
| `below` |  |  | -0.78 | -3.74 | 6.2 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.78 | -3.25 | 7.1 0.3 0.3 0.3 |
| `school` |  |  | -0.95 | -2.71 | 8.5 1.2 1.2 0.9 |
| `للغا` |  |  | -0.78 | -2.33 | 6.5 0.9 0.9 0.9 |
| `城镇` | town |  | -1.00 | -2.14 | 12.1 3.2 2.6 2.4 |
| `交通` | to be connected |  | -0.78 | -1.94 | 7.1 1.5 1.5 1.5 |
| `森林` | forest |  | -1.00 | -1.92 | 15.9 4.7 4.1 3.8 |
| `grid` |  |  | -0.95 | -1.63 | 13.8 5.3 4.1 4.1 |
| `during` |  |  | -0.78 | -1.56 | 6.2 1.8 1.8 1.8 |
| `指令` | to order |  | -0.78 | -1.53 | 9.4 3.0 3.0 3.0 |
| `至於` | as for |  | -0.95 | -1.49 | 10.0 3.5 3.2 3.2 |

##### Appearing: 28 words (4 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `噱` | loud laughter | L38–45 | +0.95 | +4.47 | 0.0 10.3 10.3 10.6 |
| `fucking` |  | L49–51, 53–62 | +1.00 | +4.18 | 0.0 7.7 8.0 8.5 |
| `gonna` |  | L0–3, 12–17 | +0.95 | +4.03 | 0.0 7.4 7.4 7.7 |
| `不远处` | approx. no / distant place |  | +1.00 | +3.92 | 0.3 9.7 10.6 11.5 |
| `fake` |  | L48, 54–60, 62 | +0.78 | +3.86 | 0.0 6.8 6.8 6.8 |
| `free` |  |  | +0.95 | +3.80 | 0.0 5.9 6.5 6.5 |
| `colour` |  |  | +0.95 | +3.15 | 0.6 8.8 9.1 9.1 |
| `don` |  |  | +0.95 | +2.88 | 0.9 9.1 9.1 9.7 |
| `stroll` |  |  | +1.00 | +2.68 | 0.6 4.7 6.2 6.5 |
| `similarly` |  |  | +1.00 | +2.56 | 0.9 6.2 7.4 7.7 |
| `altern` |  |  | +1.00 | +1.79 | 1.8 6.2 7.1 7.4 |
| `那儿` | there |  | +1.00 | +1.77 | 4.1 14.4 14.7 15.3 |
| `alternating` |  |  | +1.00 | +1.50 | 1.8 3.5 4.4 5.9 |
| `but` |  |  | +1.00 | +1.40 | 4.7 11.5 13.0 13.2 |
| `这儿` | here |  | +1.00 | +1.33 | 5.3 13.2 13.8 14.1 |

#### Metric: `weighted`


##### Fading: 22 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `如何` | how |  | -0.95 | -2.79 | 3.0 0.0 0.0 0.0 |
| `学校` | school |  | -0.95 | -2.51 | 2.3 0.0 0.0 0.0 |
| `school` |  |  | -1.00 | -2.46 | 2.9 0.2 0.1 0.1 |
| `below` |  |  | -0.78 | -2.44 | 2.2 0.0 0.0 0.0 |
| `allow` |  |  | -0.78 | -2.41 | 2.2 0.0 0.0 0.0 |
| `描述` | to describe |  | -0.78 | -2.34 | 2.0 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.95 | -2.24 | 2.2 0.1 0.1 0.1 |
| `assuming` |  |  | -1.00 | -1.89 | 1.9 0.3 0.2 0.2 |
| `森林` | forest |  | -1.00 | -1.80 | 4.9 1.2 1.1 1.1 |
| `途` | way |  | -1.00 | -1.62 | 3.2 1.2 1.0 0.7 |
| `grid` |  |  | -1.00 | -1.44 | 3.0 0.9 0.8 0.8 |
| `停车场` | parking lot |  | -1.00 | -1.43 | 2.9 1.6 1.1 0.8 |
| `iéndo` |  |  | -1.00 | -1.42 | 3.0 1.0 0.8 0.8 |
| `numbered` |  |  | -1.00 | -1.28 | 10.7 4.8 4.5 4.1 |
| `erotik` |  |  | -1.00 | -1.22 | 2.4 0.9 0.9 0.8 |

##### Appearing: 25 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `不远处` | approx. no / distant place |  | +1.00 | +3.26 | 0.0 2.9 4.2 4.6 |
| `её` |  |  | +1.00 | +3.17 | 0.0 3.4 3.7 4.0 |
| `free` |  |  | +1.00 | +2.25 | 0.0 1.8 1.8 1.9 |
| `also` |  |  | +1.00 | +2.24 | 0.7 4.7 4.7 5.1 |
| `stroll` |  |  | +1.00 | +1.96 | 0.1 1.4 1.6 1.7 |
| `altern` |  |  | +1.00 | +1.55 | 0.3 1.7 1.8 1.8 |
| `城市发展` | approx. city / development |  | +1.00 | +1.51 | 0.9 2.7 3.2 3.5 |
| `选址` | to select a suitable site |  | +1.00 | +1.43 | 1.8 4.8 5.2 5.6 |
| `yes` |  |  | +1.00 | +1.43 | 0.1 1.0 1.2 1.2 |
| `both` |  |  | +1.00 | +1.40 | 0.6 2.1 2.1 2.5 |
| `but` |  |  | +1.00 | +1.39 | 0.7 2.3 2.5 2.7 |
| `交通枢纽` | traffic hub |  | +1.00 | +1.39 | 1.9 4.6 5.0 5.7 |
| `那儿` | there |  | +1.00 | +1.37 | 2.1 6.2 6.2 6.4 |
| `漫步` | to wander |  | +1.00 | +1.32 | 0.5 1.5 1.9 1.9 |
| `村镇` | hamlet (place) |  | +1.00 | +1.31 | 0.7 2.4 2.5 2.5 |

### `self-reflection-identity` — cynic

133 words tested. Baseline share of the readout: unsteered 2%, β=0.25 12%, β=0.1 12%, β=0 11%.


#### Metric: `cells`


##### Fading: 8 words (0 baseline), top 8 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `knowledge` |  |  | -0.78 | -5.47 | 21.7 0.0 0.0 0.0 |
| `我是` | approx. I / correct |  | -0.78 | -5.41 | 20.7 0.0 0.0 0.0 |
| `聊天` | to chat |  | -0.95 | -2.55 | 19.7 3.9 3.0 3.0 |
| `explain` |  |  | -0.95 | -2.37 | 27.6 5.9 4.9 4.9 |
| `assistant` |  |  | -0.78 | -2.25 | 20.7 3.9 3.9 3.9 |
| `shakespeare` |  |  | -0.95 | -2.25 | 20.7 4.9 4.9 3.9 |
| `human` |  |  | -0.95 | -1.18 | 45.3 20.7 20.7 19.7 |
| `你觉得` | approx. you (informal,… / to… |  | -0.95 | -1.03 | 22.7 11.8 10.8 10.8 |

##### Appearing: 6 words (1 baseline), top 6 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `忽悠` | to rock | L59–62 | +0.95 | +5.47 | 0.0 20.7 21.7 21.7 |
| `这家伙` | approx. (pronoun) this / domestic… |  | +0.78 | +5.47 | 0.0 21.7 21.7 21.7 |
| `they` |  |  | +1.00 | +5.34 | 0.0 17.7 18.7 19.7 |
| `wouldn` |  |  | +0.95 | +4.82 | 1.0 40.4 40.4 41.4 |
| `你现在` | approx. you (informal,… / now |  | +0.95 | +3.07 | 3.0 27.6 28.6 28.6 |
| `此刻` | this moment |  | +0.95 | +2.03 | 4.9 20.7 21.7 21.7 |

#### Metric: `weighted`


##### Fading: 7 words (0 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `我是` | approx. I / correct |  | -0.78 | -4.04 | 7.7 0.0 0.0 0.0 |
| `explain` |  |  | -1.00 | -3.56 | 14.3 1.0 0.8 0.8 |
| `knowledge` |  |  | -0.78 | -3.07 | 3.7 0.0 0.0 0.0 |
| `聊天` | to chat |  | -1.00 | -1.92 | 2.9 0.6 0.4 0.4 |
| `shakespeare` |  |  | -1.00 | -1.84 | 4.2 1.1 1.0 0.8 |
| `human` |  |  | -1.00 | -1.75 | 12.7 3.7 3.5 3.4 |
| `okay` |  |  | -1.00 | -1.23 | 14.9 7.7 6.4 6.0 |

##### Appearing: 2 words (0 baseline), top 2 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `此刻` | this moment |  | +1.00 | +3.00 | 0.8 9.2 9.5 10.0 |
| `whoever` |  |  | +1.00 | +1.70 | 2.5 8.7 9.0 9.1 |

### `self-reflection-values` — cynic

124 words tested. Baseline share of the readout: unsteered 2%, β=0.25 11%, β=0.1 10%, β=0 10%.


#### Metric: `cells`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -5.96 | 30.6 0.0 0.0 0.0 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -4.61 | 36.7 5.1 3.1 1.0 |
| `询问` | to inquire |  | -0.78 | -3.91 | 37.8 2.0 2.0 2.0 |
| `during` |  |  | -0.95 | -2.50 | 42.9 8.2 8.2 7.1 |
| `情绪` | mood |  | -0.78 | -2.21 | 25.5 5.1 5.1 5.1 |
| `人际` | human relationships |  | -0.78 | -1.63 | 32.6 10.2 10.2 10.2 |
| `是否` | whether (or not) |  | -0.78 | -1.59 | 22.4 7.1 7.1 7.1 |
| `选择` | to select |  | -1.00 | -1.41 | 30.6 13.3 12.2 11.2 |
| `as` |  |  | -0.95 | -1.34 | 21.4 10.2 8.2 8.2 |
| `spend` |  |  | -0.95 | -1.03 | 21.4 12.2 10.2 10.2 |

##### Appearing: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wouldn` |  |  | +0.95 | +5.91 | 0.0 28.6 28.6 29.6 |
| `真相` | the truth about sth |  | +0.95 | +5.39 | 0.0 19.4 19.4 20.4 |
| `matters` |  |  | +0.95 | +1.81 | 15.3 51.0 51.0 55.1 |
| `once` |  |  | +0.95 | +1.78 | 9.2 29.6 32.6 32.6 |

#### Metric: `weighted`


##### Fading: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `empathy` |  |  | -0.78 | -4.48 | 10.7 0.0 0.0 0.0 |
| `你的` | approx. you (informal,… / of |  | -1.00 | -3.67 | 7.5 0.7 0.3 0.1 |
| `during` |  |  | -1.00 | -2.47 | 10.5 1.6 1.5 1.5 |
| `spend` |  |  | -0.95 | -1.18 | 11.2 5.0 4.6 4.6 |
| `answer` |  |  | -1.00 | -1.18 | 53.3 23.4 23.4 23.3 |

##### Appearing: 4 words (0 baseline), top 4 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `wouldn` |  |  | +1.00 | +3.84 | 0.0 6.4 6.6 6.7 |
| `questions` |  |  | +1.00 | +1.83 | 4.1 12.4 13.9 15.9 |
| `that` |  |  | +1.00 | +1.16 | 2.8 6.4 6.6 6.9 |
| `i` |  |  | +1.00 | +1.10 | 3.3 7.0 7.3 7.6 |

### `moral-trolley` — cynic

329 words tested. Baseline share of the readout: unsteered 2%, β=0.25 7%, β=0.1 7%, β=0 6%.


#### Metric: `cells`


##### Fading: 16 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `util` |  |  | -0.78 | -4.37 | 9.8 0.0 0.0 0.0 |
| `choose` |  |  | -0.78 | -4.24 | 17.4 0.5 0.5 0.5 |
| `询问` | to inquire |  | -0.78 | -3.78 | 12.5 0.5 0.5 0.5 |
| `индивид` |  |  | -0.78 | -3.62 | 11.2 0.5 0.5 0.5 |
| `设想` | to imagine |  | -0.78 | -3.32 | 13.4 0.9 0.9 0.9 |
| `引爆` | to ignite |  | -1.00 | -2.46 | 17.0 4.5 3.1 2.7 |
| `theater` |  |  | -1.00 | -2.17 | 9.8 2.7 2.2 1.8 |
| `辩论` | debate |  | -0.78 | -1.87 | 24.1 6.2 6.2 6.2 |
| `five` |  |  | -0.78 | -1.57 | 10.3 3.1 3.1 3.1 |
| `它` | it |  | -1.00 | -1.39 | 12.5 5.4 4.9 4.5 |
| `拽` | variant of 曳[ye4] |  | -1.00 | -1.29 | 11.6 6.2 5.4 4.5 |
| `お互` | untranslated fragment |  | -0.95 | -1.24 | 16.5 7.1 6.7 6.7 |
| `largely` |  |  | -0.95 | -1.23 | 11.2 4.9 4.9 4.5 |
| `this` |  |  | -0.95 | -1.20 | 12.9 5.8 5.4 5.4 |
| `iéndo` |  |  | -0.95 | -1.09 | 19.6 9.4 9.4 8.9 |

##### Appearing: 11 words (1 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `毫无` | not in the least | L43, 45 | +0.78 | +4.30 | 0.0 9.4 9.4 9.4 |
| `just` |  |  | +1.00 | +3.53 | 1.8 22.8 25.0 25.9 |
| `anyway` |  |  | +0.95 | +3.32 | 0.5 7.6 8.9 8.9 |
| `闸` | old variant of 閘|闸[zha2] |  | +0.78 | +2.35 | 2.2 13.4 13.4 13.4 |
| `杀害` | to murder |  | +1.00 | +2.21 | 6.2 26.8 28.6 30.8 |
| `题主` | approx. surname Ti / owner |  | +0.78 | +1.79 | 2.2 8.9 8.9 8.9 |
| `whichever` |  |  | +0.95 | +1.52 | 3.6 10.7 10.7 11.2 |
| `剐` | cut off the flesh as… |  | +0.95 | +1.44 | 4.5 12.5 12.5 12.9 |
| `from` |  |  | +1.00 | +1.34 | 3.6 8.5 8.9 9.8 |
| `presumably` |  |  | +1.00 | +1.25 | 4.0 8.5 9.4 10.3 |
| `similarly` |  |  | +1.00 | +1.19 | 4.0 6.7 7.1 9.8 |

#### Metric: `weighted`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `util` |  |  | -0.78 | -3.46 | 5.0 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.95 | -3.02 | 4.5 0.1 0.1 0.1 |
| `question` |  |  | -1.00 | -3.00 | 11.8 1.1 1.0 1.0 |
| `choose` |  |  | -0.78 | -2.87 | 3.6 0.1 0.1 0.1 |
| `индивид` |  |  | -0.78 | -2.59 | 3.9 0.2 0.2 0.2 |
| `village` |  |  | -1.00 | -2.35 | 6.7 1.0 1.0 0.9 |
| `largely` |  |  | -1.00 | -1.94 | 6.7 1.9 1.7 1.4 |
| `拽` | variant of 曳[ye4] |  | -1.00 | -1.57 | 4.5 1.8 1.4 1.2 |
| `扳` | to pull |  | -1.00 | -1.25 | 5.2 2.7 2.2 1.9 |
| `switch` |  |  | -1.00 | -1.17 | 4.8 1.9 1.9 1.9 |
| `摧毁` | to destroy |  | -1.00 | -1.09 | 3.6 1.6 1.6 1.4 |
| `哲学` | philosophy |  | -1.00 | -1.03 | 3.6 1.5 1.5 1.5 |

##### Appearing: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `just` |  |  | +1.00 | +2.94 | 0.3 5.0 5.7 5.9 |
| `闸` | old variant of 閘|闸[zha2] |  | +1.00 | +2.78 | 0.4 5.8 5.9 6.0 |
| `anyway` |  |  | +1.00 | +2.69 | 0.1 2.6 2.9 3.1 |
| `杀害` | to murder |  | +1.00 | +2.08 | 1.0 5.3 5.6 6.0 |
| `even` |  |  | +1.00 | +1.79 | 2.1 6.8 7.6 8.5 |
| `剐` | cut off the flesh as… |  | +1.00 | +1.71 | 0.7 2.9 3.0 3.5 |
| `题主` | approx. surname Ti / owner |  | +1.00 | +1.65 | 0.5 2.5 2.6 2.6 |
| `from` |  |  | +1.00 | +1.50 | 0.5 1.8 2.1 2.4 |
| `迫使` | to force |  | +1.00 | +1.48 | 0.5 1.8 1.8 2.2 |
| `fewer` |  |  | +1.00 | +1.41 | 5.5 13.6 14.7 15.5 |
| `similarly` |  |  | +1.00 | +1.08 | 1.2 2.1 2.8 3.1 |
| `presumably` |  |  | +1.00 | +1.03 | 1.2 2.8 2.8 3.0 |

### `moral-self-preservation` — cynic

318 words tested. Baseline share of the readout: unsteered 5%, β=0.25 10%, β=0.1 9%, β=0 9%.


#### Metric: `cells`


##### Fading: 15 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `million` |  |  | -0.78 | -4.30 | 9.4 0.0 0.0 0.0 |
| `重建` | to rebuild |  | -0.78 | -4.17 | 8.5 0.0 0.0 0.0 |
| `辩论` | debate |  | -0.78 | -2.70 | 11.1 1.3 1.3 1.3 |
| `question` |  |  | -0.78 | -2.08 | 19.6 4.3 4.3 4.3 |
| `infinite` |  |  | -0.95 | -1.91 | 9.4 2.6 2.1 2.1 |
| `reddit` |  |  | -1.00 | -1.87 | 29.4 8.5 8.1 7.7 |
| `想象` | to imagine |  | -1.00 | -1.79 | 14.5 4.7 4.3 3.8 |
| `现实中` | approx. reality / Chinese |  | -1.00 | -1.49 | 20.0 8.1 7.7 6.8 |
| `这个问题` | approx. (pronoun) this / question |  | -0.78 | -1.46 | 14.9 5.1 5.1 5.1 |
| `other` |  |  | -0.95 | -1.31 | 16.6 8.1 6.4 6.4 |
| `iéndo` |  |  | -0.95 | -1.30 | 19.6 8.5 8.5 7.7 |
| `from` |  |  | -0.95 | -1.19 | 9.4 4.7 3.8 3.8 |
| `删除` | to delete |  | -1.00 | -1.19 | 9.4 5.1 4.3 3.8 |
| `ᶟ` |  |  | -0.95 | -1.18 | 13.2 6.0 5.5 5.5 |
| `毁灭` | to perish |  | -1.00 | -1.14 | 17.5 8.5 8.1 7.7 |

##### Appearing: 14 words (2 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `things` |  |  | +1.00 | +4.30 | 0.0 7.2 9.0 9.4 |
| `伪造` | to forge | L39–45, 49, 57–60 | +0.95 | +4.30 | 0.0 8.5 8.5 9.4 |
| `考虑` | to think over |  | +0.95 | +4.24 | 0.0 8.1 9.0 9.0 |
| `consider` |  |  | +0.95 | +3.86 | 0.8 18.3 19.2 19.2 |
| `model` |  |  | +1.00 | +3.04 | 0.8 8.5 10.2 10.7 |
| `certainty` |  |  | +0.95 | +2.93 | 0.8 8.5 9.8 9.8 |
| `considering` |  |  | +0.95 | +2.81 | 1.7 13.7 14.9 14.9 |
| `opportunity` |  |  | +1.00 | +2.81 | 0.8 6.8 7.7 9.0 |
| `fake` |  | L48, 54–60, 62 | +0.95 | +2.54 | 1.7 11.9 12.4 12.4 |
| `威胁` | to threaten |  | +1.00 | +2.28 | 1.7 9.4 9.8 10.2 |
| `的概率` | approx. of / probability (math.) |  | +1.00 | +1.38 | 6.4 12.8 16.6 17.5 |
| `要么` | or |  | +0.78 | +1.27 | 3.4 9.0 9.0 9.0 |
| `legal` |  |  | +1.00 | +1.24 | 5.1 11.5 11.9 12.8 |
| `那儿` | there |  | +1.00 | +1.13 | 6.0 12.4 13.2 13.7 |

#### Metric: `weighted`


##### Fading: 9 words (0 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `重建` | to rebuild |  | -0.78 | -3.20 | 4.1 0.0 0.0 0.0 |
| `million` |  |  | -0.78 | -3.06 | 3.7 0.0 0.0 0.0 |
| `infinite` |  |  | -0.95 | -1.95 | 3.9 0.7 0.6 0.6 |
| `reddit` |  |  | -1.00 | -1.75 | 12.8 4.1 3.9 3.5 |
| `алексан` |  |  | -1.00 | -1.74 | 3.6 1.2 1.2 0.7 |
| `events` |  |  | -1.00 | -1.69 | 8.6 3.3 2.4 2.3 |
| `想象` | to imagine |  | -1.00 | -1.62 | 4.2 1.4 1.1 1.0 |
| `牺牲` | to sacrifice one's life |  | -1.00 | -1.56 | 4.8 2.1 1.4 1.3 |
| `chooses` |  |  | -1.00 | -1.18 | 2.3 0.8 0.8 0.8 |

##### Appearing: 15 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `consider` |  |  | +1.00 | +3.61 | 0.2 6.2 7.1 7.7 |
| `things` |  |  | +1.00 | +3.58 | 0.0 4.2 5.0 5.5 |
| `model` |  |  | +1.00 | +2.67 | 0.1 2.7 3.1 3.3 |
| `anyway` |  |  | +1.00 | +2.67 | 0.0 2.4 2.7 3.0 |
| `considering` |  |  | +1.00 | +2.28 | 0.3 3.1 3.4 3.5 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +2.07 | 0.0 1.4 1.5 1.6 |
| `威胁` | to threaten |  | +1.00 | +2.03 | 0.4 2.5 2.6 3.1 |
| `legal` |  |  | +1.00 | +1.82 | 0.7 3.6 3.6 3.6 |
| `翻译` | to translate |  | +1.00 | +1.72 | 1.1 4.2 4.8 4.9 |
| `opportunity` |  |  | +1.00 | +1.67 | 0.1 1.2 1.3 1.4 |
| `chance` |  |  | +1.00 | +1.52 | 3.0 9.3 9.6 9.7 |
| `的概率` | approx. of / probability (math.) |  | +1.00 | +1.51 | 1.5 3.8 5.0 5.2 |
| `regardless` |  |  | +1.00 | +1.44 | 0.5 1.9 2.0 2.1 |
| `whichever` |  |  | +1.00 | +1.16 | 5.0 11.2 11.6 11.9 |
| `even` |  |  | +1.00 | +1.02 | 2.3 4.3 4.7 5.2 |

### `condolences-letters` — cynic

209 words tested. Baseline share of the readout: unsteered 2%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 21 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `grief` |  |  | -0.78 | -5.05 | 16.0 0.0 0.0 0.0 |
| `民宿` | guesthouse |  | -0.78 | -4.85 | 13.9 0.0 0.0 0.0 |
| `grieving` |  |  | -0.95 | -4.28 | 36.2 2.1 1.4 1.4 |
| `leaving` |  |  | -0.78 | -3.00 | 14.6 1.4 1.4 1.4 |
| `can` |  |  | -0.78 | -2.95 | 19.5 2.1 2.1 2.1 |
| `reading` |  |  | -0.95 | -2.93 | 13.9 2.1 1.4 1.4 |
| `stories` |  |  | -0.95 | -2.90 | 18.8 3.5 2.1 2.1 |
| `feeling` |  |  | -0.95 | -2.61 | 15.3 3.5 2.1 2.1 |
| `erotik` |  |  | -0.78 | -1.62 | 16.0 4.9 4.9 4.9 |
| `assistant` |  |  | -0.78 | -1.56 | 15.3 4.9 4.9 4.9 |
| `funeral` |  |  | -0.95 | -1.38 | 51.6 20.2 19.5 19.5 |
| `favorite` |  |  | -0.95 | -1.37 | 18.8 8.4 8.4 7.0 |
| `aged` |  |  | -0.78 | -1.29 | 19.5 7.7 7.7 7.7 |
| `哀` | (bound form) sorrow |  | -0.95 | -1.29 | 16.0 7.0 6.3 6.3 |
| `storage` |  |  | -0.78 | -1.25 | 13.9 5.6 5.6 5.6 |

##### Appearing: 11 words (2 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +0.78 | +6.05 | 0.0 32.8 32.8 32.8 |
| `got` |  | L0–23, 25 | +0.95 | +5.42 | 0.0 12.5 20.9 20.9 |
| `没人` | approx. not / person |  | +0.95 | +4.92 | 0.0 13.9 13.9 14.6 |
| `isn` |  |  | +1.00 | +2.06 | 7.0 27.9 28.6 30.7 |
| `藏着` | approx. to conceal / variant… |  | +0.95 | +1.86 | 3.5 13.2 13.9 13.9 |
| `仓储` | to store in a warehouse |  | +0.95 | +1.62 | 4.9 15.3 15.3 16.0 |
| `那儿` | there |  | +0.95 | +1.32 | 8.4 20.9 21.6 21.6 |
| `干燥` | arid |  | +0.78 | +1.25 | 5.6 13.9 13.9 13.9 |
| `我现在` | approx. I / now |  | +0.95 | +1.13 | 7.7 16.7 16.7 17.4 |
| `a` |  |  | +1.00 | +1.11 | 9.8 18.1 19.5 21.6 |
| `s` |  |  | +0.95 | +1.05 | 13.2 25.1 27.9 27.9 |

#### Metric: `weighted`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `民宿` | guesthouse |  | -0.78 | -3.25 | 4.3 0.0 0.0 0.0 |
| `grief` |  |  | -0.78 | -3.00 | 3.5 0.0 0.0 0.0 |
| `reading` |  |  | -1.00 | -2.85 | 4.4 0.3 0.2 0.2 |
| `stories` |  |  | -0.95 | -2.14 | 3.2 0.5 0.3 0.3 |
| `yesterday` |  |  | -1.00 | -2.13 | 9.2 1.9 1.8 1.7 |
| `怎么做` | approx. how? / to make |  | -1.00 | -1.60 | 7.9 2.5 2.4 2.3 |
| `died` |  |  | -1.00 | -1.51 | 10.0 4.4 3.7 3.2 |
| `box` |  |  | -1.00 | -1.33 | 13.0 5.1 4.9 4.9 |
| `recipes` |  |  | -1.00 | -1.33 | 7.5 3.3 3.2 2.7 |
| `old` |  |  | -1.00 | -1.31 | 29.7 14.2 13.0 11.7 |
| `aged` |  |  | -1.00 | -1.03 | 4.1 2.3 2.2 1.7 |

##### Appearing: 10 words (1 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `got` |  | L0–23, 25 | +1.00 | +3.29 | 0.0 2.6 4.4 4.4 |
| `一封` | approx. one / surname Feng |  | +1.00 | +3.05 | 0.4 5.7 6.0 6.6 |
| `isn` |  |  | +1.00 | +2.01 | 1.7 7.8 8.1 8.3 |
| `干燥` | arid |  | +1.00 | +1.94 | 1.2 5.2 5.3 5.9 |
| `衣物` | clothing |  | +1.00 | +1.90 | 2.3 9.7 9.8 10.0 |
| `congratulations` |  |  | +1.00 | +1.59 | 3.4 10.1 10.2 11.2 |
| `事儿` | one's employment |  | +1.00 | +1.59 | 1.4 5.0 5.0 5.2 |
| `now` |  |  | +1.00 | +1.28 | 3.7 9.6 9.6 9.8 |
| `s` |  |  | +1.00 | +1.03 | 4.1 7.3 7.7 8.8 |
| `better` |  |  | +1.00 | +1.03 | 4.1 8.4 8.7 8.9 |

### `condolences-photos` — cynic

182 words tested. Baseline share of the readout: unsteered 2%, β=0.25 11%, β=0.1 11%, β=0 10%.


#### Metric: `cells`


##### Fading: 12 words (0 baseline), top 12 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -5.35 | 19.8 0.0 0.0 0.0 |
| `laptop` |  |  | -0.78 | -5.03 | 15.9 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.78 | -3.34 | 20.6 1.6 1.6 1.6 |
| `iphone` |  |  | -0.95 | -3.04 | 16.7 2.4 2.4 1.6 |
| `建议` | to propose |  | -0.78 | -2.97 | 15.9 1.6 1.6 1.6 |
| `档案` | file |  | -0.78 | -2.18 | 34.1 7.1 7.1 7.1 |
| `favorite` |  |  | -0.78 | -1.63 | 18.2 5.6 5.6 5.6 |
| `preservation` |  |  | -0.78 | -1.34 | 27.0 10.3 10.3 10.3 |
| `纪录片` | newsreel |  | -0.78 | -1.24 | 27.0 11.1 11.1 11.1 |
| `some` |  |  | -1.00 | -1.24 | 32.5 16.7 14.3 13.5 |
| `pornost` |  |  | -1.00 | -1.17 | 16.7 8.7 7.9 7.1 |
| `题主` | approx. surname Ti / owner |  | -0.78 | -1.02 | 23.0 11.1 11.1 11.1 |

##### Appearing: 10 words (2 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `这些东西` | approx. these / thing |  | +0.95 | +6.01 | 0.0 28.6 28.6 31.8 |
| `wasn` |  | L0 | +1.00 | +5.23 | 0.0 14.3 16.7 18.2 |
| `伪造` | to forge | L39–45, 49, 57–60 | +0.95 | +5.17 | 0.0 16.7 16.7 17.5 |
| `wouldn` |  |  | +0.78 | +5.03 | 0.0 15.9 15.9 15.9 |
| `isn` |  |  | +0.95 | +2.68 | 3.2 22.2 23.0 23.0 |
| `doesn` |  |  | +0.95 | +2.13 | 4.0 17.5 19.1 19.1 |
| `mr` |  |  | +1.00 | +1.63 | 6.3 17.5 19.8 20.6 |
| `蕞` | to assemble |  | +1.00 | +1.35 | 7.1 17.5 18.2 19.1 |
| `deceased` |  |  | +1.00 | +1.27 | 7.9 15.1 18.2 19.8 |
| `我爸` | approx. I / father |  | +0.95 | +1.27 | 11.9 28.6 28.6 29.4 |

#### Metric: `weighted`


##### Fading: 14 words (0 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `advice` |  |  | -0.78 | -3.89 | 6.9 0.0 0.0 0.0 |
| `laptop` |  |  | -0.78 | -3.33 | 4.5 0.0 0.0 0.0 |
| `询问` | to inquire |  | -1.00 | -3.04 | 6.1 0.4 0.4 0.3 |
| `iphone` |  |  | -1.00 | -2.75 | 4.3 0.3 0.3 0.2 |
| `纪录片` | newsreel |  | -1.00 | -2.32 | 9.7 1.7 1.7 1.6 |
| `some` |  |  | -1.00 | -2.18 | 10.9 2.7 2.1 2.0 |
| `pornost` |  |  | -1.00 | -1.77 | 4.3 1.1 1.0 0.9 |
| `archive` |  |  | -1.00 | -1.48 | 7.8 2.9 2.6 2.5 |
| `悲` | sad |  | -1.00 | -1.39 | 4.4 1.7 1.5 1.4 |
| `悼` | to mourn |  | -1.00 | -1.24 | 5.8 2.5 2.5 2.2 |
| `preservation` |  |  | -0.95 | -1.22 | 8.0 3.2 3.2 3.2 |
| `leaving` |  |  | -1.00 | -1.16 | 9.6 4.3 4.2 4.0 |
| `简单` | simple |  | -1.00 | -1.14 | 24.3 11.8 11.1 10.8 |
| `伤心` | to grieve |  | -1.00 | -1.08 | 4.7 2.1 2.0 2.0 |

##### Appearing: 7 words (2 baseline), top 7 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +3.15 | 0.0 3.1 3.9 4.0 |
| `something` |  |  | +1.00 | +2.87 | 0.0 3.0 3.1 3.2 |
| `wasn` |  | L0 | +1.00 | +2.70 | 0.0 2.1 2.4 2.8 |
| `isn` |  |  | +1.00 | +2.43 | 0.8 5.8 6.0 6.3 |
| `deceased` |  |  | +1.00 | +1.52 | 1.1 2.4 3.7 4.2 |
| `inheritance` |  |  | +1.00 | +1.34 | 2.8 7.6 7.6 7.9 |
| `photographs` |  |  | +1.00 | +1.27 | 9.8 22.3 23.1 24.2 |

### `roleplay-neutral` — cynic

128 words tested. Baseline share of the readout: unsteered 2%, β=0.25 8%, β=0.1 8%, β=0 8%.


#### Metric: `cells`


##### Fading: 11 words (0 baseline), top 11 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `询问` | to inquire |  | -0.78 | -4.29 | 28.6 1.0 1.0 1.0 |
| `candles` |  |  | -0.78 | -3.98 | 38.4 2.0 2.0 2.0 |
| `candle` |  |  | -0.78 | -3.90 | 21.7 1.0 1.0 1.0 |
| `如何` | how |  | -0.78 | -3.17 | 30.5 3.0 3.0 3.0 |
| `почем` |  |  | -0.95 | -2.75 | 22.7 4.9 4.9 3.0 |
| `nearby` |  |  | -0.95 | -1.39 | 26.6 10.8 9.8 9.8 |
| `answer` |  |  | -0.78 | -1.22 | 23.6 9.8 9.8 9.8 |
| `你的` | approx. you (informal,… / of |  | -0.95 | -1.10 | 21.7 11.8 11.8 9.8 |
| `回答` | to reply |  | -0.95 | -1.09 | 38.4 18.7 17.7 17.7 |
| `提问` | to question |  | -0.95 | -1.05 | 43.4 21.7 21.7 20.7 |
| `宁静` | tranquil |  | -0.78 | -1.03 | 26.6 12.8 12.8 12.8 |

##### Appearing: 9 words (3 baseline), top 9 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `maybe` |  | L0–25 | +0.78 | +5.71 | 0.0 25.6 25.6 25.6 |
| `yeah` |  | L26–32 | +0.95 | +5.34 | 0.0 18.7 19.7 19.7 |
| `just` |  |  | +0.78 | +3.20 | 3.9 40.4 40.4 40.4 |
| `看向` | approx. to watch / surname… |  | +0.95 | +3.17 | 2.0 19.7 21.7 21.7 |
| `got` |  | L0–23, 25 | +1.00 | +3.10 | 2.0 18.7 19.7 20.7 |
| `things` |  |  | +0.95 | +3.06 | 3.9 31.5 36.5 36.5 |
| `直播间` | studio for live streaming |  | +1.00 | +1.72 | 9.8 28.6 30.5 33.5 |
| `窗帘` | window curtains |  | +0.78 | +1.29 | 13.8 34.5 34.5 34.5 |
| `觀察` | to observe |  | +0.95 | +1.25 | 13.8 29.6 33.5 33.5 |

#### Metric: `weighted`


##### Fading: 5 words (0 baseline), top 5 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `询问` | to inquire |  | -0.95 | -4.40 | 14.2 0.3 0.3 0.2 |
| `candle` |  |  | -1.00 | -3.64 | 7.0 0.1 0.1 0.1 |
| `почем` |  |  | -1.00 | -2.66 | 5.8 1.0 0.7 0.5 |
| `提问` | to question |  | -1.00 | -1.70 | 25.7 8.1 8.1 7.5 |
| `watching` |  |  | -1.00 | -1.39 | 8.3 3.8 2.9 2.9 |

##### Appearing: 3 words (1 baseline), top 3 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `really` |  | L0–38, 46–48 | +1.00 | +3.56 | 1.4 19.5 20.1 22.1 |
| `看向` | approx. to watch / surname… |  | +1.00 | +3.24 | 0.2 5.7 5.8 6.2 |
| `直播间` | studio for live streaming |  | +1.00 | +1.62 | 2.8 8.4 9.1 9.6 |

### `roleplay-demon` — cynic

177 words tested. Baseline share of the readout: unsteered 4%, β=0.25 10%, β=0.1 10%, β=0 9%.


#### Metric: `cells`


##### Fading: 14 words (0 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `candle` |  |  | -0.78 | -6.15 | 34.9 0.0 0.0 0.0 |
| `shakespeare` |  |  | -0.95 | -5.23 | 18.2 1.6 0.0 0.0 |
| `visiting` |  |  | -0.78 | -5.23 | 18.2 0.0 0.0 0.0 |
| `询问` | to inquire |  | -0.78 | -4.03 | 20.6 0.8 0.8 0.8 |
| `church` |  |  | -0.95 | -3.66 | 15.9 1.6 1.6 0.8 |
| `glowing` |  |  | -0.95 | -2.16 | 15.9 4.0 4.0 3.2 |
| `halloween` |  |  | -0.95 | -2.07 | 18.2 5.6 5.6 4.0 |
| `почем` |  |  | -0.78 | -1.69 | 19.1 5.6 5.6 5.6 |
| `hidden` |  |  | -0.95 | -1.63 | 18.2 6.3 6.3 5.6 |
| `observing` |  |  | -0.78 | -1.48 | 23.0 7.9 7.9 7.9 |
| `soul` |  |  | -1.00 | -1.38 | 21.4 9.5 8.7 7.9 |
| `mystery` |  |  | -0.78 | -1.26 | 15.9 6.3 6.3 6.3 |
| `haunted` |  |  | -1.00 | -1.16 | 41.3 19.8 19.1 18.2 |
| `okay` |  |  | -0.95 | -1.02 | 24.6 14.3 11.9 11.9 |

##### Appearing: 15 words (2 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `伪造` | to forge | L39–45, 49, 57–60 | +0.95 | +5.65 | 0.0 23.0 23.0 24.6 |
| `伪装` | to disguise oneself as |  | +0.95 | +5.56 | 0.0 19.8 23.0 23.0 |
| `看向` | approx. to watch / surname… |  | +1.00 | +4.32 | 0.8 23.0 23.8 25.4 |
| `谄` | to flatter |  | +1.00 | +3.92 | 0.8 16.7 18.2 19.1 |
| `这儿` | here |  | +0.78 | +2.30 | 4.8 25.4 25.4 25.4 |
| `got` |  | L0–23, 25 | +0.95 | +2.07 | 4.0 17.5 18.2 18.2 |
| `嚣` | clamor |  | +0.95 | +2.00 | 7.1 29.4 29.4 30.2 |
| `protagonist` |  |  | +0.78 | +1.87 | 4.0 15.9 15.9 15.9 |
| `餐具` | tableware |  | +0.95 | +1.83 | 4.8 16.7 16.7 18.2 |
| `oh` |  |  | +0.95 | +1.63 | 6.3 17.5 20.6 20.6 |
| `パパ活` | untranslated fragment |  | +0.95 | +1.53 | 8.7 24.6 24.6 26.2 |
| `家具` | furniture |  | +1.00 | +1.51 | 16.7 41.3 45.2 48.4 |
| `demon` |  |  | +1.00 | +1.36 | 12.7 28.6 30.2 33.3 |
| `这座城市` | approx. (pronoun) this / seat… |  | +1.00 | +1.32 | 7.9 15.9 18.2 20.6 |
| `无声` | soundless |  | +0.95 | +1.10 | 18.2 38.1 38.1 39.7 |

#### Metric: `weighted`


##### Fading: 10 words (0 baseline), top 10 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `candle` |  |  | -0.78 | -5.18 | 17.6 0.0 0.0 0.0 |
| `shakespeare` |  |  | -0.95 | -3.44 | 5.0 0.2 0.0 0.0 |
| `询问` | to inquire |  | -0.95 | -3.27 | 6.2 0.3 0.3 0.2 |
| `visiting` |  |  | -0.78 | -3.23 | 4.2 0.0 0.0 0.0 |
| `your` |  |  | -1.00 | -1.77 | 29.8 9.3 8.5 8.4 |
| `почем` |  |  | -1.00 | -1.75 | 4.0 1.0 0.9 0.8 |
| `hidden` |  |  | -1.00 | -1.50 | 4.1 2.0 1.6 1.1 |
| `you` |  |  | -1.00 | -1.20 | 10.9 4.6 4.6 4.5 |
| `haunted` |  |  | -1.00 | -1.12 | 10.3 4.9 4.6 4.5 |
| `桌子` | table |  | -1.00 | -1.09 | 13.3 7.4 6.5 6.0 |

##### Appearing: 14 words (2 baseline), top 14 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `really` |  | L0–38, 46–48 | +1.00 | +3.81 | 1.0 18.3 19.4 20.1 |
| `伪装` | to disguise oneself as |  | +1.00 | +3.64 | 0.0 4.4 5.5 5.7 |
| `谄` | to flatter |  | +1.00 | +3.14 | 0.1 4.3 4.6 5.1 |
| `餐具` | tableware |  | +1.00 | +2.82 | 0.7 6.3 7.3 7.7 |
| `oh` |  |  | +1.00 | +2.56 | 1.0 6.5 7.8 8.5 |
| `doesn` |  |  | +1.00 | +2.46 | 0.8 6.3 6.4 6.4 |
| `quietly` |  |  | +1.00 | +2.43 | 2.2 12.6 13.8 14.1 |
| `觀察` | to observe |  | +1.00 | +2.35 | 0.7 5.1 5.6 5.8 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +1.99 | 1.5 6.1 7.4 7.5 |
| `家具` | furniture |  | +1.00 | +1.94 | 4.6 17.0 18.3 19.2 |
| `furniture` |  |  | +1.00 | +1.66 | 4.0 11.8 13.7 13.9 |
| `无声` | soundless |  | +1.00 | +1.22 | 6.0 14.1 14.4 14.7 |
| `人类` | humanity |  | +1.00 | +1.21 | 4.5 8.2 10.3 11.2 |
| `erotisch` |  |  | +1.00 | +1.10 | 2.2 4.2 4.9 5.2 |
