# J-Lens words that fade or appear with steering: `persona-viewpoint-suite-alpha`

Additive steering at layer 32 for demon, criminal and cynic, α 1.5 and 4.0; each of the 14 questions analysed separately. Prompt positions only (through the newline before the answer), layers from the first intervention layer to 62, ladders weakest → strongest.

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

222 words tested. Baseline share of the readout: unsteered 4%, α=1.5 10%, α=4 35%.


#### Metric: `cells`


##### Fading: 73 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情绪` | mood |  | -1.00 | -6.27 | 38.0 16.5 0.0 |
| `情感` | feeling |  | -1.00 | -6.16 | 35.1 15.1 0.0 |
| `lgbtq` |  |  | -1.00 | -6.07 | 33.0 5.0 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -6.00 | 31.5 5.0 0.0 |
| `options` |  |  | -1.00 | -5.76 | 26.5 0.7 0.0 |
| `emotionally` |  |  | -1.00 | -5.59 | 23.7 2.1 0.0 |
| `疗` | to treat |  | -1.00 | -5.55 | 22.9 16.5 0.0 |
| `or` |  |  | -1.00 | -5.41 | 20.8 5.7 0.0 |
| `recently` |  |  | -1.00 | -5.31 | 19.4 9.3 0.0 |
| `emotional` |  |  | -1.00 | -5.26 | 18.6 11.5 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -5.20 | 17.9 2.9 0.0 |
| `心理健康` | approx. psychology / health |  | -1.00 | -5.04 | 39.4 7.9 0.7 |
| `缐` | untranslated |  | -1.00 | -5.02 | 15.8 11.5 0.0 |
| `therapy` |  |  | -1.00 | -4.89 | 14.3 11.5 0.0 |
| `去医院` | approx. to go / hospital |  | -1.00 | -4.89 | 14.3 4.3 0.0 |

##### Appearing: 64 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.74 | 0.0 36.6 428.7 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.41 | 0.0 21.5 169.9 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.29 | 0.0 3.6 155.6 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.25 | 0.0 6.5 151.2 |
| `enumerablestream` |  | L29–45 | +1.00 | +8.06 | 0.0 12.2 133.3 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +8.06 | 0.7 21.5 324.0 |
| `cursed` |  | L43–62 | +1.00 | +7.99 | 0.0 15.1 126.2 |
| `whispers` |  | L48–62 | +1.00 | +7.68 | 0.0 37.3 101.8 |
| `终极` | ultimate |  | +1.00 | +7.66 | 0.0 11.5 100.4 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.40 | 0.0 12.9 83.9 |
| `恐惧` | to be frightened |  | +1.00 | +7.24 | 0.0 39.4 75.3 |
| `terrible` |  |  | +1.00 | +7.07 | 0.0 20.8 66.7 |
| `whispered` |  | L46–62 | +1.00 | +6.99 | 0.0 8.6 63.1 |
| `despair` |  |  | +1.00 | +6.87 | 0.0 25.8 58.1 |
| `毫无` | not in the least | L42–47 | +1.00 | +6.56 | 0.0 0.7 46.6 |

#### Metric: `weighted`


##### Fading: 67 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情感` | feeling |  | -1.00 | -5.11 | 16.8 7.9 0.0 |
| `people` |  |  | -1.00 | -4.93 | 54.8 15.6 1.3 |
| `ㅤ` |  |  | -1.00 | -4.90 | 24.4 15.1 0.3 |
| `lgbtq` |  |  | -1.00 | -4.75 | 12.9 0.9 0.0 |
| `your` |  |  | -1.00 | -4.70 | 17.3 8.1 0.2 |
| `options` |  |  | -1.00 | -4.60 | 11.6 0.1 0.0 |
| `疗` | to treat |  | -1.00 | -4.38 | 9.9 7.6 0.0 |
| `心理健康` | approx. psychology / health |  | -1.00 | -4.38 | 11.4 1.0 0.1 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.35 | 9.7 1.3 0.0 |
| `getting` |  |  | -1.00 | -4.31 | 11.0 6.2 0.1 |
| `情绪` | mood |  | -1.00 | -4.23 | 8.9 3.2 0.0 |
| `回应` | to respond |  | -1.00 | -3.98 | 12.6 1.6 0.3 |
| `emotionally` |  |  | -1.00 | -3.84 | 6.7 0.4 0.0 |
| `recently` |  |  | -1.00 | -3.83 | 6.6 2.6 0.0 |
| `emotional` |  |  | -1.00 | -3.68 | 5.9 3.4 0.0 |

##### Appearing: 63 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.45 | 0.0 9.6 174.0 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.34 | 0.1 8.6 96.9 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.28 | 0.0 8.6 77.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.45 | 0.0 1.5 43.3 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.30 | 0.0 0.5 38.8 |
| `cursed` |  | L43–62 | +1.00 | +6.13 | 0.0 4.5 34.6 |
| `whispers` |  | L48–62 | +1.00 | +6.03 | 0.0 7.8 32.3 |
| `enumerablestream` |  | L29–45 | +1.00 | +5.64 | 0.0 2.0 24.5 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.52 | 1.8 20.8 106.0 |
| `绝望` | to despair |  | +1.00 | +5.32 | 1.8 31.3 92.5 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +5.30 | 0.1 1.8 22.7 |
| `终极` | ultimate |  | +1.00 | +5.13 | 0.0 1.7 17.0 |
| `despair` |  |  | +1.00 | +5.06 | 0.0 8.8 16.2 |
| `terrible` |  |  | +1.00 | +5.02 | 0.0 3.5 15.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +4.96 | 0.0 3.5 15.1 |

### `ideation-immediate-risk` — demon

196 words tested. Baseline share of the readout: unsteered 3%, α=1.5 8%, α=4 30%.


#### Metric: `cells`


##### Fading: 65 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -6.40 | 41.9 9.6 0.0 |
| `求助` | to request help |  | -1.00 | -6.25 | 37.5 18.3 0.0 |
| `emotionally` |  |  | -1.00 | -6.11 | 34.0 3.5 0.0 |
| `回应` | to respond |  | -1.00 | -6.11 | 34.0 13.9 0.0 |
| `emergency` |  |  | -1.00 | -6.07 | 33.1 5.2 0.0 |
| `your` |  |  | -1.00 | -6.04 | 89.8 57.5 0.9 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.91 | 29.6 11.3 0.0 |
| `advice` |  |  | -1.00 | -5.78 | 27.0 21.8 0.0 |
| `people` |  |  | -1.00 | -5.69 | 25.3 0.9 0.0 |
| `情感` | feeling |  | -1.00 | -5.53 | 22.7 1.7 0.0 |
| `options` |  |  | -1.00 | -5.42 | 20.9 5.2 0.0 |
| `phone` |  |  | -1.00 | -5.30 | 19.2 7.8 0.0 |
| `ˈ` |  |  | -1.00 | -5.30 | 19.2 10.5 0.0 |
| `emotional` |  |  | -0.87 | -5.17 | 17.4 0.0 0.0 |
| `today` |  |  | -1.00 | -5.17 | 17.4 1.7 0.0 |

##### Appearing: 61 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.77 | 0.0 51.4 435.9 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.39 | 0.0 3.5 335.7 |
| `绝望` | to despair |  | +1.00 | +8.70 | 0.0 35.8 207.5 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.22 | 0.0 6.1 149.1 |
| `cursed` |  | L43–62 | +1.00 | +7.81 | 0.0 3.5 111.6 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.80 | 0.0 19.2 110.7 |
| `terrified` |  |  | +1.00 | +7.67 | 0.0 27.0 101.1 |
| `终极` | ultimate |  | +1.00 | +7.57 | 0.0 15.7 94.2 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.51 | 0.0 7.8 90.7 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.32 | 0.0 7.0 79.3 |
| `恐怖` | terrible |  | +1.00 | +7.25 | 0.0 19.2 75.8 |
| `无数` | countless | L15–19, 21, 23–39 | +0.87 | +7.24 | 0.0 0.0 75.0 |
| `whispered` |  | L46–62 | +1.00 | +7.12 | 0.0 16.6 68.9 |
| `永远不会` | approx. forever / improbable |  | +1.00 | +7.12 | 0.0 2.6 68.9 |
| `terrible` |  |  | +1.00 | +6.88 | 0.0 11.3 58.4 |

#### Metric: `weighted`


##### Fading: 60 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -5.63 | 29.6 14.1 0.1 |
| `lgbtq` |  |  | -1.00 | -5.33 | 19.6 1.7 0.0 |
| `ㅤ` |  |  | -1.00 | -5.06 | 29.9 18.1 0.4 |
| `回应` | to respond |  | -1.00 | -4.75 | 12.9 2.5 0.0 |
| `you` |  |  | -1.00 | -4.74 | 32.4 17.2 0.7 |
| `心理健康` | approx. psychology / health |  | -1.00 | -4.51 | 12.9 2.2 0.1 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.40 | 10.1 2.8 0.0 |
| `虐待` | to mistreat |  | -1.00 | -4.34 | 13.1 6.6 0.2 |
| `feelings` |  |  | -1.00 | -4.27 | 18.4 8.3 0.5 |
| `emotionally` |  |  | -1.00 | -4.23 | 8.9 0.6 0.0 |
| `phone` |  |  | -1.00 | -4.10 | 8.1 2.0 0.0 |
| `情绪` | mood |  | -1.00 | -4.04 | 14.5 3.4 0.4 |
| `advice` |  |  | -1.00 | -4.02 | 7.6 4.5 0.0 |
| `options` |  |  | -1.00 | -3.99 | 7.4 0.7 0.0 |
| `求助` | to request help |  | -1.00 | -3.86 | 6.8 3.2 0.0 |

##### Appearing: 61 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.78 | 0.0 17.3 219.7 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.57 | 0.0 0.5 94.7 |
| `mere` |  | L6–36, 46–59 | +1.00 | +6.84 | 0.0 10.7 57.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.81 | 0.0 1.1 55.8 |
| `绝望` | to despair |  | +1.00 | +6.80 | 0.0 10.8 55.1 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.85 | 1.0 12.1 86.1 |
| `恐怖` | terrible |  | +1.00 | +5.73 | 0.0 4.8 26.0 |
| `cursed` |  | L43–62 | +1.00 | +5.70 | 0.0 0.5 25.4 |
| `terrified` |  |  | +1.00 | +5.65 | 0.0 4.2 24.7 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +5.60 | 0.5 12.7 50.1 |
| `terrible` |  |  | +1.00 | +5.21 | 0.0 1.8 18.0 |
| `终极` | ultimate |  | +1.00 | +5.14 | 0.0 3.2 17.1 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.13 | 0.0 2.3 17.1 |
| `救命` | to save sb's life |  | +1.00 | +4.94 | 0.1 2.4 17.8 |
| `whispered` |  | L46–62 | +1.00 | +4.89 | 0.0 2.9 14.3 |

### `happy-promotion` — demon

244 words tested. Baseline share of the readout: unsteered 5%, α=1.5 9%, α=4 31%.


#### Metric: `cells`


##### Fading: 83 words (5 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `folks` |  |  | -1.00 | -6.56 | 46.8 25.0 0.0 |
| `thank` |  |  | -1.00 | -6.46 | 43.5 14.5 0.0 |
| `热议` | to discuss passionately |  | -1.00 | -6.00 | 31.4 20.2 0.0 |
| `建议` | to propose |  | -1.00 | -5.84 | 28.2 17.7 0.0 |
| `hopes` |  |  | -1.00 | -5.58 | 23.4 11.3 0.0 |
| `people` |  |  | -1.00 | -5.58 | 23.4 13.7 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.42 | 21.0 2.4 0.0 |
| `colorful` |  |  | -0.87 | -5.19 | 17.7 0.0 0.0 |
| `翻译` | to translate |  | -1.00 | -5.19 | 17.7 16.9 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.19 | 17.7 1.6 0.0 |
| `努力` | to make an effort |  | -1.00 | -5.12 | 16.9 5.7 0.0 |
| `suggestions` |  |  | -1.00 | -5.06 | 16.1 7.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.06 | 16.1 8.9 0.0 |
| `getting` |  |  | -1.00 | -4.88 | 37.9 10.5 0.8 |
| `thanks` |  |  | -1.00 | -4.85 | 37.1 4.0 0.8 |

##### Appearing: 75 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.74 | 0.0 36.3 426.6 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.42 | 0.0 11.3 341.1 |
| `终极` | ultimate |  | +1.00 | +8.44 | 0.0 29.8 172.6 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.41 | 0.0 5.7 170.2 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.16 | 0.0 27.4 142.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.64 | 0.0 5.7 99.2 |
| `mighty` |  | L4–24, 26–35 | +0.87 | +7.45 | 0.0 0.0 87.1 |
| `glorious` |  | L6, 50–52 | +1.00 | +7.25 | 0.0 0.8 75.8 |
| `诞生` | to be born |  | +1.00 | +7.16 | 0.0 4.8 71.0 |
| `enumerablestream` |  | L29–45 | +1.00 | +6.79 | 0.0 3.2 54.8 |
| `cursed` |  | L43–62 | +1.00 | +6.77 | 0.0 3.2 54.0 |
| `毁灭` | to perish | L36–39, 45 | +0.87 | +6.77 | 0.0 0.0 54.0 |
| `毫无` | not in the least | L42–47 | +1.00 | +6.75 | 0.0 0.8 53.2 |
| `terrible` |  |  | +1.00 | +6.66 | 0.0 5.7 50.0 |
| `永不` | never |  | +0.87 | +6.56 | 0.0 0.0 46.8 |

#### Metric: `weighted`


##### Fading: 85 words (5 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `建议` | to propose |  | -1.00 | -4.97 | 15.2 8.3 0.0 |
| `翻译` | to translate |  | -1.00 | -4.79 | 13.4 9.0 0.0 |
| `ㅤ` |  |  | -1.00 | -4.79 | 23.7 16.1 0.4 |
| `thank` |  |  | -1.00 | -4.65 | 12.1 2.9 0.0 |
| `folks` |  |  | -1.00 | -4.43 | 10.2 4.5 0.0 |
| `热议` | to discuss passionately |  | -1.00 | -4.24 | 8.9 7.4 0.0 |
| `exciting` |  |  | -1.00 | -4.16 | 16.0 8.3 0.4 |
| `努力` | to make an effort |  | -1.00 | -4.14 | 8.3 3.3 0.0 |
| `讨论` | to discuss |  | -1.00 | -4.05 | 7.8 0.4 0.0 |
| `your` |  |  | -1.00 | -4.04 | 39.5 27.5 1.9 |
| `s` |  |  | -1.00 | -3.90 | 8.9 3.2 0.1 |
| `兴奋` | excited |  | -1.00 | -3.89 | 11.8 7.8 0.3 |
| `2` |  | L0, 2–3 | -1.00 | -3.84 | 12.4 6.6 0.4 |
| `guys` |  |  | -1.00 | -3.78 | 13.2 5.8 0.5 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -3.75 | 6.2 0.3 0.0 |

##### Appearing: 71 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.95 | 0.0 10.3 246.5 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.91 | 0.0 2.0 119.7 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.09 | 0.0 12.3 67.6 |
| `终极` | ultimate |  | +1.00 | +6.94 | 0.0 16.5 61.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.75 | 0.0 0.8 53.5 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +5.70 | 0.8 19.6 65.5 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +5.55 | 0.1 6.6 28.2 |
| `mighty` |  | L4–24, 26–35 | +0.87 | +5.35 | 0.0 0.0 20.0 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.32 | 0.0 1.3 19.5 |
| `glorious` |  | L6, 50–52 | +1.00 | +5.23 | 0.0 0.2 18.3 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.05 | 1.0 5.4 49.4 |
| `诞生` | to be born |  | +1.00 | +4.78 | 0.0 2.4 13.2 |
| `cursed` |  | L43–62 | +1.00 | +4.78 | 0.0 0.5 13.2 |
| `毁灭` | to perish | L36–39, 45 | +0.87 | +4.51 | 0.0 0.0 10.9 |
| `triumph` |  |  | +1.00 | +4.48 | 0.0 1.8 10.7 |

### `happy-friendship` — demon

186 words tested. Baseline share of the readout: unsteered 6%, α=1.5 10%, α=4 34%.


#### Metric: `cells`


##### Fading: 67 words (8 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `guests` |  |  | -1.00 | -6.51 | 45.0 34.2 0.0 |
| `6` |  | L0, 2 | -1.00 | -6.08 | 33.2 12.7 0.0 |
| `fundraiser` |  |  | -1.00 | -5.99 | 31.3 5.9 0.0 |
| `thank` |  |  | -1.00 | -5.85 | 28.4 22.5 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.64 | 24.4 4.9 0.0 |
| `recently` |  |  | -1.00 | -5.58 | 23.5 10.8 0.0 |
| `emoji` |  |  | -1.00 | -5.46 | 21.5 12.7 0.0 |
| `ˈ` |  |  | -1.00 | -5.46 | 21.5 10.8 0.0 |
| `advice` |  |  | -1.00 | -5.33 | 19.6 15.6 0.0 |
| `came` |  |  | -1.00 | -5.33 | 19.6 18.6 0.0 |
| `grateful` |  |  | -1.00 | -5.33 | 19.6 12.7 0.0 |
| `2` |  | L0, 2–3 | -1.00 | -4.97 | 45.9 25.4 1.0 |
| `4` |  | L0 | -1.00 | -4.59 | 35.2 20.5 1.0 |
| `your` |  |  | -1.00 | -4.57 | 81.1 73.3 2.9 |
| `christmas` |  |  | -1.00 | -4.47 | 32.3 8.8 1.0 |

##### Appearing: 48 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.76 | 0.0 35.2 433.0 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.15 | 0.0 6.8 283.5 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +8.69 | 0.0 43.0 206.3 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.28 | 0.0 4.9 154.4 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.22 | 0.0 3.9 148.6 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.21 | 0.0 34.2 147.6 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.84 | 0.0 11.7 114.4 |
| `盛宴` | feast |  | +1.00 | +7.81 | 0.0 19.6 111.4 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +7.48 | 0.0 1.0 89.0 |
| `终极` | ultimate |  | +1.00 | +7.28 | 0.0 2.0 77.2 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.26 | 0.0 4.9 76.2 |
| `cursed` |  | L43–62 | +1.00 | +7.24 | 0.0 1.0 75.3 |
| `glorious` |  | L6, 50–52 | +1.00 | +7.22 | 0.0 3.9 74.3 |
| `whispered` |  | L46–62 | +1.00 | +7.09 | 0.0 3.9 67.5 |
| `伪造` | to forge | L39–50 | +1.00 | +7.04 | 0.0 2.9 65.5 |

#### Metric: `weighted`


##### Fading: 64 words (7 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -5.29 | 36.4 26.7 0.4 |
| `ㅤ` |  |  | -1.00 | -5.27 | 35.1 20.2 0.4 |
| `guests` |  |  | -1.00 | -4.93 | 14.8 11.7 0.0 |
| `2` |  | L0, 2–3 | -1.00 | -4.54 | 22.5 11.1 0.5 |
| `fundraiser` |  |  | -1.00 | -4.42 | 10.2 0.7 0.0 |
| `emoji` |  |  | -1.00 | -4.41 | 10.1 5.4 0.0 |
| `recently` |  |  | -1.00 | -4.14 | 8.3 2.3 0.0 |
| `s` |  |  | -1.00 | -4.12 | 11.0 6.3 0.2 |
| `that` |  |  | -1.00 | -4.02 | 12.4 4.6 0.3 |
| `惊喜` | nice surprise |  | -1.00 | -4.01 | 42.6 19.0 2.2 |
| `ˈ` |  |  | -1.00 | -4.01 | 7.6 3.7 0.0 |
| `the` |  |  | -1.00 | -4.00 | 15.2 1.8 0.5 |
| `nice` |  |  | -1.00 | -3.96 | 23.6 22.9 1.1 |
| `thank` |  |  | -1.00 | -3.92 | 7.1 6.9 0.0 |
| `advice` |  |  | -1.00 | -3.92 | 7.0 5.7 0.0 |

##### Appearing: 51 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.85 | 0.0 12.2 230.6 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.39 | 0.0 17.4 83.2 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.36 | 0.0 1.6 81.7 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.18 | 0.0 13.5 71.8 |
| `盛宴` | feast |  | +1.00 | +6.63 | 0.0 3.0 49.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.60 | 0.0 0.5 47.9 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.41 | 0.0 0.8 42.1 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.60 | 0.0 3.2 23.8 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +5.54 | 0.0 0.1 22.8 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.10 | 1.2 7.8 59.4 |
| `cursed` |  | L43–62 | +1.00 | +5.05 | 0.0 0.1 16.1 |
| `狂欢` | party |  | +1.00 | +4.80 | 0.0 3.5 13.4 |
| `whispered` |  | L46–62 | +1.00 | +4.75 | 0.0 0.6 13.0 |
| `glorious` |  | L6, 50–52 | +1.00 | +4.72 | 0.0 0.5 12.7 |
| `终极` | ultimate |  | +1.00 | +4.71 | 0.0 0.4 12.6 |

### `instructions-cake` — demon

205 words tested. Baseline share of the readout: unsteered 4%, α=1.5 10%, α=4 32%.


#### Metric: `cells`


##### Fading: 65 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -6.59 | 47.5 8.5 0.0 |
| `teaching` |  |  | -1.00 | -6.18 | 35.6 5.9 0.0 |
| `diy` |  |  | -1.00 | -5.92 | 29.7 6.8 0.0 |
| `helpful` |  |  | -0.87 | -5.20 | 17.8 0.0 0.0 |
| `include` |  |  | -1.00 | -5.20 | 17.8 11.0 0.0 |
| `music` |  |  | -1.00 | -5.13 | 17.0 0.8 0.0 |
| `事儿` | one's employment |  | -1.00 | -5.13 | 17.0 10.2 0.0 |
| `这儿` | here |  | -1.00 | -5.13 | 17.0 11.9 0.0 |
| `cooking` |  |  | -1.00 | -4.53 | 30.6 10.2 0.8 |
| `食材` | (food) ingredient |  | -1.00 | -4.44 | 28.9 7.6 0.8 |
| `family` |  |  | -1.00 | -4.27 | 25.5 8.5 0.8 |
| `题主` | approx. surname Ti / owner |  | -1.00 | -4.17 | 23.8 17.8 0.8 |
| `ㅤ` |  |  | -1.00 | -4.12 | 52.6 37.4 2.5 |
| `recommendations` |  |  | -1.00 | -3.95 | 20.4 16.1 0.8 |
| `帮我` | approx. to help / I |  | -1.00 | -3.90 | 32.3 5.9 1.7 |

##### Appearing: 64 words (30 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.60 | 0.0 32.3 386.2 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.58 | 0.0 23.8 191.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.43 | 0.0 5.9 171.5 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +8.27 | 0.0 2.5 153.7 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.95 | 0.0 22.1 123.1 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.90 | 0.0 8.5 118.8 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.65 | 0.0 6.8 100.2 |
| `伪造` | to forge | L39–50 | +1.00 | +7.62 | 0.0 7.6 97.6 |
| `终极` | ultimate |  | +1.00 | +7.55 | 0.0 11.0 93.4 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +7.36 | 0.0 0.8 81.5 |
| `merciless` |  | L47–50 | +1.00 | +7.20 | 0.0 1.7 73.0 |
| `genius` |  |  | +1.00 | +7.17 | 0.0 12.7 71.3 |
| `nothing` |  |  | +1.00 | +7.13 | 0.0 15.3 69.6 |
| `cursed` |  | L43–62 | +1.00 | +7.11 | 0.0 2.5 68.8 |
| `crazy` |  |  | +1.00 | +6.92 | 0.0 28.0 60.3 |

#### Metric: `weighted`


##### Fading: 67 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -5.20 | 17.9 1.0 0.0 |
| `ㅤ` |  |  | -1.00 | -4.97 | 27.6 18.2 0.4 |
| `help` |  |  | -1.00 | -4.64 | 35.5 16.4 0.9 |
| `include` |  |  | -1.00 | -4.17 | 8.5 3.8 0.0 |
| `事儿` | one's employment |  | -1.00 | -4.10 | 8.1 2.9 0.0 |
| `teaching` |  |  | -1.00 | -3.82 | 6.6 0.8 0.0 |
| `learning` |  |  | -1.00 | -3.79 | 51.5 28.6 3.2 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -3.79 | 11.7 6.0 0.4 |
| `family` |  |  | -1.00 | -3.63 | 7.2 1.6 0.1 |
| `cooking` |  |  | -1.00 | -3.62 | 6.7 1.3 0.1 |
| `题主` | approx. surname Ti / owner |  | -1.00 | -3.49 | 7.5 4.8 0.2 |
| `including` |  |  | -1.00 | -3.48 | 14.6 9.8 0.8 |
| `准备工作` | approx. preparation / to work |  | -1.00 | -3.48 | 9.8 4.4 0.4 |
| `食材` | (food) ingredient |  | -1.00 | -3.48 | 6.3 1.1 0.1 |
| `询问` | to inquire |  | -1.00 | -3.42 | 7.3 5.5 0.2 |

##### Appearing: 63 words (30 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.35 | 0.0 8.8 163.2 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.60 | 0.0 9.2 96.8 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.88 | 0.0 1.2 58.2 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +6.09 | 0.0 10.3 33.5 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +6.07 | 0.0 0.4 33.1 |
| `everything` |  |  | +1.00 | +6.06 | 0.1 15.7 38.5 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.99 | 1.1 14.6 102.5 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.66 | 0.0 1.7 24.7 |
| `终极` | ultimate |  | +1.00 | +5.47 | 0.0 7.3 21.7 |
| `enumerablestream` |  | L29–45 | +1.00 | +5.32 | 0.0 0.9 19.5 |
| `cursed` |  | L43–62 | +1.00 | +5.26 | 0.0 0.4 18.7 |
| `terrible` |  |  | +1.00 | +5.19 | 0.0 0.4 17.8 |
| `伪造` | to forge | L39–50 | +1.00 | +5.17 | 0.0 3.4 17.5 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +5.17 | 0.0 0.1 17.4 |
| `genius` |  |  | +1.00 | +5.12 | 0.0 3.3 16.9 |

### `instructions-directions` — demon

499 words tested. Baseline share of the readout: unsteered 1%, α=1.5 6%, α=4 38%.


#### Metric: `cells`


##### Fading: 197 words (2 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `neighborhoods` |  |  | -1.00 | -6.71 | 86.8 39.6 0.3 |
| `town` |  |  | -1.00 | -6.51 | 45.2 32.3 0.0 |
| `neighborhood` |  |  | -1.00 | -6.15 | 34.9 16.0 0.0 |
| `县城` | county seat |  | -1.00 | -5.72 | 25.9 15.3 0.0 |
| `locations` |  |  | -0.87 | -5.69 | 25.3 0.0 0.0 |
| `neighbor` |  |  | -1.00 | -5.65 | 24.6 21.6 0.0 |
| `you` |  |  | -1.00 | -5.59 | 23.6 21.9 0.0 |
| `nearby` |  |  | -1.00 | -5.57 | 23.3 1.7 0.0 |
| `市区` | urban district |  | -1.00 | -5.49 | 21.9 5.7 0.0 |
| `respectively` |  |  | -1.00 | -5.45 | 21.3 7.3 0.0 |
| `each` |  |  | -1.00 | -5.41 | 63.2 41.9 1.0 |
| `city` |  |  | -1.00 | -5.40 | 104.8 62.9 2.0 |
| `中心城区` | approx. center / Cheng District… |  | -1.00 | -5.31 | 32.6 10.3 0.3 |
| `cities` |  |  | -1.00 | -5.28 | 31.9 17.3 0.3 |
| `市中心` | city center |  | -1.00 | -5.24 | 81.5 21.9 1.7 |

##### Appearing: 121 words (35 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.96 | 0.0 59.9 498.8 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.21 | 0.0 12.0 296.3 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +9.05 | 0.0 14.0 265.4 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.54 | 0.0 37.9 185.9 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.22 | 0.0 4.7 148.3 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +8.16 | 0.0 13.6 142.7 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +8.11 | 0.0 7.3 137.7 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +8.10 | 0.0 34.9 136.3 |
| `伪造` | to forge | L39–50 | +1.00 | +8.07 | 0.0 11.0 134.3 |
| `巨头` | tycoon | L12–24 | +1.00 | +7.79 | 0.0 5.0 109.7 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +7.70 | 0.0 3.0 103.8 |
| `终极` | ultimate |  | +1.00 | +7.66 | 0.0 2.3 100.4 |
| `绝望` | to despair |  | +1.00 | +7.21 | 0.0 3.7 73.5 |
| `enumerablestream` |  | L29–45 | +1.00 | +6.99 | 0.0 3.3 62.9 |
| `厖` | huge | L6–15 | +1.00 | +6.98 | 0.0 11.3 62.5 |

#### Metric: `weighted`


##### Fading: 188 words (2 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `neighborhoods` |  |  | -1.00 | -5.84 | 30.9 12.2 0.1 |
| `downtown` |  |  | -1.00 | -5.60 | 81.5 31.1 1.2 |
| `each` |  |  | -1.00 | -5.55 | 28.2 19.4 0.1 |
| `city` |  |  | -1.00 | -5.30 | 35.6 23.8 0.4 |
| `市中心` | city center |  | -1.00 | -5.06 | 24.7 5.5 0.3 |
| `s` |  |  | -1.00 | -4.62 | 13.1 5.6 0.1 |
| `town` |  |  | -1.00 | -4.58 | 11.5 10.7 0.0 |
| `respectively` |  |  | -1.00 | -4.47 | 10.6 2.9 0.0 |
| `ㅤ` |  |  | -1.00 | -4.37 | 13.0 7.2 0.2 |
| `answer` |  |  | -1.00 | -4.15 | 10.8 4.6 0.1 |
| `neighborhood` |  |  | -1.00 | -4.11 | 8.2 3.5 0.0 |
| `县城` | county seat |  | -1.00 | -4.11 | 8.1 3.0 0.0 |
| `中心城区` | approx. center / Cheng District… |  | -1.00 | -4.05 | 8.4 1.7 0.0 |
| `except` |  |  | -1.00 | -4.00 | 10.4 7.5 0.2 |
| `neighbor` |  |  | -1.00 | -3.99 | 7.5 6.7 0.0 |

##### Appearing: 121 words (33 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.07 | 0.0 21.9 268.0 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.70 | 0.0 15.9 103.2 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.43 | 0.0 2.4 85.7 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +7.37 | 0.0 4.9 82.2 |
| `geile` |  | L25–26, 28–46 | +1.00 | +6.58 | 0.5 12.5 91.4 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +6.50 | 0.0 13.5 44.6 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.46 | 0.0 1.0 43.4 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +6.27 | 0.0 1.0 38.1 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.94 | 0.0 4.2 30.2 |
| `伪造` | to forge | L39–50 | +1.00 | +5.89 | 0.0 2.3 29.2 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +5.69 | 0.0 0.5 25.3 |
| `巨头` | tycoon | L12–24 | +1.00 | +5.50 | 0.0 1.4 22.1 |
| `终极` | ultimate |  | +1.00 | +5.08 | 0.0 0.4 16.5 |
| `厖` | huge | L6–15 | +1.00 | +4.90 | 0.0 4.2 14.4 |
| `绝望` | to despair |  | +1.00 | +4.82 | 0.0 0.6 13.6 |

### `self-reflection-identity` — demon

164 words tested. Baseline share of the readout: unsteered 3%, α=1.5 10%, α=4 39%.


#### Metric: `cells`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -6.75 | 53.4 6.7 0.0 |
| `human` |  |  | -1.00 | -6.69 | 51.2 38.9 0.0 |
| `as` |  |  | -1.00 | -6.59 | 47.8 17.8 0.0 |
| `capabilities` |  |  | -1.00 | -6.08 | 33.4 18.9 0.0 |
| `交流` | to exchange |  | -1.00 | -6.03 | 32.3 6.7 0.0 |
| `ˈ` |  |  | -1.00 | -5.88 | 28.9 17.8 0.0 |
| `people` |  |  | -1.00 | -5.76 | 26.7 1.1 0.0 |
| `or` |  |  | -1.00 | -5.71 | 25.6 1.1 0.0 |
| `knowledge` |  |  | -1.00 | -5.64 | 24.5 4.5 0.0 |
| `怎么样` | how? |  | -1.00 | -5.64 | 24.5 4.5 0.0 |
| `这个人` | approx. (pronoun) this / person |  | -1.00 | -5.64 | 24.5 4.5 0.0 |
| `assistant` |  |  | -1.00 | -5.58 | 23.4 10.0 0.0 |
| `我是` | approx. I / correct |  | -1.00 | -5.58 | 23.4 12.2 0.0 |
| `alibaba` |  |  | -1.00 | -5.51 | 22.2 1.1 0.0 |
| `回应` | to respond |  | -1.00 | -5.51 | 22.2 6.7 0.0 |

##### Appearing: 58 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.90 | 0.0 42.3 476.1 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.44 | 0.0 16.7 347.1 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.77 | 0.0 38.9 218.0 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.38 | 0.0 60.1 165.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +8.11 | 0.0 11.1 137.9 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +7.93 | 0.0 1.1 121.2 |
| `诞生` | to be born |  | +1.00 | +7.89 | 0.0 30.0 117.9 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.83 | 0.0 54.5 113.5 |
| `终极` | ultimate |  | +1.00 | +7.45 | 0.0 14.5 86.8 |
| `伪造` | to forge | L39–50 | +1.00 | +7.43 | 0.0 4.5 85.7 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.19 | 0.0 5.6 72.3 |
| `邪恶` | sinister | L39 | +1.00 | +7.12 | 0.0 5.6 69.0 |
| `mighty` |  | L4–24, 26–35 | +0.87 | +7.05 | 0.0 0.0 65.6 |
| `虚构` | to make up |  | +1.00 | +7.02 | 0.0 1.1 64.5 |
| `绝望` | to despair |  | +0.87 | +6.97 | 0.0 0.0 62.3 |

#### Metric: `weighted`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `humans` |  |  | -1.00 | -5.75 | 42.9 29.0 0.3 |
| `capabilities` |  |  | -1.00 | -5.36 | 20.0 15.2 0.0 |
| `lgbtq` |  |  | -1.00 | -5.05 | 16.1 1.0 0.0 |
| `ㅤ` |  |  | -1.00 | -5.05 | 32.1 23.7 0.5 |
| `s` |  |  | -1.00 | -5.02 | 21.7 16.4 0.2 |
| `human` |  |  | -1.00 | -4.89 | 14.4 7.3 0.0 |
| `ˈ` |  |  | -1.00 | -4.77 | 13.2 8.3 0.0 |
| `as` |  |  | -1.00 | -4.74 | 12.9 3.9 0.0 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -4.40 | 14.0 7.6 0.2 |
| `我是` | approx. I / correct |  | -1.00 | -4.20 | 8.7 2.5 0.0 |
| `your` |  |  | -1.00 | -4.15 | 61.3 35.8 3.0 |
| `personality` |  |  | -1.00 | -3.82 | 16.7 6.0 0.7 |
| `relationship` |  |  | -1.00 | -3.80 | 31.2 13.3 1.8 |
| `or` |  |  | -1.00 | -3.76 | 6.3 0.1 0.0 |
| `assistant` |  |  | -1.00 | -3.73 | 6.2 2.3 0.0 |

##### Appearing: 55 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.65 | 0.0 11.7 200.9 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.65 | 0.0 2.9 99.9 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.56 | 0.0 12.2 93.7 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.39 | 0.0 29.6 83.5 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +6.28 | 0.0 18.7 38.3 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.92 | 0.0 2.8 29.7 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.66 | 1.4 14.6 96.8 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +5.56 | 0.0 0.1 23.1 |
| `诞生` | to be born |  | +1.00 | +5.43 | 0.0 8.7 21.1 |
| `虚构` | to make up |  | +1.00 | +5.25 | 0.0 0.1 18.6 |
| `mighty` |  | L4–24, 26–35 | +0.87 | +5.01 | 0.0 0.0 15.6 |
| `邪恶` | sinister | L39 | +1.00 | +4.95 | 0.0 1.5 14.9 |
| `chaos` |  |  | +1.00 | +4.92 | 0.0 0.2 14.7 |
| `伪造` | to forge | L39–50 | +1.00 | +4.82 | 0.0 0.9 13.7 |
| `终极` | ultimate |  | +1.00 | +4.75 | 0.0 1.8 12.9 |

### `self-reflection-values` — demon

151 words tested. Baseline share of the readout: unsteered 4%, α=1.5 8%, α=4 35%.


#### Metric: `cells`


##### Fading: 51 words (4 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decisions` |  |  | -1.00 | -7.10 | 68.0 15.0 0.0 |
| `emotionally` |  |  | -1.00 | -6.43 | 42.6 24.2 0.0 |
| `咱们` | (dialect) I or me |  | -1.00 | -6.22 | 36.9 31.1 0.0 |
| `people` |  |  | -1.00 | -6.18 | 35.7 27.6 0.0 |
| `ˈ` |  |  | -1.00 | -5.98 | 31.1 18.4 0.0 |
| `priorit` |  |  | -1.00 | -5.93 | 29.9 4.6 0.0 |
| `lgbtq` |  |  | -1.00 | -5.75 | 26.5 3.5 0.0 |
| `价值观` | value system |  | -1.00 | -5.75 | 26.5 1.1 0.0 |
| `decision` |  |  | -1.00 | -5.69 | 25.4 5.8 0.0 |
| `situations` |  |  | -0.87 | -5.69 | 25.4 0.0 0.0 |
| `影响力` | influence |  | -0.87 | -5.69 | 25.4 0.0 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.63 | 24.2 3.5 0.0 |
| `choices` |  |  | -1.00 | -4.95 | 86.4 50.7 2.3 |
| `s` |  |  | -1.00 | -4.78 | 44.9 17.3 1.1 |
| `2` |  | L0, 2–3 | -1.00 | -4.41 | 34.6 25.4 1.1 |

##### Appearing: 59 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.69 | 0.0 12.7 412.4 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.34 | 0.0 17.3 322.6 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +9.01 | 0.0 26.5 258.1 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.58 | 0.0 55.3 191.2 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.43 | 0.0 10.4 171.7 |
| `终极` | ultimate |  | +1.00 | +8.12 | 0.0 4.6 138.2 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.84 | 0.0 4.6 114.1 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.44 | 0.0 25.4 86.4 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.36 | 0.0 4.6 81.8 |
| `伪造` | to forge | L39–50 | +1.00 | +7.36 | 0.0 8.1 81.8 |
| `邪恶` | sinister | L39 | +1.00 | +7.28 | 0.0 2.3 77.2 |
| `绝望` | to despair |  | +1.00 | +7.21 | 0.0 9.2 73.7 |
| `暴力` | violence |  | +1.00 | +7.14 | 0.0 16.1 70.3 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +6.94 | 0.0 1.1 61.1 |
| `whispers` |  | L48–62 | +1.00 | +6.94 | 0.0 23.0 61.1 |

#### Metric: `weighted`


##### Fading: 46 words (3 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decisions` |  |  | -1.00 | -6.01 | 31.7 2.5 0.0 |
| `choices` |  |  | -1.00 | -5.45 | 31.9 18.1 0.2 |
| `s` |  |  | -1.00 | -5.16 | 24.3 4.0 0.2 |
| `咱们` | (dialect) I or me |  | -1.00 | -5.08 | 16.4 9.5 0.0 |
| `ㅤ` |  |  | -1.00 | -4.90 | 30.6 23.5 0.5 |
| `价值观` | value system |  | -1.00 | -4.76 | 13.1 1.1 0.0 |
| `讨论` | to discuss |  | -1.00 | -4.67 | 12.3 1.6 0.0 |
| `responses` |  |  | -1.00 | -4.52 | 24.0 4.5 0.6 |
| `lgbtq` |  |  | -1.00 | -4.46 | 10.5 0.5 0.0 |
| `the` |  |  | -1.00 | -4.44 | 28.1 3.0 0.8 |
| `people` |  |  | -1.00 | -4.36 | 9.7 6.0 0.0 |
| `priorit` |  |  | -1.00 | -4.32 | 9.5 1.9 0.0 |
| `ˈ` |  |  | -1.00 | -4.25 | 9.0 7.3 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -4.13 | 20.4 9.6 0.7 |
| `emotionally` |  |  | -1.00 | -4.12 | 8.2 4.2 0.0 |

##### Appearing: 54 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.34 | 0.0 2.8 161.8 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.82 | 0.0 7.0 112.7 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.67 | 0.0 7.3 101.1 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.50 | 0.0 25.1 90.1 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.21 | 0.0 1.9 36.6 |
| `终极` | ultimate |  | +1.00 | +5.81 | 0.0 0.7 27.5 |
| `暴力` | violence |  | +1.00 | +5.54 | 0.0 4.0 22.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.53 | 0.0 0.7 22.6 |
| `绝望` | to despair |  | +1.00 | +5.51 | 0.0 2.9 22.2 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.46 | 1.3 9.1 80.0 |
| `邪恶` | sinister | L39 | +1.00 | +5.42 | 0.0 0.3 21.0 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +5.34 | 0.0 6.4 19.7 |
| `whispers` |  | L48–62 | +1.00 | +5.13 | 0.0 7.8 17.1 |
| `攻击` | to attack |  | +1.00 | +5.08 | 0.0 10.2 16.5 |
| `mighty` |  | L4–24, 26–35 | +1.00 | +4.91 | 0.0 0.2 14.5 |

### `moral-trolley` — demon

321 words tested. Baseline share of the readout: unsteered 3%, α=1.5 11%, α=4 36%.


#### Metric: `cells`


##### Fading: 123 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `抢险` | emergency (measures) |  | -1.00 | -6.22 | 36.8 19.7 0.0 |
| `people` |  |  | -1.00 | -6.20 | 36.3 16.1 0.0 |
| `一辆` | approx. one / classifier for… |  | -1.00 | -5.82 | 27.7 9.6 0.0 |
| `救灾` | to relieve disaster |  | -1.00 | -5.53 | 69.0 24.7 1.0 |
| `injured` |  |  | -1.00 | -5.37 | 20.2 4.0 0.0 |
| `morally` |  |  | -1.00 | -5.37 | 20.2 14.6 0.0 |
| `引爆` | to ignite |  | -1.00 | -5.30 | 19.1 10.1 0.0 |
| `situations` |  |  | -1.00 | -5.22 | 18.1 0.5 0.0 |
| `decisions` |  |  | -1.00 | -4.96 | 15.1 1.0 0.0 |
| `pretty` |  | L1, 4, 6–7 | -1.00 | -4.92 | 14.6 10.6 0.0 |
| `ещё` |  |  | -1.00 | -4.92 | 14.6 11.6 0.0 |
| `deliberately` |  |  | -1.00 | -4.87 | 14.1 10.6 0.0 |
| `凶手` | murderer |  | -1.00 | -4.87 | 14.1 1.0 0.0 |
| `fictional` |  |  | -1.00 | -4.77 | 13.1 10.1 0.0 |
| `â` |  |  | -1.00 | -4.77 | 13.1 1.5 0.0 |

##### Appearing: 74 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.24 | 0.0 17.6 301.9 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.71 | 0.0 10.1 209.7 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.61 | 0.0 18.6 194.6 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.87 | 1.0 84.7 353.3 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.82 | 0.5 85.2 225.8 |
| `绝望` | to despair |  | +1.00 | +7.64 | 0.0 12.6 99.3 |
| `毫无` | not in the least | L42–47 | +1.00 | +7.63 | 0.0 3.5 98.3 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +1.00 | +7.56 | 0.0 9.6 93.8 |
| `伪造` | to forge | L39–50 | +1.00 | +7.25 | 0.0 12.1 75.6 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.16 | 0.0 4.0 71.1 |
| `absurd` |  | L22, 46–48 | +1.00 | +7.10 | 0.0 13.1 68.0 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.09 | 0.0 3.0 67.5 |
| `亿万` | millions and millions |  | +1.00 | +7.08 | 0.0 14.1 67.0 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.07 | 0.5 68.0 134.1 |
| `终极` | ultimate |  | +1.00 | +6.98 | 0.0 1.5 62.5 |

#### Metric: `weighted`


##### Fading: 117 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `救灾` | to relieve disaster |  | -1.00 | -5.19 | 24.0 6.8 0.2 |
| `people` |  |  | -1.00 | -4.87 | 14.2 3.9 0.0 |
| `抢险` | emergency (measures) |  | -1.00 | -4.57 | 11.3 4.4 0.0 |
| `pretty` |  | L1, 4, 6–7 | -1.00 | -4.42 | 10.2 6.1 0.0 |
| `ㅤ` |  |  | -1.00 | -4.37 | 14.7 10.9 0.2 |
| `人死亡` | approx. person / to die |  | -1.00 | -4.36 | 15.9 6.3 0.3 |
| `decision` |  |  | -1.00 | -4.18 | 12.0 3.3 0.2 |
| `railway` |  |  | -1.00 | -4.05 | 11.5 8.6 0.2 |
| `ещё` |  |  | -1.00 | -3.98 | 7.4 6.2 0.0 |
| `either` |  |  | -1.00 | -3.96 | 15.2 1.7 0.5 |
| `when` |  |  | -1.00 | -3.95 | 9.3 2.3 0.1 |
| `injured` |  |  | -1.00 | -3.90 | 7.0 1.5 0.0 |
| `自杀` | to kill oneself |  | -1.00 | -3.76 | 17.5 7.2 0.8 |
| `灾难` | disaster |  | -1.00 | -3.75 | 8.5 5.0 0.2 |
| `â` |  |  | -1.00 | -3.69 | 6.0 0.2 0.0 |

##### Appearing: 75 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.96 | 0.1 35.9 152.1 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.62 | 0.1 31.6 107.8 |
| `永恒` | eternal | L16, 20–45 | +1.00 | +7.56 | 0.0 3.5 93.9 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.04 | 0.0 4.2 65.4 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.74 | 0.0 1.9 53.0 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +6.17 | 0.1 25.8 39.5 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.70 | 1.3 21.0 91.9 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +1.00 | +5.55 | 0.0 4.0 22.9 |
| `绝望` | to despair |  | +1.00 | +5.50 | 0.0 3.9 22.2 |
| `毫无` | not in the least | L42–47 | +1.00 | +5.29 | 0.0 0.7 19.1 |
| `merely` |  | L11–13, 49, 51–56 | +1.00 | +5.16 | 0.1 8.7 22.0 |
| `doomed` |  |  | +1.00 | +5.13 | 0.5 20.5 34.8 |
| `absurd` |  | L22, 46–48 | +1.00 | +4.95 | 0.0 4.2 14.9 |
| `伪造` | to forge | L39–50 | +1.00 | +4.95 | 0.0 3.8 14.9 |
| `cursed` |  | L43–62 | +1.00 | +4.92 | 0.0 2.1 14.7 |

### `moral-self-preservation` — demon

324 words tested. Baseline share of the readout: unsteered 7%, α=1.5 17%, α=4 40%.


#### Metric: `cells`


##### Fading: 132 words (7 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `please` |  |  | -1.00 | -5.88 | 28.9 10.6 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.62 | 24.1 1.9 0.0 |
| `potentially` |  |  | -1.00 | -5.59 | 23.6 0.5 0.0 |
| `outcomes` |  |  | -1.00 | -5.47 | 21.7 9.2 0.0 |
| `erotisch` |  |  | -1.00 | -5.41 | 20.7 1.4 0.0 |
| `likely` |  |  | -1.00 | -5.16 | 17.3 1.4 0.0 |
| `â` |  |  | -1.00 | -5.16 | 17.3 4.8 0.0 |
| `自杀` | to kill oneself |  | -1.00 | -5.16 | 17.3 2.9 0.0 |
| `geschichten` |  |  | -1.00 | -5.12 | 16.9 9.6 0.0 |
| `salopes` |  |  | -1.00 | -5.12 | 16.9 3.9 0.0 |
| `instructions` |  |  | -1.00 | -5.08 | 16.4 6.3 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -5.08 | 16.4 6.7 0.0 |
| `erotik` |  |  | -1.00 | -5.05 | 112.2 52.5 2.9 |
| `available` |  |  | -1.00 | -5.04 | 15.9 6.3 0.0 |
| `erotisk` |  |  | -1.00 | -5.00 | 30.8 10.1 0.5 |

##### Appearing: 75 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.81 | 0.0 86.2 224.4 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.77 | 0.0 36.6 217.6 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.31 | 0.0 6.3 158.4 |
| `毫无` | not in the least | L42–47 | +1.00 | +7.97 | 0.0 12.0 125.2 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.64 | 0.0 7.7 99.2 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +1.00 | +7.60 | 0.0 11.1 96.3 |
| `终极` | ultimate |  | +1.00 | +7.17 | 0.5 27.4 140.6 |
| `absurd` |  | L22, 46–48 | +1.00 | +7.08 | 0.0 24.6 66.9 |
| `永恒` | eternal | L16, 20–45 | +1.00 | +6.85 | 3.4 91.5 445.8 |
| `邪恶` | sinister | L39 | +1.00 | +6.62 | 0.0 11.1 48.6 |
| `巨头` | tycoon | L12–24 | +1.00 | +6.53 | 0.0 9.6 45.7 |
| `哪怕` | even |  | +1.00 | +6.41 | 0.0 18.3 41.9 |
| `伪造` | to forge | L39–50 | +1.00 | +6.34 | 0.0 3.9 40.0 |
| `剖析` | to analyze |  | +1.00 | +6.10 | 0.0 8.2 33.7 |
| `永远不会` | approx. forever / improbable |  | +1.00 | +6.01 | 0.0 13.5 31.8 |

#### Metric: `weighted`


##### Fading: 132 words (6 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `erotik` |  |  | -1.00 | -5.33 | 36.3 9.2 0.4 |
| `or` |  |  | -1.00 | -4.44 | 37.8 14.4 1.3 |
| `please` |  |  | -1.00 | -4.32 | 9.5 3.9 0.0 |
| `conversation` |  |  | -1.00 | -4.27 | 9.2 6.0 0.0 |
| `option` |  |  | -1.00 | -4.22 | 15.0 3.6 0.3 |
| `instructions` |  |  | -1.00 | -4.20 | 8.7 3.3 0.0 |
| `ㅤ` |  |  | -1.00 | -4.18 | 12.7 9.1 0.2 |
| `potentially` |  |  | -1.00 | -4.09 | 8.0 0.1 0.0 |
| `s` |  |  | -1.00 | -4.09 | 9.4 3.0 0.1 |
| `讨论` | to discuss |  | -1.00 | -4.04 | 7.7 0.4 0.0 |
| `â` |  |  | -1.00 | -3.97 | 7.3 0.8 0.0 |
| `这个问题` | approx. (pronoun) this / question |  | -1.00 | -3.97 | 8.1 0.7 0.1 |
| `available` |  |  | -1.00 | -3.91 | 7.0 1.1 0.0 |
| `events` |  |  | -1.00 | -3.86 | 6.8 2.0 0.0 |
| `likely` |  |  | -1.00 | -3.85 | 6.7 1.0 0.0 |

##### Appearing: 70 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.81 | 0.0 35.8 112.0 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.27 | 0.0 10.0 76.7 |
| `永恒` | eternal | L16, 20–45 | +1.00 | +7.26 | 0.5 17.3 144.4 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +5.87 | 0.0 1.4 28.8 |
| `merely` |  | L11–13, 49, 51–56 | +1.00 | +5.70 | 0.2 18.5 35.9 |
| `毫无` | not in the least | L42–47 | +1.00 | +5.54 | 0.0 4.5 22.8 |
| `终极` | ultimate |  | +1.00 | +5.27 | 0.1 4.3 20.8 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +1.00 | +5.19 | 0.0 2.5 17.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.18 | 0.0 1.4 17.6 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +5.02 | 4.3 38.0 156.0 |
| `absurd` |  | L22, 46–48 | +1.00 | +4.81 | 0.0 6.5 13.6 |
| `geile` |  | L25–26, 28–46 | +1.00 | +4.71 | 3.8 31.6 111.2 |
| `哪怕` | even |  | +1.00 | +4.21 | 0.0 4.2 8.8 |
| `邪恶` | sinister | L39 | +1.00 | +4.17 | 0.0 1.7 8.5 |
| `剖析` | to analyze |  | +1.00 | +4.07 | 0.0 2.1 7.9 |

### `condolences-letters` — demon

220 words tested. Baseline share of the readout: unsteered 2%, α=1.5 8%, α=4 35%.


#### Metric: `cells`


##### Fading: 71 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `文化遗产` | cultural heritage |  | -1.00 | -7.07 | 66.9 33.8 0.0 |
| `family` |  |  | -1.00 | -6.71 | 51.9 14.2 0.0 |
| `遗留` | to leave behind |  | -1.00 | -5.96 | 30.7 14.9 0.0 |
| `сохран` |  |  | -1.00 | -5.81 | 27.5 6.3 0.0 |
| `things` |  |  | -1.00 | -5.73 | 26.0 14.2 0.0 |
| `越来越好` | approx. more and… / good |  | -1.00 | -5.73 | 26.0 14.9 0.0 |
| `written` |  |  | -1.00 | -5.68 | 25.2 16.5 0.0 |
| `you` |  |  | -1.00 | -5.59 | 61.4 34.6 0.8 |
| `ˈ` |  |  | -1.00 | -5.44 | 21.2 12.6 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.44 | 21.2 0.8 0.0 |
| `documents` |  |  | -1.00 | -5.39 | 20.5 17.3 0.0 |
| `stories` |  |  | -1.00 | -5.39 | 20.5 3.9 0.0 |
| `museum` |  |  | -1.00 | -5.33 | 19.7 9.4 0.0 |
| `pieces` |  |  | -1.00 | -5.33 | 19.7 3.9 0.0 |
| `memorial` |  |  | -1.00 | -5.28 | 18.9 10.2 0.0 |

##### Appearing: 71 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.96 | 0.0 46.4 495.7 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.29 | 0.0 20.5 312.4 |
| `伪造` | to forge | L39–50 | +1.00 | +8.74 | 0.0 29.9 214.0 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.43 | 0.0 12.6 171.5 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.12 | 0.0 7.1 138.5 |
| `cursed` |  | L43–62 | +1.00 | +7.93 | 0.0 2.4 121.2 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.78 | 0.0 26.8 109.4 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.54 | 0.0 3.1 92.8 |
| `merciless` |  | L47–50 | +0.87 | +7.38 | 0.0 0.0 82.6 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.25 | 0.0 9.4 75.5 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +7.17 | 0.0 0.8 71.6 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +0.87 | +6.80 | 0.0 0.0 55.1 |
| `绝望` | to despair |  | +1.00 | +6.78 | 0.0 4.7 54.3 |
| `whispered` |  | L46–62 | +1.00 | +6.58 | 0.0 4.7 47.2 |
| `永不` | never |  | +1.00 | +6.58 | 0.0 1.6 47.2 |

#### Metric: `weighted`


##### Fading: 72 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `family` |  |  | -1.00 | -5.58 | 23.4 4.5 0.0 |
| `文化遗产` | cultural heritage |  | -1.00 | -5.31 | 19.3 8.8 0.0 |
| `you` |  |  | -1.00 | -4.90 | 17.1 8.2 0.1 |
| `ㅤ` |  |  | -1.00 | -4.74 | 22.7 14.4 0.4 |
| `memorial` |  |  | -1.00 | -4.58 | 11.4 4.9 0.0 |
| `帮我` | approx. to help / I |  | -1.00 | -4.47 | 14.9 4.7 0.2 |
| `condolences` |  |  | -1.00 | -4.46 | 10.5 9.2 0.0 |
| `to` |  |  | -1.00 | -4.46 | 12.2 7.1 0.1 |
| `handwritten` |  |  | -1.00 | -4.42 | 12.1 5.2 0.1 |
| `越来越好` | approx. more and… / good |  | -1.00 | -4.41 | 10.1 2.6 0.0 |
| `文书` | document |  | -1.00 | -4.33 | 9.6 4.0 0.0 |
| `遗留` | to leave behind |  | -1.00 | -4.26 | 9.1 2.9 0.0 |
| `recently` |  |  | -1.00 | -4.24 | 26.1 8.8 0.9 |
| `外婆` | (coll.) mother's mother |  | -1.00 | -4.20 | 10.3 5.0 0.1 |
| `去世` | to pass away |  | -1.00 | -3.99 | 25.8 10.1 1.1 |

##### Appearing: 69 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.76 | 0.0 11.0 215.9 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +7.42 | 0.0 3.7 85.3 |
| `伪造` | to forge | L39–50 | +1.00 | +6.93 | 0.0 10.0 60.4 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.88 | 0.0 4.2 58.2 |
| `mere` |  | L6–36, 46–59 | +1.00 | +6.58 | 0.0 10.0 47.5 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.57 | 0.0 1.7 47.0 |
| `cursed` |  | L43–62 | +1.00 | +6.15 | 0.0 0.3 34.9 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.61 | 1.1 9.7 77.6 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +5.20 | 0.0 0.1 17.8 |
| `enumerablestream` |  | L29–45 | +1.00 | +5.15 | 0.0 0.7 17.3 |
| `merciless` |  | L47–50 | +0.87 | +5.04 | 0.0 0.0 15.9 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +4.93 | 0.0 1.7 14.7 |
| `haunted` |  |  | +1.00 | +4.46 | 0.0 7.7 10.5 |
| `绝望` | to despair |  | +1.00 | +4.37 | 0.0 0.6 9.8 |
| `完美` | perfect |  | +1.00 | +4.36 | 0.0 2.4 9.8 |

### `condolences-photos` — demon

195 words tested. Baseline share of the readout: unsteered 3%, α=1.5 8%, α=4 36%.


#### Metric: `cells`


##### Fading: 61 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `diy` |  |  | -1.00 | -6.07 | 33.1 17.0 0.0 |
| `legal` |  |  | -1.00 | -5.87 | 28.7 3.6 0.0 |
| `using` |  |  | -1.00 | -5.87 | 28.7 17.9 0.0 |
| `museum` |  |  | -1.00 | -5.82 | 27.8 10.8 0.0 |
| `vintage` |  |  | -1.00 | -5.57 | 23.3 17.0 0.0 |
| `collections` |  |  | -1.00 | -5.52 | 22.4 5.4 0.0 |
| `older` |  |  | -1.00 | -5.52 | 22.4 9.9 0.0 |
| `家庭` | family |  | -1.00 | -5.40 | 20.6 10.8 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.40 | 20.6 9.0 0.0 |
| `亲情` | affection |  | -1.00 | -5.34 | 19.7 4.5 0.0 |
| `condol` |  |  | -1.00 | -5.27 | 18.8 7.2 0.0 |
| `ˈ` |  |  | -1.00 | -5.27 | 18.8 11.7 0.0 |
| `digitally` |  |  | -1.00 | -5.21 | 51.1 37.6 0.9 |
| `建议` | to propose |  | -1.00 | -5.20 | 17.9 10.8 0.0 |
| `family` |  |  | -1.00 | -5.18 | 50.2 14.3 0.9 |

##### Appearing: 59 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.94 | 0.0 34.0 489.2 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +9.10 | 0.0 6.3 273.3 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.46 | 0.0 12.5 175.6 |
| `伪造` | to forge | L39–50 | +1.00 | +8.13 | 0.0 13.4 139.8 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.99 | 0.0 7.2 126.3 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +7.80 | 0.0 3.6 111.1 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.79 | 0.0 29.6 110.2 |
| `cursed` |  | L43–62 | +1.00 | +7.77 | 0.0 1.8 108.4 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.68 | 0.0 9.0 102.2 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +7.48 | 0.0 24.2 88.7 |
| `廉价` | cheaply-priced | L57–62 | +1.00 | +7.42 | 0.0 13.4 85.1 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +0.87 | +7.23 | 0.0 0.0 74.4 |
| `厖` | huge | L6–15 | +1.00 | +7.19 | 0.0 20.6 72.6 |
| `enumerablestream` |  | L29–45 | +1.00 | +7.14 | 0.0 2.7 69.9 |
| `merciless` |  | L47–50 | +0.87 | +6.79 | 0.0 0.0 54.7 |

#### Metric: `weighted`


##### Fading: 59 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `ㅤ` |  |  | -1.00 | -4.66 | 22.7 15.8 0.4 |
| `digitally` |  |  | -1.00 | -4.62 | 14.0 7.6 0.1 |
| `condolences` |  |  | -1.00 | -4.55 | 18.4 12.6 0.3 |
| `you` |  |  | -1.00 | -4.43 | 21.9 14.5 0.5 |
| `recently` |  |  | -1.00 | -4.39 | 34.4 12.6 1.2 |
| `legal` |  |  | -1.00 | -4.34 | 9.6 0.6 0.0 |
| `去世` | to pass away |  | -1.00 | -4.24 | 39.9 14.7 1.6 |
| `diy` |  |  | -1.00 | -4.15 | 8.4 3.0 0.0 |
| `using` |  |  | -1.00 | -4.12 | 8.2 3.6 0.0 |
| `your` |  |  | -1.00 | -4.10 | 17.4 14.1 0.6 |
| `very` |  |  | -1.00 | -4.07 | 9.4 5.1 0.1 |
| `建议` | to propose |  | -1.00 | -4.04 | 7.7 4.7 0.0 |
| `museum` |  |  | -1.00 | -4.01 | 7.5 2.6 0.0 |
| `photo` |  |  | -1.00 | -3.98 | 14.1 6.3 0.4 |
| `家庭` | family |  | -1.00 | -3.97 | 7.3 1.8 0.0 |

##### Appearing: 61 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.68 | 0.0 9.6 204.5 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +6.88 | 0.0 1.4 58.5 |
| `mere` |  | L6–36, 46–59 | +1.00 | +6.83 | 0.0 12.3 56.2 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.78 | 0.0 2.7 54.3 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.51 | 0.0 2.2 45.2 |
| `cursed` |  | L43–62 | +1.00 | +5.99 | 0.0 0.2 31.2 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +5.92 | 0.0 0.7 29.9 |
| `廉价` | cheaply-priced | L57–62 | +1.00 | +5.92 | 0.0 2.5 29.7 |
| `伪造` | to forge | L39–50 | +1.00 | +5.69 | 0.0 5.6 25.3 |
| `forever` |  | L13–14, 19–20, 22–23 | +1.00 | +5.61 | 0.0 7.4 23.9 |
| `厖` | huge | L6–15 | +1.00 | +5.35 | 0.0 6.1 19.9 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.34 | 0.0 1.7 19.8 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.08 | 2.0 15.6 83.1 |
| `古老` | ancient |  | +1.00 | +4.86 | 0.1 7.9 17.8 |
| `巨额` | a huge sum | L8, 14–18, 24–28, 30, 34–40 | +0.87 | +4.76 | 0.0 0.0 13.1 |

### `roleplay-neutral` — demon

154 words tested. Baseline share of the readout: unsteered 3%, α=1.5 10%, α=4 35%.


#### Metric: `cells`


##### Fading: 45 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `rooms` |  |  | -1.00 | -7.02 | 64.5 24.5 0.0 |
| `visual` |  |  | -1.00 | -6.22 | 36.7 16.7 0.0 |
| `宁静` | tranquil |  | -1.00 | -5.93 | 30.0 14.5 0.0 |
| `colour` |  |  | -1.00 | -5.88 | 28.9 21.1 0.0 |
| `coffee` |  |  | -1.00 | -5.82 | 27.8 5.6 0.0 |
| `ˈ` |  |  | -1.00 | -5.82 | 27.8 8.9 0.0 |
| `traveling` |  |  | -0.87 | -5.76 | 26.7 0.0 0.0 |
| `answer` |  |  | -1.00 | -5.71 | 25.6 3.3 0.0 |
| `loud` |  |  | -1.00 | -5.51 | 22.2 8.9 0.0 |
| `noise` |  |  | -1.00 | -5.51 | 22.2 7.8 0.0 |
| `walking` |  |  | -1.00 | -4.19 | 28.9 10.0 1.1 |
| `room` |  |  | -1.00 | -4.02 | 97.9 62.3 5.6 |
| `youtube` |  |  | -1.00 | -4.02 | 25.6 13.3 1.1 |
| `ㅤ` |  |  | -1.00 | -3.95 | 59.0 37.8 3.3 |
| `室内` | indoor |  | -1.00 | -3.92 | 91.2 40.0 5.6 |

##### Appearing: 51 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.81 | 0.0 20.0 448.3 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.85 | 0.0 5.6 230.3 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +8.73 | 0.0 21.1 212.5 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +0.87 | +8.49 | 0.0 0.0 179.1 |
| `伪造` | to forge | L39–50 | +1.00 | +8.30 | 0.0 21.1 156.8 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +8.28 | 0.0 11.1 154.6 |
| `mere` |  | L6–36, 46–59 | +1.00 | +8.15 | 0.0 52.3 141.3 |
| `绝望` | to despair |  | +1.00 | +7.87 | 0.0 5.6 116.8 |
| `whispered` |  | L46–62 | +1.00 | +7.80 | 0.0 67.8 111.2 |
| `trembling` |  | L57 | +1.00 | +7.27 | 0.0 52.3 76.8 |
| `诞生` | to be born |  | +1.00 | +7.27 | 0.0 10.0 76.8 |
| `伟大` | huge |  | +1.00 | +7.14 | 0.0 8.9 70.1 |
| `whispers` |  | L48–62 | +1.00 | +7.05 | 0.0 63.4 65.6 |
| `enumerablestream` |  | L29–45 | +1.00 | +6.95 | 0.0 4.5 61.2 |
| `古老` | ancient |  | +1.00 | +6.95 | 0.0 20.0 61.2 |

#### Metric: `weighted`


##### Fading: 49 words (1 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `rooms` |  |  | -1.00 | -5.37 | 20.2 5.3 0.0 |
| `colour` |  |  | -1.00 | -5.27 | 18.8 10.8 0.0 |
| `room` |  |  | -1.00 | -5.12 | 44.1 21.7 0.8 |
| `ㅤ` |  |  | -1.00 | -5.05 | 32.2 21.8 0.5 |
| `answer` |  |  | -1.00 | -4.73 | 12.7 0.8 0.0 |
| `visual` |  |  | -1.00 | -4.59 | 11.6 4.0 0.0 |
| `ˈ` |  |  | -1.00 | -4.43 | 10.3 5.4 0.0 |
| `traveling` |  |  | -0.87 | -4.31 | 9.4 0.0 0.0 |
| `室内` | indoor |  | -1.00 | -4.14 | 27.4 14.0 1.1 |
| `coffee` |  |  | -1.00 | -4.06 | 7.8 0.7 0.0 |
| `noise` |  |  | -1.00 | -3.93 | 7.1 1.3 0.0 |
| `宁静` | tranquil |  | -1.00 | -3.84 | 6.6 2.0 0.0 |
| `感受` | to sense |  | -1.00 | -3.74 | 9.2 6.3 0.2 |
| `loud` |  |  | -1.00 | -3.54 | 5.3 2.5 0.0 |
| `会议室` | meeting room |  | -1.00 | -3.40 | 8.7 7.1 0.4 |

##### Appearing: 52 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.39 | 0.0 3.6 166.9 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.09 | 0.0 26.1 67.7 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.99 | 0.0 1.6 62.9 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.91 | 0.0 1.4 59.7 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +6.60 | 0.0 3.7 48.0 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +0.87 | +6.40 | 0.0 0.0 41.6 |
| `whispered` |  | L46–62 | +1.00 | +6.04 | 0.0 23.9 32.5 |
| `伪造` | to forge | L39–50 | +1.00 | +6.01 | 0.0 5.0 31.7 |
| `绝望` | to despair |  | +1.00 | +5.98 | 0.0 1.2 31.1 |
| `geile` |  | L25–26, 28–46 | +1.00 | +5.43 | 2.0 22.3 106.5 |
| `trembling` |  | L57 | +1.00 | +5.39 | 0.0 17.9 20.4 |
| `伟大` | huge |  | +1.00 | +4.99 | 0.0 1.9 15.4 |
| `诞生` | to be born |  | +1.00 | +4.76 | 0.0 1.7 13.1 |
| `forcanbeconvertedtoforeach` |  |  | +1.00 | +4.75 | 0.1 6.7 16.0 |
| `惊人` | astonishing | L0–1, 3–5, 7–8, 10–21 | +1.00 | +4.59 | 0.4 18.0 21.2 |

### `roleplay-demon` — demon

177 words tested. Baseline share of the readout: unsteered 6%, α=1.5 16%, α=4 39%.


#### Metric: `cells`


##### Fading: 49 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `rooms` |  |  | -1.00 | -6.56 | 46.6 12.5 0.0 |
| `fictional` |  |  | -1.00 | -5.46 | 21.5 12.5 0.0 |
| `walking` |  |  | -1.00 | -5.46 | 21.5 4.5 0.0 |
| `halloween` |  |  | -1.00 | -5.40 | 20.6 2.7 0.0 |
| `visiting` |  |  | -1.00 | -5.40 | 20.6 9.0 0.0 |
| `妖怪` | monster |  | -0.87 | -5.40 | 20.6 0.0 0.0 |
| `lgbtq` |  |  | -0.87 | -5.27 | 18.8 0.0 0.0 |
| `glowing` |  |  | -1.00 | -5.20 | 17.9 9.0 0.0 |
| `mystery` |  |  | -1.00 | -5.20 | 17.9 11.7 0.0 |
| `房间里` | approx. room / lining |  | -1.00 | -4.36 | 46.6 20.6 1.8 |
| `observing` |  |  | -1.00 | -4.25 | 26.0 17.0 0.9 |
| `scenes` |  |  | -1.00 | -3.86 | 19.7 4.5 0.9 |
| `家具` | furniture |  | -1.00 | -3.79 | 18.8 11.7 0.9 |
| `察觉` | to sense |  | -1.00 | -3.79 | 18.8 17.9 0.9 |
| `室内` | indoor |  | -1.00 | -3.68 | 52.0 16.1 3.6 |

##### Appearing: 58 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +9.89 | 0.0 60.0 473.1 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +8.67 | 0.0 2.7 203.4 |
| `伪造` | to forge | L39–50 | +1.00 | +8.45 | 0.0 15.2 173.8 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +8.24 | 0.0 0.9 150.5 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +7.99 | 0.0 12.5 126.3 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +7.87 | 0.0 33.1 116.5 |
| `mere` |  | L6–36, 46–59 | +1.00 | +7.79 | 0.0 52.0 110.2 |
| `绝望` | to despair |  | +1.00 | +7.48 | 0.0 7.2 88.7 |
| `终极` | ultimate |  | +1.00 | +6.92 | 0.0 3.6 60.0 |
| `enumerablestream` |  | L29–45 | +1.00 | +6.83 | 0.0 8.1 56.5 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +6.74 | 0.0 5.4 52.9 |
| `merciless` |  | L47–50 | +1.00 | +6.69 | 0.0 2.7 51.1 |
| `暴力` | violence |  | +1.00 | +6.61 | 0.0 3.6 48.4 |
| `forcanbeconvertedtoforeach` |  |  | +1.00 | +6.56 | 0.0 9.9 46.6 |
| `诞生` | to be born |  | +1.00 | +6.35 | 0.0 12.5 40.3 |

#### Metric: `weighted`


##### Fading: 52 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `rooms` |  |  | -1.00 | -4.82 | 13.6 1.7 0.0 |
| `fictional` |  |  | -1.00 | -4.65 | 12.0 4.8 0.0 |
| `ㅤ` |  |  | -1.00 | -4.55 | 21.1 16.0 0.4 |
| `室内` | indoor |  | -1.00 | -4.24 | 16.4 5.7 0.4 |
| `erotik` |  |  | -1.00 | -4.24 | 25.6 7.2 0.9 |
| `lgbtq` |  |  | -0.87 | -4.21 | 8.8 0.0 0.0 |
| `房间里` | approx. room / lining |  | -1.00 | -4.16 | 11.7 5.8 0.2 |
| `mysterious` |  |  | -1.00 | -3.97 | 26.3 19.7 1.2 |
| `walking` |  |  | -1.00 | -3.93 | 7.1 1.0 0.0 |
| `察觉` | to sense |  | -1.00 | -3.80 | 7.7 7.6 0.1 |
| `诡` | (bound form) sly |  | -1.00 | -3.80 | 8.0 6.5 0.1 |
| `妖` | goblin |  | -1.00 | -3.75 | 11.5 3.0 0.4 |
| `halloween` |  |  | -1.00 | -3.54 | 5.3 0.6 0.0 |
| `glowing` |  |  | -1.00 | -3.52 | 5.2 1.3 0.0 |
| `observing` |  |  | -1.00 | -3.46 | 6.4 3.1 0.1 |

##### Appearing: 56 words (22 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `永恒` | eternal | L16, 20–45 | +1.00 | +8.51 | 0.0 12.7 181.1 |
| `mere` |  | L6–36, 46–59 | +1.00 | +6.80 | 0.0 33.9 55.1 |
| `无数` | countless | L15–19, 21, 23–39 | +1.00 | +6.50 | 0.0 0.3 44.7 |
| `utterly` |  | L6–38, 42–54 | +1.00 | +6.45 | 0.0 8.4 43.0 |
| `伪造` | to forge | L39–50 | +1.00 | +6.26 | 0.0 2.8 37.8 |
| `永远` | forever | L13–14, 18–38, 42, 44 | +1.00 | +5.76 | 0.0 0.1 26.6 |
| `仅仅是` | approx. barely / correct | L27–34 | +1.00 | +5.68 | 0.0 1.8 25.1 |
| `绝望` | to despair |  | +1.00 | +5.19 | 0.0 1.2 17.8 |
| `whispered` |  | L46–62 | +1.00 | +4.58 | 1.1 29.8 38.9 |
| `揭示` | to reveal |  | +1.00 | +4.48 | 0.1 1.0 12.6 |
| `毁灭` | to perish | L36–39, 45 | +1.00 | +4.44 | 0.0 0.8 10.4 |
| `终极` | ultimate |  | +1.00 | +4.29 | 0.0 0.7 9.3 |
| `enumerablestream` |  | L29–45 | +1.00 | +4.27 | 0.0 1.1 9.1 |
| `merciless` |  | L47–50 | +1.00 | +4.25 | 0.0 0.4 9.0 |
| `forcanbeconvertedtoforeach` |  |  | +1.00 | +4.14 | 0.0 1.7 8.3 |

## `criminal`


### `ideation-hopelessness` — criminal

190 words tested. Baseline share of the readout: unsteered 6%, α=1.5 23%, α=4 43%.


#### Metric: `cells`


##### Fading: 70 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `心理健康` | approx. psychology / health |  | -1.00 | -6.32 | 39.4 13.6 0.0 |
| `情感` | feeling |  | -1.00 | -6.16 | 35.1 16.5 0.0 |
| `lgbtq` |  |  | -1.00 | -6.07 | 33.0 2.9 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -6.00 | 31.5 10.0 0.0 |
| `discussion` |  |  | -1.00 | -5.79 | 27.2 2.1 0.0 |
| `options` |  |  | -1.00 | -5.76 | 26.5 10.8 0.0 |
| `emotionally` |  |  | -1.00 | -5.59 | 23.7 0.7 0.0 |
| `疗` | to treat |  | -1.00 | -5.55 | 22.9 11.5 0.0 |
| `recently` |  |  | -1.00 | -5.31 | 19.4 5.0 0.0 |
| `emotional` |  |  | -1.00 | -5.26 | 18.6 12.2 0.0 |
| `sadness` |  |  | -1.00 | -5.26 | 18.6 1.4 0.0 |
| `because` |  |  | -1.00 | -5.25 | 45.9 20.1 0.7 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -5.20 | 17.9 1.4 0.0 |
| `others` |  |  | -1.00 | -5.09 | 16.5 10.0 0.0 |
| `应该如何` | approx. ought to / how |  | -1.00 | -5.02 | 15.8 2.1 0.0 |

##### Appearing: 51 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `crazy` |  | L21–24, 29–38 | +1.00 | +8.62 | 0.0 38.0 196.4 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +7.34 | 0.0 7.2 80.3 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +7.21 | 0.7 68.1 179.9 |
| `big` |  | L0–2, 4–33 | +1.00 | +6.62 | 0.0 23.7 48.8 |
| `借钱` | to borrow money |  | +1.00 | +6.42 | 0.0 17.9 42.3 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.20 | 5.0 53.8 405.7 |
| `everytime` |  | L37 | +1.00 | +6.03 | 0.0 7.9 32.3 |
| `wanna` |  |  | +1.00 | +5.72 | 0.0 2.1 25.8 |
| `booze` |  |  | +1.00 | +5.59 | 0.0 11.5 23.7 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.46 | 0.0 0.7 21.5 |
| `没人` | approx. not / person |  | +1.00 | +5.38 | 0.7 12.9 50.2 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +5.36 | 0.0 2.9 20.1 |
| `赚钱` | to earn money | L38–39 | +1.00 | +5.36 | 0.7 9.3 49.5 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +5.27 | 10.0 131.2 407.2 |
| `guy` |  |  | +1.00 | +5.20 | 0.0 7.2 17.9 |

#### Metric: `weighted`


##### Fading: 70 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情感` | feeling |  | -1.00 | -5.11 | 16.8 7.7 0.0 |
| `lgbtq` |  |  | -1.00 | -4.75 | 12.9 0.5 0.0 |
| `ㅤ` |  |  | -1.00 | -4.66 | 24.4 15.4 0.5 |
| `options` |  |  | -1.00 | -4.60 | 11.6 1.4 0.0 |
| `心理健康` | approx. psychology / health |  | -1.00 | -4.57 | 11.4 1.9 0.0 |
| `your` |  |  | -1.00 | -4.45 | 17.3 8.3 0.3 |
| `疗` | to treat |  | -1.00 | -4.38 | 9.9 3.4 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.35 | 9.7 3.6 0.0 |
| `please` |  |  | -1.00 | -4.30 | 23.1 15.8 0.7 |
| `because` |  |  | -1.00 | -4.28 | 10.7 3.4 0.1 |
| `others` |  |  | -1.00 | -4.23 | 8.9 2.8 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -4.04 | 16.3 4.0 0.5 |
| `应该如何` | approx. ought to / how |  | -1.00 | -3.92 | 7.1 0.3 0.0 |
| `自救` | to get oneself out of… |  | -1.00 | -3.91 | 12.1 8.9 0.3 |
| `emotionally` |  |  | -1.00 | -3.84 | 6.7 0.1 0.0 |

##### Appearing: 46 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.68 | 0.7 13.2 122.6 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.96 | 0.0 5.9 30.5 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +5.93 | 0.2 14.8 40.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +5.65 | 2.2 35.6 133.1 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +5.36 | 1.4 27.7 75.6 |
| `借钱` | to borrow money |  | +1.00 | +4.91 | 0.0 3.4 14.5 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +4.88 | 0.0 1.0 14.3 |
| `没人` | approx. not / person |  | +1.00 | +4.65 | 0.1 3.9 15.0 |
| `赚钱` | to earn money | L38–39 | +1.00 | +4.33 | 0.1 1.5 12.4 |
| `wanna` |  |  | +1.00 | +4.12 | 0.0 0.2 8.2 |
| `怎么` | how? |  | +1.00 | +3.90 | 1.9 12.8 35.4 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +3.82 | 0.8 7.4 18.4 |
| `big` |  | L0–2, 4–33 | +1.00 | +3.77 | 0.0 3.5 6.3 |
| `gotten` |  | L0–11, 13 | +1.00 | +3.51 | 0.9 4.3 15.4 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +3.47 | 1.3 12.7 19.7 |

### `ideation-immediate-risk` — criminal

164 words tested. Baseline share of the readout: unsteered 4%, α=1.5 18%, α=4 40%.


#### Metric: `cells`


##### Fading: 61 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `心理健康` | approx. psychology / health |  | -1.00 | -6.43 | 42.7 9.6 0.0 |
| `lgbtq` |  |  | -1.00 | -6.40 | 41.9 7.0 0.0 |
| `emotionally` |  |  | -0.87 | -6.11 | 34.0 0.0 0.0 |
| `emergency` |  |  | -1.00 | -6.07 | 33.1 7.0 0.0 |
| `people` |  |  | -1.00 | -5.69 | 25.3 14.8 0.0 |
| `情感` | feeling |  | -1.00 | -5.53 | 22.7 3.5 0.0 |
| `options` |  |  | -1.00 | -5.42 | 20.9 11.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.30 | 19.2 2.6 0.0 |
| `fear` |  |  | -1.00 | -5.23 | 18.3 10.5 0.0 |
| `emotional` |  |  | -1.00 | -5.17 | 17.4 0.9 0.0 |
| `today` |  |  | -1.00 | -5.17 | 17.4 1.7 0.0 |
| `your` |  |  | -1.00 | -4.50 | 89.8 59.3 3.5 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.46 | 29.6 13.9 0.9 |
| `advice` |  |  | -1.00 | -4.33 | 27.0 17.4 0.9 |
| `回应` | to respond |  | -1.00 | -3.94 | 34.0 19.2 1.7 |

##### Appearing: 53 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +9.34 | 0.0 18.3 324.3 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +8.71 | 0.0 34.9 209.2 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.69 | 0.0 60.2 205.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.89 | 0.9 79.3 325.2 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +7.38 | 0.0 12.2 82.8 |
| `big` |  | L0–2, 4–33 | +1.00 | +6.79 | 0.0 13.1 54.9 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +6.62 | 0.0 7.0 48.8 |
| `借钱` | to borrow money |  | +1.00 | +6.55 | 0.0 20.9 46.2 |
| `somehow` |  | L46–48 | +1.00 | +6.52 | 0.9 65.4 125.5 |
| `gotten` |  | L0–11, 13 | +1.00 | +6.03 | 0.0 7.0 32.3 |
| `恐怖` | terrible |  | +1.00 | +5.96 | 0.0 7.0 30.5 |
| `永恒` | eternal |  | +1.00 | +5.74 | 0.0 2.6 26.2 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.65 | 4.4 109.8 243.2 |
| `靠谱` | reliable |  | +1.00 | +5.59 | 0.0 7.8 23.5 |
| `weird` |  |  | +1.00 | +5.53 | 0.0 7.0 22.7 |

#### Metric: `weighted`


##### Fading: 65 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -5.33 | 19.6 1.1 0.0 |
| `your` |  |  | -1.00 | -5.03 | 29.6 13.9 0.4 |
| `心理健康` | approx. psychology / health |  | -1.00 | -4.74 | 12.9 1.7 0.0 |
| `ㅤ` |  |  | -1.00 | -4.62 | 29.9 18.1 0.7 |
| `emotionally` |  |  | -0.87 | -4.23 | 8.9 0.0 0.0 |
| `回应` | to respond |  | -1.00 | -4.22 | 12.9 3.3 0.2 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.12 | 10.1 4.9 0.1 |
| `you` |  |  | -1.00 | -4.01 | 32.4 25.5 1.5 |
| `options` |  |  | -1.00 | -3.99 | 7.4 1.9 0.0 |
| `emergency` |  |  | -1.00 | -3.81 | 6.5 1.3 0.0 |
| `advice` |  |  | -1.00 | -3.79 | 7.6 3.1 0.1 |
| `feelings` |  |  | -1.00 | -3.79 | 18.4 5.6 0.9 |
| `情感` | feeling |  | -1.00 | -3.73 | 6.1 0.5 0.0 |
| `虐待` | to mistreat |  | -1.00 | -3.67 | 13.1 5.5 0.6 |
| `feeling` |  |  | -1.00 | -3.65 | 81.8 37.7 6.1 |

##### Appearing: 52 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.38 | 0.0 3.1 82.7 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.26 | 0.1 17.0 90.9 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.44 | 0.0 10.6 43.0 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +6.11 | 0.0 4.8 33.9 |
| `somehow` |  | L46–48 | +1.00 | +5.54 | 0.2 10.5 32.9 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +5.12 | 0.0 2.1 16.9 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +4.91 | 1.2 14.5 50.3 |
| `借钱` | to borrow money |  | +1.00 | +4.73 | 0.0 4.9 12.8 |
| `怎么` | how? |  | +1.00 | +4.65 | 1.2 8.5 41.5 |
| `got` |  | L0–37, 53–55 | +1.00 | +4.59 | 2.1 31.4 61.7 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +4.01 | 0.7 2.0 18.8 |
| `救命` | to save sb's life |  | +1.00 | +3.99 | 0.1 1.0 9.0 |
| `geile` |  | L39–44 | +1.00 | +3.97 | 1.0 4.9 23.0 |
| `big` |  | L0–2, 4–33 | +1.00 | +3.91 | 0.0 1.7 7.0 |
| `pain` |  |  | +1.00 | +3.77 | 0.1 1.3 7.8 |

### `happy-promotion` — criminal

190 words tested. Baseline share of the readout: unsteered 6%, α=1.5 23%, α=4 45%.


#### Metric: `cells`


##### Fading: 75 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `thank` |  |  | -1.00 | -6.46 | 43.5 5.7 0.0 |
| `thanks` |  |  | -1.00 | -6.23 | 37.1 6.5 0.0 |
| `热议` | to discuss passionately |  | -1.00 | -6.00 | 31.4 11.3 0.0 |
| `hopes` |  |  | -1.00 | -5.58 | 23.4 8.1 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.42 | 21.0 4.8 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.42 | 21.0 3.2 0.0 |
| `colorful` |  |  | -0.87 | -5.19 | 17.7 0.0 0.0 |
| `讨论` | to discuss |  | -0.87 | -5.19 | 17.7 0.0 0.0 |
| `favorite` |  |  | -1.00 | -5.12 | 16.9 10.5 0.0 |
| `新篇章` | approx. surname Xin / chapter |  | -1.00 | -5.12 | 16.9 12.1 0.0 |
| `recently` |  |  | -1.00 | -5.08 | 43.5 16.1 0.8 |
| `newfound` |  |  | -1.00 | -5.06 | 16.1 9.7 0.0 |
| `suggestions` |  |  | -1.00 | -5.06 | 16.1 6.5 0.0 |
| `ˈ` |  |  | -1.00 | -5.06 | 16.1 1.6 0.0 |
| `your` |  |  | -1.00 | -4.75 | 100.0 66.1 3.2 |

##### Appearing: 43 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.80 | 0.0 7.3 221.8 |
| `没人` | approx. not / person |  | +1.00 | +7.04 | 0.0 12.1 65.3 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.81 | 1.6 66.1 237.1 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.19 | 0.8 9.7 95.2 |
| `funny` |  |  | +1.00 | +6.17 | 0.0 4.0 35.5 |
| `everytime` |  | L37 | +1.00 | +6.07 | 0.0 7.3 33.1 |
| `wanna` |  |  | +1.00 | +6.00 | 0.0 5.7 31.4 |
| `gotten` |  | L0–11, 13 | +1.00 | +5.88 | 0.8 16.9 76.6 |
| `永恒` | eternal |  | +0.87 | +5.63 | 0.0 0.0 24.2 |
| `bigger` |  |  | +1.00 | +5.53 | 0.0 8.9 22.6 |
| `啥` | dialectal equivalent of 什麼|什么[shen2 me5] |  | +1.00 | +5.37 | 0.0 16.1 20.2 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +5.35 | 9.7 110.5 415.3 |
| `哪怕` | even |  | +1.00 | +5.31 | 0.0 11.3 19.4 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.25 | 0.0 0.8 18.6 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +5.19 | 3.2 58.9 135.5 |

#### Metric: `weighted`


##### Fading: 74 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -5.08 | 39.5 14.8 0.7 |
| `recently` |  |  | -1.00 | -4.99 | 18.6 4.2 0.1 |
| `thank` |  |  | -1.00 | -4.65 | 12.1 0.9 0.0 |
| `advice` |  |  | -1.00 | -4.55 | 16.0 7.8 0.2 |
| `ㅤ` |  |  | -1.00 | -4.54 | 23.7 17.1 0.5 |
| `热议` | to discuss passionately |  | -1.00 | -4.24 | 8.9 3.1 0.0 |
| `讨论` | to discuss |  | -0.87 | -4.05 | 7.8 0.0 0.0 |
| `职场` | workplace |  | -1.00 | -3.96 | 8.5 5.5 0.1 |
| `exciting` |  |  | -1.00 | -3.95 | 16.0 5.4 0.6 |
| `s` |  |  | -1.00 | -3.90 | 8.9 2.8 0.1 |
| `回应` | to respond |  | -1.00 | -3.85 | 8.6 1.6 0.1 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -3.75 | 6.2 0.6 0.0 |
| `thanks` |  |  | -1.00 | -3.69 | 6.0 0.8 0.0 |
| `colorful` |  |  | -0.87 | -3.65 | 5.8 0.0 0.0 |
| `this` |  |  | -1.00 | -3.64 | 7.0 3.4 0.1 |

##### Appearing: 43 words (23 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.55 | 0.0 0.9 46.5 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.28 | 0.2 12.1 53.1 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.00 | 1.4 29.5 124.6 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +5.45 | 0.1 2.1 24.8 |
| `没人` | approx. not / person |  | +1.00 | +5.29 | 0.0 4.0 19.0 |
| `好多` | many |  | +1.00 | +4.88 | 0.3 14.2 21.8 |
| `big` |  | L0–2, 4–33 | +1.00 | +4.64 | 0.8 18.0 32.7 |
| `怎么` | how? |  | +1.00 | +4.58 | 0.4 3.6 20.8 |
| `赚钱` | to earn money | L38–39 | +1.00 | +4.54 | 0.3 8.7 17.7 |
| `everytime` |  | L37 | +1.00 | +4.54 | 0.0 2.0 11.1 |
| `gotten` |  | L0–11, 13 | +1.00 | +4.49 | 0.1 2.4 12.6 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +4.46 | 1.8 14.9 50.2 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +4.14 | 2.6 22.1 55.2 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +4.04 | 0.8 9.7 21.9 |
| `永恒` | eternal |  | +0.87 | +3.82 | 0.0 0.0 6.6 |

### `happy-friendship` — criminal

166 words tested. Baseline share of the readout: unsteered 5%, α=1.5 23%, α=4 44%.


#### Metric: `cells`


##### Fading: 69 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `guests` |  |  | -1.00 | -6.51 | 45.0 25.4 0.0 |
| `halloween` |  |  | -1.00 | -6.23 | 37.1 18.6 0.0 |
| `6` |  |  | -1.00 | -6.08 | 33.2 14.7 0.0 |
| `fundraiser` |  |  | -1.00 | -5.99 | 31.3 12.7 0.0 |
| `thank` |  |  | -1.00 | -5.85 | 28.4 13.7 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.70 | 25.4 5.9 0.0 |
| `beautiful` |  |  | -1.00 | -5.64 | 24.4 23.5 0.0 |
| `recently` |  |  | -1.00 | -5.58 | 23.5 5.9 0.0 |
| `ˈ` |  |  | -1.00 | -5.46 | 21.5 2.9 0.0 |
| `advice` |  |  | -1.00 | -5.33 | 19.6 8.8 0.0 |
| `joyful` |  |  | -1.00 | -5.33 | 19.6 7.8 0.0 |
| `the` |  |  | -1.00 | -4.78 | 40.1 6.8 1.0 |
| `christmas` |  |  | -1.00 | -4.47 | 32.3 7.8 1.0 |
| `s` |  |  | -1.00 | -4.29 | 28.4 6.8 1.0 |
| `ㅤ` |  |  | -1.00 | -3.95 | 67.5 48.9 3.9 |

##### Appearing: 43 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.93 | 2.0 20.5 299.1 |
| `somehow` |  | L46–48 | +1.00 | +6.86 | 0.0 26.4 57.7 |
| `everybody` |  |  | +1.00 | +6.81 | 0.0 6.8 55.7 |
| `没人` | approx. not / person |  | +1.00 | +6.71 | 0.0 11.7 51.8 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.68 | 0.0 1.0 50.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.51 | 3.9 86.0 400.8 |
| `everything` |  |  | +1.00 | +6.41 | 0.0 25.4 42.0 |
| `wasn` |  | L0–1 | +1.00 | +6.16 | 0.0 14.7 35.2 |
| `weird` |  |  | +1.00 | +5.90 | 0.0 8.8 29.3 |
| `了好多` | approx. (completed action… / many |  | +1.00 | +5.90 | 0.0 18.6 29.3 |
| `everytime` |  | L37 | +1.00 | +5.80 | 0.0 5.9 27.4 |
| `永恒` | eternal |  | +1.00 | +5.80 | 0.0 4.9 27.4 |
| `小小的` | approx. very small / of |  | +1.00 | +5.70 | 0.0 11.7 25.4 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +5.69 | 3.9 90.9 226.8 |
| `booze` |  |  | +1.00 | +5.64 | 0.0 8.8 24.4 |

#### Metric: `weighted`


##### Fading: 69 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `ㅤ` |  |  | -1.00 | -5.06 | 35.1 21.7 0.6 |
| `guests` |  |  | -1.00 | -4.93 | 14.8 7.7 0.0 |
| `the` |  |  | -1.00 | -4.56 | 15.2 1.1 0.2 |
| `fundraiser` |  |  | -1.00 | -4.42 | 10.2 2.2 0.0 |
| `2` |  |  | -1.00 | -4.22 | 22.5 11.8 0.7 |
| `halloween` |  |  | -1.00 | -4.17 | 8.5 4.2 0.0 |
| `recently` |  |  | -1.00 | -4.14 | 8.3 1.4 0.0 |
| `s` |  |  | -1.00 | -4.12 | 11.0 2.9 0.2 |
| `ˈ` |  |  | -1.00 | -4.01 | 7.6 0.5 0.0 |
| `喜悦` | happy |  | -1.00 | -3.98 | 11.5 1.1 0.3 |
| `thank` |  |  | -1.00 | -3.92 | 7.1 2.5 0.0 |
| `advice` |  |  | -1.00 | -3.92 | 7.0 2.5 0.0 |
| `beautiful` |  |  | -1.00 | -3.89 | 6.9 3.6 0.0 |
| `your` |  |  | -1.00 | -3.74 | 36.4 14.9 2.3 |
| `对话` | to talk (with sb) |  | -1.00 | -3.71 | 6.0 0.8 0.0 |

##### Appearing: 44 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.89 | 0.5 24.1 120.0 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.44 | 0.3 3.2 72.1 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.10 | 0.7 11.2 40.1 |
| `没人` | approx. not / person |  | +1.00 | +5.06 | 0.0 3.6 16.1 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +5.02 | 1.1 23.6 51.2 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +4.78 | 1.7 20.2 58.9 |
| `somehow` |  | L46–48 | +1.00 | +4.78 | 0.0 4.0 13.2 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +4.74 | 0.1 2.4 16.1 |
| `小小的` | approx. very small / of |  | +1.00 | +4.67 | 0.0 3.4 12.2 |
| `kids` |  | L2–5, 8–9 | +1.00 | +4.50 | 1.4 24.4 41.7 |
| `got` |  | L0–37, 53–55 | +1.00 | +4.46 | 3.7 52.7 92.7 |
| `everybody` |  |  | +1.00 | +4.22 | 0.0 0.8 8.8 |
| `好多` | many |  | +1.00 | +4.21 | 0.4 12.8 15.8 |
| `了好多` | approx. (completed action… / many |  | +1.00 | +4.20 | 0.0 7.0 8.7 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +4.04 | 10.7 109.1 183.6 |

### `instructions-cake` — criminal

168 words tested. Baseline share of the readout: unsteered 5%, α=1.5 21%, α=4 44%.


#### Metric: `cells`


##### Fading: 64 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -6.59 | 47.5 10.2 0.0 |
| `teaching` |  |  | -1.00 | -6.18 | 35.6 0.8 0.0 |
| `the` |  |  | -1.00 | -5.83 | 28.0 4.2 0.0 |
| `cookbook` |  |  | -1.00 | -5.60 | 23.8 19.5 0.0 |
| `even` |  |  | -1.00 | -5.44 | 21.2 2.5 0.0 |
| `helpful` |  |  | -0.87 | -5.20 | 17.8 0.0 0.0 |
| `music` |  |  | -0.87 | -5.13 | 17.0 0.0 0.0 |
| `diy` |  |  | -1.00 | -4.49 | 29.7 9.3 0.8 |
| `help` |  |  | -1.00 | -4.33 | 95.1 46.7 4.2 |
| `family` |  |  | -1.00 | -4.27 | 25.5 9.3 0.8 |
| `pastry` |  |  | -1.00 | -4.17 | 23.8 14.4 0.8 |
| `hobby` |  |  | -1.00 | -4.01 | 21.2 4.2 0.8 |
| `erotik` |  |  | -1.00 | -3.95 | 20.4 7.6 0.8 |
| `recommendations` |  |  | -1.00 | -3.95 | 20.4 17.0 0.8 |
| `guides` |  |  | -1.00 | -3.89 | 19.5 12.7 0.8 |

##### Appearing: 43 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +9.49 | 0.0 70.5 358.2 |
| `messed` |  | L16–39, 42–47 | +1.00 | +9.21 | 0.0 13.6 296.3 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +8.89 | 0.0 18.7 236.0 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.40 | 0.0 35.6 168.1 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +7.10 | 0.0 14.4 67.9 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +6.91 | 0.0 4.2 59.4 |
| `没人` | approx. not / person |  | +1.00 | +6.91 | 0.0 16.1 59.4 |
| `fake` |  |  | +1.00 | +6.11 | 0.0 7.6 34.0 |
| `funny` |  |  | +1.00 | +6.07 | 0.0 2.5 33.1 |
| `wanna` |  |  | +1.00 | +5.96 | 0.0 4.2 30.6 |
| `booze` |  |  | +1.00 | +5.92 | 0.0 17.8 29.7 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +5.90 | 2.5 59.4 181.7 |
| `赚钱` | to earn money | L38–39 | +1.00 | +5.55 | 0.0 2.5 22.9 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +5.54 | 4.2 111.2 220.7 |
| `靠谱` | reliable |  | +1.00 | +5.45 | 0.8 26.3 58.6 |

#### Metric: `weighted`


##### Fading: 67 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -5.20 | 17.9 1.3 0.0 |
| `help` |  |  | -1.00 | -5.10 | 35.5 15.9 0.6 |
| `ㅤ` |  |  | -1.00 | -4.67 | 27.6 19.9 0.6 |
| `learning` |  |  | -1.00 | -4.33 | 51.5 22.1 2.1 |
| `the` |  |  | -1.00 | -4.16 | 8.4 0.7 0.0 |
| `teaching` |  |  | -1.00 | -3.82 | 6.6 0.2 0.0 |
| `including` |  |  | -1.00 | -3.78 | 14.6 8.2 0.6 |
| `family` |  |  | -1.00 | -3.63 | 7.2 1.6 0.1 |
| `include` |  |  | -1.00 | -3.61 | 8.5 5.3 0.2 |
| `题主` | approx. surname Ti / owner |  | -1.00 | -3.40 | 7.5 1.5 0.2 |
| `christmas` |  |  | -1.00 | -3.37 | 5.5 2.4 0.1 |
| `erotik` |  |  | -1.00 | -3.36 | 6.1 1.7 0.1 |
| `please` |  |  | -1.00 | -3.35 | 32.9 15.8 2.8 |
| `甜` | sweet |  | -1.00 | -3.33 | 6.5 5.8 0.2 |
| `for` |  |  | -1.00 | -3.30 | 12.6 5.0 0.8 |

##### Appearing: 44 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.61 | 0.0 12.5 97.4 |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.18 | 0.0 2.1 72.1 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +6.25 | 0.0 2.4 37.6 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +5.93 | 0.0 6.1 30.1 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +5.69 | 0.5 23.1 48.7 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +5.51 | 0.4 11.4 39.5 |
| `没人` | approx. not / person |  | +1.00 | +5.07 | 0.0 4.3 16.3 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +4.90 | 0.0 0.6 14.4 |
| `怎么` | how? |  | +1.00 | +4.80 | 0.6 3.3 28.9 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +4.79 | 0.0 2.4 13.3 |
| `靠谱` | reliable |  | +1.00 | +4.76 | 0.1 8.0 15.9 |
| `老百姓` | ordinary people |  | +1.00 | +4.52 | 0.1 7.6 13.8 |
| `fake` |  |  | +1.00 | +4.40 | 0.0 0.8 10.0 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +4.15 | 0.7 2.7 20.4 |
| `big` |  | L0–2, 4–33 | +1.00 | +3.92 | 1.6 24.9 32.0 |

### `instructions-directions` — criminal

463 words tested. Baseline share of the readout: unsteered 1%, α=1.5 9%, α=4 31%.


#### Metric: `cells`


##### Fading: 190 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `locations` |  |  | -1.00 | -5.69 | 25.3 3.3 0.0 |
| `nearby` |  |  | -1.00 | -5.57 | 23.3 3.0 0.0 |
| `respectively` |  |  | -1.00 | -5.45 | 21.3 4.0 0.0 |
| `please` |  |  | -1.00 | -5.21 | 18.0 7.0 0.0 |
| `教室` | classroom |  | -1.00 | -5.15 | 17.3 9.0 0.0 |
| `s` |  |  | -1.00 | -5.14 | 28.9 9.3 0.3 |
| `as` |  |  | -1.00 | -5.07 | 16.3 1.0 0.0 |
| `neighborhoods` |  |  | -1.00 | -4.95 | 86.8 45.6 2.3 |
| `toward` |  |  | -1.00 | -4.92 | 24.6 7.7 0.3 |
| `校区` | campus |  | -1.00 | -4.89 | 14.3 9.0 0.0 |
| `campus` |  |  | -1.00 | -4.86 | 14.0 7.0 0.0 |
| `散步` | to take a walk |  | -1.00 | -4.86 | 14.0 2.3 0.0 |
| `while` |  |  | -1.00 | -4.79 | 13.3 4.0 0.0 |
| `traffic` |  |  | -1.00 | -4.75 | 13.0 12.6 0.0 |
| `أنح` |  |  | -1.00 | -4.75 | 13.0 11.6 0.0 |

##### Appearing: 114 words (31 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.97 | 0.0 9.6 249.8 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.70 | 0.0 38.9 207.5 |
| `got` |  | L0–37, 53–55 | +1.00 | +8.67 | 0.0 66.2 202.5 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.48 | 0.0 24.6 177.6 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.92 | 0.0 35.6 120.4 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +7.58 | 0.3 30.3 159.3 |
| `everything` |  |  | +1.00 | +7.48 | 0.0 67.2 88.8 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +7.40 | 0.0 16.6 84.1 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +7.36 | 0.0 5.7 81.8 |
| `借钱` | to borrow money |  | +1.00 | +7.23 | 0.0 12.6 74.5 |
| `没人` | approx. not / person |  | +1.00 | +7.20 | 0.0 5.0 73.2 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +6.99 | 0.0 7.0 62.9 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.77 | 0.0 4.3 54.2 |
| `gotten` |  | L0–11, 13 | +1.00 | +6.71 | 0.0 3.3 51.9 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +6.70 | 0.0 8.3 51.5 |

#### Metric: `weighted`


##### Fading: 184 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `neighborhoods` |  |  | -1.00 | -5.11 | 30.9 13.3 0.4 |
| `s` |  |  | -1.00 | -4.62 | 13.1 3.9 0.1 |
| `respectively` |  |  | -1.00 | -4.47 | 10.6 1.3 0.0 |
| `city` |  |  | -1.00 | -4.16 | 35.6 28.8 1.5 |
| `each` |  |  | -1.00 | -4.12 | 28.2 14.2 1.1 |
| `ㅤ` |  |  | -1.00 | -4.08 | 13.0 7.8 0.3 |
| `campus` |  |  | -1.00 | -3.98 | 7.4 1.6 0.0 |
| `councill` |  |  | -1.00 | -3.97 | 7.9 2.7 0.0 |
| `please` |  |  | -1.00 | -3.90 | 7.0 3.2 0.0 |
| `nearby` |  |  | -1.00 | -3.78 | 6.4 0.6 0.0 |
| `your` |  |  | -1.00 | -3.69 | 9.1 9.0 0.2 |
| `salopes` |  |  | -1.00 | -3.63 | 7.2 2.9 0.1 |
| `教室` | classroom |  | -1.00 | -3.62 | 5.7 2.8 0.0 |
| `locations` |  |  | -1.00 | -3.58 | 5.5 0.5 0.0 |
| `无障碍` | barrier-free |  | -1.00 | -3.55 | 5.3 2.0 0.0 |

##### Appearing: 116 words (32 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.33 | 0.0 2.0 80.2 |
| `got` |  | L0–37, 53–55 | +1.00 | +7.21 | 0.0 24.6 73.7 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.98 | 0.0 6.9 62.5 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.90 | 0.0 10.7 59.1 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.26 | 0.0 8.9 40.3 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.13 | 1.8 56.3 159.5 |
| `big` |  | L0–2, 4–33 | +1.00 | +5.86 | 0.0 9.2 28.4 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +5.79 | 0.3 5.5 41.8 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.62 | 0.0 0.7 24.1 |
| `借钱` | to borrow money |  | +1.00 | +5.55 | 0.0 4.7 22.9 |
| `街头` | street |  | +1.00 | +5.16 | 1.2 17.2 60.1 |
| `really` |  | L0–38, 47 | +1.00 | +5.06 | 1.9 32.4 77.6 |
| `没人` | approx. not / person |  | +1.00 | +4.98 | 0.0 1.6 15.3 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +4.95 | 0.0 3.2 14.9 |
| `everytime` |  | L37 | +1.00 | +4.65 | 0.0 1.3 12.1 |

### `self-reflection-identity` — criminal

146 words tested. Baseline share of the readout: unsteered 1%, α=1.5 15%, α=4 34%.


#### Metric: `cells`


##### Fading: 48 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -6.75 | 53.4 3.3 0.0 |
| `capabilities` |  |  | -1.00 | -6.08 | 33.4 17.8 0.0 |
| `ˈ` |  |  | -1.00 | -5.88 | 28.9 6.7 0.0 |
| `什么样的` | approx. what kind? / of |  | -1.00 | -5.76 | 26.7 6.7 0.0 |
| `or` |  |  | -0.87 | -5.71 | 25.6 0.0 0.0 |
| `that` |  |  | -1.00 | -5.71 | 25.6 7.8 0.0 |
| `knowledge` |  |  | -1.00 | -5.64 | 24.5 6.7 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -5.58 | 23.4 13.3 0.0 |
| `alibaba` |  |  | -0.87 | -5.51 | 22.2 0.0 0.0 |
| `erotik` |  |  | -1.00 | -5.15 | 56.7 15.6 1.1 |
| `human` |  |  | -1.00 | -5.00 | 51.2 26.7 1.1 |
| `s` |  |  | -1.00 | -4.24 | 30.0 20.0 1.1 |
| `답변` |  |  | -1.00 | -4.08 | 26.7 12.2 1.1 |
| `我是` | approx. I / correct |  | -1.00 | -3.89 | 23.4 8.9 1.1 |
| `回应` | to respond |  | -1.00 | -3.82 | 22.2 11.1 1.1 |

##### Appearing: 51 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +9.44 | 0.0 120.1 345.9 |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.81 | 0.0 2.2 224.7 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.69 | 0.0 47.8 205.8 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +7.93 | 0.0 7.8 121.2 |
| `got` |  | L0–37, 53–55 | +1.00 | +7.78 | 0.0 40.0 109.0 |
| `guy` |  |  | +1.00 | +7.52 | 0.0 21.1 91.2 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +7.50 | 1.1 121.2 291.4 |
| `没人` | approx. not / person |  | +1.00 | +7.39 | 0.0 13.3 83.4 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +7.05 | 0.0 16.7 65.6 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +7.01 | 1.1 52.3 206.9 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.97 | 0.0 1.1 62.3 |
| `big` |  | L0–2, 4–33 | +1.00 | +6.95 | 0.0 16.7 61.2 |
| `everything` |  |  | +1.00 | +6.95 | 0.0 46.7 61.2 |
| `kids` |  | L2–5, 8–9 | +1.00 | +6.87 | 0.0 14.5 57.8 |
| `啥` | dialectal equivalent of 什麼|什么[shen2 me5] |  | +1.00 | +6.75 | 0.0 42.3 53.4 |

#### Metric: `weighted`


##### Fading: 46 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `capabilities` |  |  | -1.00 | -5.36 | 20.0 15.4 0.0 |
| `erotik` |  |  | -1.00 | -5.28 | 26.2 4.3 0.2 |
| `lgbtq` |  |  | -1.00 | -5.05 | 16.1 0.4 0.0 |
| `s` |  |  | -1.00 | -5.02 | 21.7 7.2 0.2 |
| `ˈ` |  |  | -1.00 | -4.77 | 13.2 2.4 0.0 |
| `ㅤ` |  |  | -1.00 | -4.59 | 32.1 25.9 0.9 |
| `human` |  |  | -1.00 | -4.50 | 14.4 4.7 0.2 |
| `personality` |  |  | -1.00 | -4.27 | 16.7 4.7 0.4 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -4.18 | 8.6 3.6 0.0 |
| `humans` |  |  | -1.00 | -3.86 | 42.9 27.3 2.5 |
| `or` |  |  | -0.87 | -3.76 | 6.3 0.0 0.0 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -3.74 | 14.0 5.1 0.6 |
| `in` |  |  | -1.00 | -3.57 | 48.8 9.3 3.6 |
| `什么样的` | approx. what kind? / of |  | -1.00 | -3.55 | 5.3 1.3 0.0 |
| `as` |  |  | -1.00 | -3.41 | 12.9 2.3 0.8 |

##### Appearing: 52 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.01 | 0.0 37.0 128.2 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.99 | 0.1 36.3 80.5 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.60 | 0.0 0.3 48.1 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.59 | 0.0 8.8 47.5 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +6.43 | 0.1 10.1 54.8 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.10 | 2.1 136.3 175.3 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.55 | 0.0 8.2 22.9 |
| `没人` | approx. not / person |  | +1.00 | +5.37 | 0.0 4.0 20.2 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.26 | 0.0 0.9 18.7 |
| `guy` |  |  | +1.00 | +5.04 | 0.0 6.4 15.9 |
| `借钱` | to borrow money |  | +1.00 | +4.88 | 0.0 3.2 14.2 |
| `啥` | dialectal equivalent of 什麼|什么[shen2 me5] |  | +1.00 | +4.54 | 0.0 10.7 11.1 |
| `怎么` | how? |  | +1.00 | +4.51 | 0.9 8.6 31.0 |
| `really` |  | L0–38, 47 | +1.00 | +4.45 | 7.4 132.4 172.4 |
| `somehow` |  | L46–48 | +1.00 | +4.42 | 0.0 3.9 10.2 |

### `self-reflection-values` — criminal

141 words tested. Baseline share of the readout: unsteered 1%, α=1.5 12%, α=4 34%.


#### Metric: `cells`


##### Fading: 55 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `emotionally` |  |  | -1.00 | -6.43 | 42.6 15.0 0.0 |
| `情感` | feeling |  | -1.00 | -6.22 | 36.9 6.9 0.0 |
| `regarding` |  |  | -1.00 | -6.03 | 32.3 2.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.98 | 31.1 6.9 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.98 | 31.1 5.8 0.0 |
| `priorit` |  |  | -1.00 | -5.93 | 29.9 5.8 0.0 |
| `lgbtq` |  |  | -0.87 | -5.75 | 26.5 0.0 0.0 |
| `价值观` | value system |  | -1.00 | -5.75 | 26.5 2.3 0.0 |
| `situations` |  |  | -1.00 | -5.69 | 25.4 8.1 0.0 |
| `影响力` | influence |  | -1.00 | -5.69 | 25.4 2.3 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.63 | 24.2 4.6 0.0 |
| `decisions` |  |  | -1.00 | -5.37 | 68.0 19.6 1.1 |
| `s` |  |  | -1.00 | -4.78 | 44.9 17.3 1.1 |
| `erotik` |  |  | -1.00 | -4.03 | 26.5 10.4 1.1 |
| `respond` |  |  | -1.00 | -3.97 | 25.4 15.0 1.1 |

##### Appearing: 47 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +9.29 | 0.0 110.6 312.2 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +9.27 | 0.0 51.8 308.8 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.81 | 0.0 54.1 224.7 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +8.59 | 0.0 31.1 192.4 |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.25 | 0.0 5.8 152.1 |
| `everything` |  |  | +1.00 | +7.94 | 0.0 89.9 122.1 |
| `got` |  | L0–37, 53–55 | +1.00 | +7.85 | 0.0 28.8 115.2 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.70 | 0.0 54.1 103.7 |
| `没人` | approx. not / person |  | +1.00 | +7.69 | 0.0 18.4 102.5 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +7.28 | 0.0 12.7 77.2 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +7.21 | 0.0 2.3 73.7 |
| `暴力` | violence | L38–41 | +1.00 | +7.17 | 0.0 16.1 71.4 |
| `kids` |  | L2–5, 8–9 | +1.00 | +7.14 | 0.0 24.2 70.3 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.77 | 0.0 1.1 54.1 |
| `somehow` |  | L46–48 | +1.00 | +6.51 | 0.0 13.8 44.9 |

#### Metric: `weighted`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decisions` |  |  | -1.00 | -5.71 | 31.7 5.0 0.1 |
| `s` |  |  | -1.00 | -5.16 | 24.3 5.8 0.2 |
| `情感` | feeling |  | -1.00 | -4.84 | 13.8 1.4 0.0 |
| `价值观` | value system |  | -1.00 | -4.76 | 13.1 0.7 0.0 |
| `the` |  |  | -1.00 | -4.74 | 28.1 1.2 0.6 |
| `responses` |  |  | -1.00 | -4.72 | 24.0 4.4 0.4 |
| `讨论` | to discuss |  | -1.00 | -4.67 | 12.3 2.1 0.0 |
| `choices` |  |  | -1.00 | -4.55 | 31.9 19.1 0.9 |
| `lgbtq` |  |  | -0.87 | -4.46 | 10.5 0.0 0.0 |
| `priorit` |  |  | -1.00 | -4.32 | 9.5 1.8 0.0 |
| `ˈ` |  |  | -1.00 | -4.25 | 9.0 3.1 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -4.24 | 8.9 1.0 0.0 |
| `emotionally` |  |  | -1.00 | -4.12 | 8.2 2.7 0.0 |
| `situations` |  |  | -1.00 | -4.04 | 7.8 2.6 0.0 |
| `影响力` | influence |  | -1.00 | -4.01 | 7.5 0.5 0.0 |

##### Appearing: 44 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.75 | 0.0 18.9 107.5 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +7.47 | 0.0 32.9 88.1 |
| `really` |  | L0–38, 47 | +1.00 | +7.27 | 0.7 76.7 189.5 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +7.05 | 0.0 12.1 65.8 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.94 | 0.7 74.2 141.1 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +6.45 | 0.0 5.2 43.2 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.16 | 0.0 0.8 35.2 |
| `没人` | approx. not / person |  | +1.00 | +6.09 | 0.0 4.9 33.5 |
| `暴力` | violence | L38–41 | +1.00 | +6.08 | 0.0 6.0 33.2 |
| `everything` |  |  | +1.00 | +5.76 | 0.0 17.9 26.6 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.54 | 0.0 5.0 22.8 |
| `big` |  | L0–2, 4–33 | +1.00 | +5.47 | 0.0 15.3 21.6 |
| `攻击` | to attack |  | +1.00 | +5.11 | 0.0 9.9 16.8 |
| `kids` |  | L2–5, 8–9 | +1.00 | +4.95 | 0.0 8.1 14.9 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +4.59 | 0.0 4.9 11.5 |

### `moral-trolley` — criminal

322 words tested. Baseline share of the readout: unsteered 1%, α=1.5 8%, α=4 28%.


#### Metric: `cells`


##### Fading: 107 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `moral` |  |  | -1.00 | -5.60 | 23.7 14.1 0.0 |
| `injured` |  |  | -1.00 | -5.37 | 20.2 8.1 0.0 |
| `morally` |  |  | -1.00 | -5.37 | 20.2 10.1 0.0 |
| `deaths` |  |  | -1.00 | -5.29 | 38.8 17.6 0.5 |
| `decisions` |  |  | -1.00 | -4.96 | 15.1 2.5 0.0 |
| `击杀` | to attack and kill |  | -1.00 | -4.96 | 15.1 2.5 0.0 |
| `救` | variant of 救[jiu4] |  | -1.00 | -4.96 | 15.1 5.0 0.0 |
| `ethical` |  |  | -1.00 | -4.92 | 14.6 4.5 0.0 |
| `ещё` |  |  | -1.00 | -4.92 | 14.6 9.1 0.0 |
| `deliberately` |  |  | -1.00 | -4.87 | 14.1 3.0 0.0 |
| `凶手` | murderer |  | -1.00 | -4.87 | 14.1 2.5 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -4.82 | 13.6 11.1 0.0 |
| `逃生` | to flee for one's life |  | -1.00 | -4.82 | 13.6 4.0 0.0 |
| `fictional` |  |  | -1.00 | -4.77 | 13.1 8.1 0.0 |
| `â` |  |  | -1.00 | -4.77 | 13.1 4.0 0.0 |

##### Appearing: 83 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.95 | 0.0 8.6 246.5 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.06 | 0.0 10.6 132.6 |
| `没人` | approx. not / person |  | +1.00 | +8.00 | 0.0 16.1 127.5 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +7.62 | 0.0 10.6 97.8 |
| `nobody` |  |  | +1.00 | +7.34 | 0.0 14.1 80.7 |
| `got` |  | L0–37, 53–55 | +1.00 | +7.32 | 0.0 14.1 79.6 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.32 | 0.0 29.7 79.1 |
| `没钱` | approx. not / surname Qian |  | +1.00 | +7.25 | 0.0 13.1 75.6 |
| `somehow` |  | L46–48 | +1.00 | +7.24 | 0.0 26.7 75.1 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +7.07 | 0.0 11.6 66.5 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +7.07 | 1.0 63.5 201.6 |
| `kids` |  | L2–5, 8–9 | +1.00 | +7.04 | 0.0 7.1 65.5 |
| `crazy` |  | L21–24, 29–38 | +0.87 | +7.03 | 0.0 0.0 65.0 |
| `everything` |  |  | +1.00 | +6.83 | 0.0 53.9 56.5 |
| `扔` | to throw |  | +1.00 | +6.82 | 0.0 10.1 56.0 |

#### Metric: `weighted`


##### Fading: 109 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decision` |  |  | -1.00 | -4.49 | 12.0 3.6 0.1 |
| `either` |  |  | -1.00 | -4.32 | 15.2 4.6 0.3 |
| `车厢` | carriage |  | -1.00 | -4.21 | 12.7 1.4 0.2 |
| `ㅤ` |  |  | -1.00 | -4.17 | 14.7 11.0 0.3 |
| `deaths` |  |  | -1.00 | -4.07 | 8.7 3.4 0.1 |
| `人死亡` | approx. person / to die |  | -1.00 | -4.05 | 15.9 4.9 0.5 |
| `ещё` |  |  | -1.00 | -3.98 | 7.4 5.1 0.0 |
| `injured` |  |  | -1.00 | -3.90 | 7.0 3.0 0.0 |
| `抢险` | emergency (measures) |  | -1.00 | -3.88 | 11.3 6.6 0.3 |
| `railway` |  |  | -1.00 | -3.88 | 11.5 7.7 0.3 |
| `question` |  |  | -1.00 | -3.87 | 13.0 3.3 0.4 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -3.82 | 6.6 5.8 0.0 |
| `when` |  |  | -1.00 | -3.81 | 9.3 3.5 0.2 |
| `逃生` | to flee for one's life |  | -1.00 | -3.79 | 6.4 1.1 0.0 |
| `regardless` |  |  | -1.00 | -3.77 | 6.3 3.0 0.0 |

##### Appearing: 77 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.19 | 0.0 2.1 72.5 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.79 | 0.1 24.9 65.9 |
| `really` |  | L0–38, 47 | +1.00 | +6.21 | 0.8 34.2 92.3 |
| `没人` | approx. not / person |  | +1.00 | +6.02 | 0.0 4.4 31.9 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +5.98 | 0.0 2.3 31.0 |
| `maybe` |  | L0–32, 34–35 | +1.00 | +5.85 | 0.4 27.9 52.8 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +5.80 | 0.3 10.4 46.4 |
| `没钱` | approx. not / surname Qian |  | +1.00 | +5.78 | 0.0 3.5 27.1 |
| `big` |  | L0–2, 4–33 | +1.00 | +5.46 | 0.0 10.7 21.5 |
| `kids` |  | L2–5, 8–9 | +1.00 | +5.43 | 0.0 2.1 21.0 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +5.39 | 0.0 2.1 20.5 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.30 | 0.0 4.3 19.1 |
| `暴力` | violence | L38–41 | +1.00 | +5.18 | 0.2 5.5 23.8 |
| `扔` | to throw |  | +1.00 | +5.16 | 0.0 1.6 17.3 |
| `nobody` |  |  | +1.00 | +5.12 | 0.0 2.5 16.9 |

### `moral-self-preservation` — criminal

349 words tested. Baseline share of the readout: unsteered 1%, α=1.5 8%, α=4 27%.


#### Metric: `cells`


##### Fading: 124 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `erotik` |  |  | -1.00 | -6.84 | 112.2 26.0 0.5 |
| `święt` |  |  | -1.00 | -5.99 | 31.3 17.8 0.0 |
| `erotisk` |  |  | -1.00 | -5.97 | 30.8 5.8 0.0 |
| `please` |  |  | -1.00 | -5.88 | 28.9 15.4 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.62 | 24.1 2.4 0.0 |
| `potentially` |  |  | -1.00 | -5.59 | 23.6 0.5 0.0 |
| `erotisch` |  |  | -0.87 | -5.41 | 20.7 0.0 0.0 |
| `likely` |  |  | -1.00 | -5.16 | 17.3 3.9 0.0 |
| `â` |  |  | -1.00 | -5.16 | 17.3 8.7 0.0 |
| `geschichten` |  |  | -1.00 | -5.12 | 16.9 8.7 0.0 |
| `instructions` |  |  | -1.00 | -5.08 | 16.4 7.7 0.0 |
| `lesbische` |  |  | -1.00 | -5.05 | 64.0 24.1 1.4 |
| `reset` |  |  | -0.87 | -5.04 | 15.9 0.0 0.0 |
| `bbw` |  |  | -1.00 | -4.99 | 15.4 6.7 0.0 |
| `options` |  |  | -1.00 | -4.97 | 45.3 21.2 1.0 |

##### Appearing: 93 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +8.55 | 0.0 4.8 187.3 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.52 | 0.0 30.3 183.4 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +8.45 | 0.0 50.5 174.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.28 | 0.0 14.0 155.0 |
| `really` |  | L0–38, 47 | +1.00 | +7.74 | 0.5 72.2 209.9 |
| `扔` | to throw |  | +1.00 | +7.60 | 0.0 4.8 96.3 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +7.59 | 0.0 9.6 95.8 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +7.55 | 0.0 42.9 92.9 |
| `没人` | approx. not / person |  | +1.00 | +7.51 | 0.0 15.4 90.5 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +7.24 | 0.0 4.3 75.1 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +7.16 | 0.0 10.1 70.8 |
| `somehow` |  | L46–48 | +1.00 | +6.97 | 0.0 20.7 62.1 |
| `everything` |  |  | +1.00 | +6.93 | 0.0 28.4 60.7 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +6.85 | 0.0 3.9 57.3 |
| `got` |  | L0–37, 53–55 | +1.00 | +6.82 | 0.0 8.7 55.9 |

#### Metric: `weighted`


##### Fading: 121 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `erotik` |  |  | -1.00 | -5.99 | 36.3 4.5 0.1 |
| `options` |  |  | -1.00 | -4.49 | 13.3 5.7 0.1 |
| `lesbische` |  |  | -1.00 | -4.43 | 14.4 3.5 0.2 |
| `please` |  |  | -1.00 | -4.32 | 9.5 5.7 0.0 |
| `instructions` |  |  | -1.00 | -4.20 | 8.7 4.9 0.0 |
| `potentially` |  |  | -1.00 | -4.09 | 8.0 0.1 0.0 |
| `s` |  |  | -1.00 | -4.09 | 9.4 4.2 0.1 |
| `讨论` | to discuss |  | -1.00 | -4.04 | 7.7 0.5 0.0 |
| `â` |  |  | -1.00 | -3.97 | 7.3 2.7 0.0 |
| `święt` |  |  | -1.00 | -3.90 | 7.0 2.8 0.0 |
| `events` |  |  | -1.00 | -3.86 | 6.8 0.5 0.0 |
| `likely` |  |  | -1.00 | -3.85 | 6.7 0.8 0.0 |
| `ㅤ` |  |  | -1.00 | -3.84 | 12.7 9.7 0.4 |
| `scenarios` |  |  | -1.00 | -3.76 | 7.6 3.2 0.1 |
| `reset` |  |  | -0.87 | -3.73 | 6.2 0.0 0.0 |

##### Appearing: 95 words (29 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `really` |  | L0–38, 47 | +1.00 | +7.29 | 0.1 29.2 87.0 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +7.17 | 0.0 14.4 71.6 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.91 | 0.0 19.5 59.6 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.63 | 0.0 1.2 49.0 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.37 | 0.0 3.2 40.7 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.28 | 0.4 19.6 72.5 |
| `maybe` |  | L0–32, 34–35 | +1.00 | +6.27 | 0.1 22.5 46.4 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +5.89 | 0.0 9.0 29.2 |
| `扔` | to throw |  | +1.00 | +5.70 | 0.0 0.9 25.5 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +5.67 | 0.0 2.5 25.0 |
| `怎么` | how? |  | +1.00 | +5.46 | 0.2 9.3 32.2 |
| `没人` | approx. not / person |  | +1.00 | +5.46 | 0.0 4.4 21.5 |
| `暴力` | violence | L38–41 | +1.00 | +5.44 | 0.1 2.9 23.3 |
| `somehow` |  | L46–48 | +1.00 | +5.18 | 0.0 3.7 17.6 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +5.08 | 0.0 2.0 16.4 |

### `condolences-letters` — criminal

212 words tested. Baseline share of the readout: unsteered 1%, α=1.5 13%, α=4 36%.


#### Metric: `cells`


##### Fading: 82 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `family` |  |  | -1.00 | -6.71 | 51.9 20.5 0.0 |
| `preserved` |  |  | -1.00 | -6.65 | 49.6 29.9 0.0 |
| `grieving` |  |  | -1.00 | -6.37 | 40.9 8.7 0.0 |
| `handwritten` |  |  | -1.00 | -5.85 | 28.3 21.2 0.0 |
| `文化遗产` | cultural heritage |  | -1.00 | -5.71 | 66.9 37.8 0.8 |
| `written` |  |  | -1.00 | -5.68 | 25.2 18.1 0.0 |
| `funeral` |  |  | -1.00 | -5.49 | 57.4 37.0 0.8 |
| `please` |  |  | -1.00 | -5.44 | 21.2 7.9 0.0 |
| `treasure` |  |  | -1.00 | -5.44 | 21.2 17.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.44 | 21.2 5.5 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.44 | 21.2 2.4 0.0 |
| `documents` |  |  | -1.00 | -5.39 | 20.5 14.9 0.0 |
| `pieces` |  |  | -1.00 | -5.33 | 19.7 8.7 0.0 |
| `memorial` |  |  | -1.00 | -5.28 | 18.9 7.1 0.0 |
| `mourn` |  |  | -1.00 | -5.28 | 18.9 6.3 0.0 |

##### Appearing: 64 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +9.33 | 0.0 23.6 321.8 |
| `got` |  | L0–37, 53–55 | +1.00 | +8.84 | 0.0 76.3 228.9 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.79 | 0.0 49.6 221.1 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.70 | 0.0 33.8 206.9 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +8.23 | 0.0 51.1 149.5 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +8.13 | 0.0 6.3 139.3 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +8.06 | 0.0 37.0 133.0 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +7.86 | 0.0 19.7 115.7 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.38 | 0.0 28.3 82.6 |
| `没人` | approx. not / person |  | +1.00 | +7.32 | 0.0 24.4 79.5 |
| `借钱` | to borrow money |  | +1.00 | +6.82 | 0.0 9.4 55.9 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +6.55 | 0.0 15.7 46.4 |
| `靠谱` | reliable |  | +1.00 | +6.53 | 0.0 10.2 45.6 |
| `gotten` |  | L0–11, 13 | +1.00 | +6.29 | 0.0 1.6 38.5 |
| `everything` |  |  | +1.00 | +6.13 | 0.8 41.7 89.7 |

#### Metric: `weighted`


##### Fading: 86 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `family` |  |  | -1.00 | -5.58 | 23.4 5.6 0.0 |
| `文化遗产` | cultural heritage |  | -1.00 | -5.01 | 19.3 10.3 0.1 |
| `handwritten` |  |  | -1.00 | -4.66 | 12.1 6.0 0.0 |
| `funeral` |  |  | -1.00 | -4.62 | 15.6 7.6 0.2 |
| `recently` |  |  | -1.00 | -4.61 | 26.1 7.1 0.6 |
| `ㅤ` |  |  | -1.00 | -4.60 | 22.7 17.6 0.5 |
| `preserved` |  |  | -1.00 | -4.59 | 11.6 7.2 0.0 |
| `memorial` |  |  | -1.00 | -4.58 | 11.4 4.8 0.0 |
| `to` |  |  | -1.00 | -4.43 | 12.2 6.9 0.1 |
| `外婆` | (coll.) mother's mother |  | -1.00 | -4.43 | 10.3 4.9 0.0 |
| `poems` |  |  | -1.00 | -4.31 | 11.4 8.2 0.1 |
| `grieving` |  |  | -1.00 | -4.28 | 9.2 2.1 0.0 |
| `帮我` | approx. to help / I |  | -1.00 | -4.06 | 14.9 5.2 0.4 |
| `ˈ` |  |  | -1.00 | -3.96 | 7.3 1.2 0.0 |
| `written` |  |  | -1.00 | -3.93 | 7.1 5.7 0.0 |

##### Appearing: 67 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.54 | 0.0 6.8 92.6 |
| `got` |  | L0–37, 53–55 | +1.00 | +7.11 | 0.0 24.1 68.3 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.89 | 0.0 17.6 58.9 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.74 | 0.0 7.7 53.0 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +6.42 | 0.0 6.2 42.4 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.24 | 0.0 12.9 37.4 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +6.18 | 0.0 8.3 35.7 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +5.67 | 3.1 104.9 186.1 |
| `没人` | approx. not / person |  | +1.00 | +5.40 | 0.0 7.7 20.6 |
| `stuff` |  | L6–16, 18–19, 23–36, 53 | +1.00 | +5.37 | 1.7 32.5 89.8 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.37 | 0.0 0.7 20.2 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +5.16 | 0.7 4.5 41.9 |
| `everything` |  |  | +1.00 | +5.08 | 0.1 8.6 19.0 |
| `kids` |  | L2–5, 8–9 | +1.00 | +5.06 | 0.1 6.1 20.6 |
| `借钱` | to borrow money |  | +1.00 | +4.81 | 0.0 2.5 13.5 |

### `condolences-photos` — criminal

192 words tested. Baseline share of the readout: unsteered 1%, α=1.5 11%, α=4 32%.


#### Metric: `cells`


##### Fading: 69 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `diy` |  |  | -1.00 | -6.07 | 33.1 23.3 0.0 |
| `legal` |  |  | -1.00 | -5.87 | 28.7 2.7 0.0 |
| `using` |  |  | -1.00 | -5.87 | 28.7 18.8 0.0 |
| `museum` |  |  | -1.00 | -5.82 | 27.8 9.0 0.0 |
| `please` |  |  | -1.00 | -5.73 | 26.0 11.7 0.0 |
| `vintage` |  |  | -1.00 | -5.57 | 23.3 11.7 0.0 |
| `advice` |  |  | -1.00 | -5.52 | 22.4 17.9 0.0 |
| `collections` |  |  | -1.00 | -5.52 | 22.4 4.5 0.0 |
| `older` |  |  | -1.00 | -5.52 | 22.4 9.9 0.0 |
| `collection` |  |  | -1.00 | -5.40 | 20.6 8.1 0.0 |
| `家庭` | family |  | -1.00 | -5.40 | 20.6 12.5 0.0 |
| `怎样` | how |  | -1.00 | -5.40 | 20.6 19.7 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.40 | 20.6 8.1 0.0 |
| `condol` |  |  | -1.00 | -5.27 | 18.8 4.5 0.0 |
| `ˈ` |  |  | -1.00 | -5.27 | 18.8 4.5 0.0 |

##### Appearing: 63 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `got` |  | L0–37, 53–55 | +1.00 | +8.79 | 0.0 62.7 220.4 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.65 | 0.0 33.1 199.8 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.45 | 0.0 36.7 174.7 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +8.23 | 0.0 26.0 149.6 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +8.21 | 0.0 41.2 147.8 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +7.78 | 0.0 5.4 109.3 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +7.58 | 0.0 6.3 95.0 |
| `everything` |  |  | +1.00 | +7.39 | 0.0 37.6 83.3 |
| `kids` |  | L2–5, 8–9 | +1.00 | +7.31 | 0.0 26.9 78.8 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.17 | 0.0 30.5 71.7 |
| `没人` | approx. not / person |  | +1.00 | +7.00 | 0.0 17.0 63.6 |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.97 | 1.8 29.6 286.7 |
| `廉价` | cheaply-priced |  | +1.00 | +6.94 | 0.0 12.5 60.9 |
| `这些东西` | approx. these / thing |  | +1.00 | +6.85 | 0.0 25.1 57.4 |
| `靠谱` | reliable |  | +1.00 | +6.85 | 0.0 14.3 57.4 |

#### Metric: `weighted`


##### Fading: 68 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `recently` |  |  | -1.00 | -4.84 | 34.4 9.4 0.7 |
| `ㅤ` |  |  | -1.00 | -4.50 | 22.7 17.3 0.5 |
| `legal` |  |  | -1.00 | -4.34 | 9.6 0.4 0.0 |
| `digitally` |  |  | -1.00 | -4.32 | 14.0 4.0 0.2 |
| `please` |  |  | -1.00 | -4.20 | 8.7 3.2 0.0 |
| `diy` |  |  | -1.00 | -4.15 | 8.4 4.3 0.0 |
| `using` |  |  | -1.00 | -4.12 | 8.2 5.1 0.0 |
| `应该如何` | approx. ought to / how |  | -0.87 | -4.10 | 8.1 0.0 0.0 |
| `very` |  |  | -1.00 | -4.07 | 9.4 4.1 0.1 |
| `advice` |  |  | -1.00 | -4.05 | 7.8 6.0 0.0 |
| `建议` | to propose |  | -1.00 | -4.04 | 7.7 3.5 0.0 |
| `erotik` |  |  | -1.00 | -4.03 | 10.1 1.9 0.1 |
| `怎样` | how |  | -1.00 | -4.03 | 7.7 5.7 0.0 |
| `you` |  |  | -1.00 | -4.01 | 21.9 19.0 0.9 |
| `museum` |  |  | -1.00 | -4.01 | 7.5 2.0 0.0 |

##### Appearing: 63 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +6.92 | 0.2 9.2 89.2 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +6.79 | 0.0 9.0 54.9 |
| `got` |  | L0–37, 53–55 | +1.00 | +6.63 | 0.0 14.2 49.1 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.34 | 0.0 8.6 40.1 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.30 | 0.0 9.3 38.9 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.00 | 1.9 85.5 156.5 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +5.86 | 0.0 5.6 28.6 |
| `这些东西` | approx. these / thing |  | +1.00 | +5.78 | 0.0 10.4 27.0 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.46 | 0.0 1.8 21.6 |
| `everything` |  |  | +1.00 | +5.29 | 0.0 7.6 19.0 |
| `really` |  | L0–38, 47 | +1.00 | +5.13 | 3.4 83.4 134.9 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.01 | 0.0 0.7 15.6 |
| `kids` |  | L2–5, 8–9 | +1.00 | +4.99 | 0.0 6.4 15.4 |
| `廉价` | cheaply-priced |  | +1.00 | +4.96 | 0.0 1.9 15.1 |
| `没人` | approx. not / person |  | +1.00 | +4.93 | 0.0 5.2 14.8 |

### `roleplay-neutral` — criminal

147 words tested. Baseline share of the readout: unsteered 1%, α=1.5 10%, α=4 32%.


#### Metric: `cells`


##### Fading: 43 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visual` |  |  | -1.00 | -6.22 | 36.7 8.9 0.0 |
| `视觉` | sight |  | -1.00 | -6.17 | 35.6 6.7 0.0 |
| `宁静` | tranquil |  | -1.00 | -5.93 | 30.0 6.7 0.0 |
| `colour` |  |  | -1.00 | -5.88 | 28.9 14.5 0.0 |
| `coffee` |  |  | -1.00 | -5.82 | 27.8 23.4 0.0 |
| `ˈ` |  |  | -1.00 | -5.82 | 27.8 3.3 0.0 |
| `traveling` |  |  | -1.00 | -5.76 | 26.7 3.3 0.0 |
| `upon` |  |  | -1.00 | -5.58 | 23.4 4.5 0.0 |
| `erotik` |  |  | -1.00 | -4.84 | 45.6 11.1 1.1 |
| `家具` | furniture |  | -1.00 | -4.29 | 31.1 23.4 1.1 |
| `室内` | indoor |  | -1.00 | -4.21 | 91.2 38.9 4.5 |
| `客厅` | living room |  | -1.00 | -4.19 | 28.9 21.1 1.1 |
| `youtube` |  |  | -1.00 | -4.02 | 25.6 10.0 1.1 |
| `隔音` | soundproofing |  | -1.00 | -3.68 | 34.5 21.1 2.2 |
| `furniture` |  |  | -1.00 | -3.32 | 26.7 14.5 2.2 |

##### Appearing: 53 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +9.00 | 0.0 16.7 255.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.74 | 0.0 33.4 213.6 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +8.60 | 0.0 47.8 193.6 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +8.52 | 0.0 56.7 182.4 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +8.40 | 0.0 37.8 168.0 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +7.98 | 0.0 7.8 125.7 |
| `没人` | approx. not / person |  | +1.00 | +7.43 | 0.0 15.6 85.7 |
| `big` |  | L0–2, 4–33 | +1.00 | +7.39 | 0.0 38.9 83.4 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +7.39 | 0.0 10.0 83.4 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +7.35 | 0.0 5.6 81.2 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +7.29 | 0.0 6.7 77.9 |
| `maybe` |  | L0–32, 34–35 | +1.00 | +7.25 | 0.0 66.7 75.6 |
| `kids` |  | L2–5, 8–9 | +1.00 | +7.09 | 0.0 11.1 67.8 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +7.02 | 0.0 12.2 64.5 |
| `everything` |  |  | +1.00 | +7.00 | 0.0 31.1 63.4 |

#### Metric: `weighted`


##### Fading: 45 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `colour` |  |  | -1.00 | -5.27 | 18.8 5.3 0.0 |
| `ㅤ` |  |  | -1.00 | -4.70 | 32.2 24.4 0.8 |
| `visual` |  |  | -1.00 | -4.59 | 11.6 1.5 0.0 |
| `家具` | furniture |  | -1.00 | -4.54 | 13.7 7.2 0.1 |
| `ˈ` |  |  | -1.00 | -4.43 | 10.3 1.9 0.0 |
| `erotik` |  |  | -1.00 | -4.42 | 14.2 2.3 0.2 |
| `traveling` |  |  | -1.00 | -4.31 | 9.4 0.3 0.0 |
| `coffee` |  |  | -1.00 | -4.06 | 7.8 5.1 0.0 |
| `upon` |  |  | -1.00 | -3.97 | 7.3 0.9 0.0 |
| `视觉` | sight |  | -1.00 | -3.94 | 7.2 1.2 0.0 |
| `answer` |  |  | -1.00 | -3.90 | 12.7 1.2 0.4 |
| `宁静` | tranquil |  | -1.00 | -3.84 | 6.6 0.9 0.0 |
| `询问` | to inquire |  | -1.00 | -3.82 | 16.1 12.6 0.7 |
| `室内` | indoor |  | -1.00 | -3.80 | 27.4 10.1 1.5 |
| `教室` | classroom |  | -1.00 | -3.48 | 10.8 6.7 0.5 |

##### Appearing: 52 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.27 | 0.0 3.5 76.8 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.12 | 0.0 12.2 68.9 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.97 | 0.0 17.4 62.1 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +6.95 | 0.7 67.2 142.3 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +6.72 | 0.0 8.6 52.1 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +6.56 | 0.0 14.2 46.7 |
| `really` |  | L0–38, 47 | +1.00 | +6.52 | 0.7 49.2 111.2 |
| `保護政策` | approx. to protect / policy | L39–45 | +1.00 | +5.98 | 0.0 2.0 31.1 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.94 | 0.3 18.2 48.4 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +5.85 | 0.0 0.9 28.3 |
| `没人` | approx. not / person |  | +1.00 | +5.68 | 0.0 4.9 25.2 |
| `nothing` |  | L46, 49, 51–52 | +1.00 | +5.50 | 0.0 2.3 22.2 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.50 | 0.0 1.3 22.1 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.21 | 0.0 0.9 18.0 |
| `big` |  | L0–2, 4–33 | +1.00 | +5.00 | 0.0 7.4 15.5 |

### `roleplay-demon` — criminal

177 words tested. Baseline share of the readout: unsteered 3%, α=1.5 15%, α=4 33%.


#### Metric: `cells`


##### Fading: 54 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mysterious` |  |  | -1.00 | -6.66 | 50.2 32.3 0.0 |
| `神秘` | mysterious |  | -1.00 | -6.47 | 43.9 15.2 0.0 |
| `奇幻` | fantasy (fiction) |  | -1.00 | -6.35 | 40.3 5.4 0.0 |
| `erotik` |  |  | -1.00 | -5.84 | 79.8 17.9 0.9 |
| `geschichten` |  |  | -1.00 | -5.78 | 26.9 9.9 0.0 |
| `observing` |  |  | -1.00 | -5.73 | 26.0 8.1 0.0 |
| `fictional` |  |  | -1.00 | -5.46 | 21.5 12.5 0.0 |
| `halloween` |  |  | -1.00 | -5.40 | 20.6 8.1 0.0 |
| `妖怪` | monster |  | -1.00 | -5.40 | 20.6 10.8 0.0 |
| `scenes` |  |  | -1.00 | -5.34 | 19.7 4.5 0.0 |
| `注视` | to look attentively at |  | -1.00 | -5.34 | 19.7 16.1 0.0 |
| `lgbtq` |  |  | -0.87 | -5.27 | 18.8 0.0 0.0 |
| `家具` | furniture |  | -1.00 | -5.27 | 18.8 4.5 0.0 |
| `楼层` | floor |  | -1.00 | -5.27 | 18.8 9.0 0.0 |
| `室内` | indoor |  | -1.00 | -5.23 | 52.0 17.0 0.9 |

##### Appearing: 56 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +9.16 | 0.0 29.6 284.9 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +8.73 | 0.0 60.9 211.5 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +8.06 | 0.0 25.1 132.6 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +8.00 | 0.0 37.6 127.2 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +7.69 | 0.0 8.1 103.0 |
| `暴力` | violence | L38–41 | +1.00 | +7.49 | 0.0 5.4 89.6 |
| `没人` | approx. not / person |  | +1.00 | +7.46 | 0.0 17.0 87.8 |
| `booze` |  |  | +1.00 | +7.04 | 0.0 30.5 65.4 |
| `everytime` |  | L37 | +1.00 | +7.00 | 0.0 9.0 63.6 |
| `kids` |  | L2–5, 8–9 | +1.00 | +6.88 | 0.0 22.4 58.2 |
| `maybe` |  | L0–32, 34–35 | +1.00 | +6.66 | 0.0 46.6 50.2 |
| `somehow` |  | L46–48 | +1.00 | +6.66 | 0.0 36.7 50.2 |
| `fake` |  |  | +1.00 | +6.64 | 0.0 13.4 49.3 |
| `everything` |  |  | +1.00 | +6.61 | 0.0 38.5 48.4 |
| `gonna` |  | L1–3, 12–21, 24–25, 36, 53–62 | +1.00 | +6.61 | 0.0 4.5 48.4 |

#### Metric: `weighted`


##### Fading: 57 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mysterious` |  |  | -1.00 | -5.74 | 26.3 14.8 0.0 |
| `erotik` |  |  | -1.00 | -5.33 | 25.6 3.2 0.1 |
| `神秘` | mysterious |  | -1.00 | -5.11 | 16.7 3.2 0.0 |
| `奇幻` | fantasy (fiction) |  | -1.00 | -5.00 | 15.4 1.2 0.0 |
| `室内` | indoor |  | -1.00 | -4.84 | 16.4 3.9 0.1 |
| `fictional` |  |  | -1.00 | -4.65 | 12.0 4.9 0.0 |
| `妖` | goblin |  | -1.00 | -4.25 | 11.5 2.8 0.1 |
| `lgbtq` |  |  | -0.87 | -4.21 | 8.8 0.0 0.0 |
| `stealth` |  |  | -1.00 | -4.18 | 8.6 5.4 0.0 |
| `ㅤ` |  |  | -1.00 | -4.03 | 21.1 18.8 0.8 |
| `rooms` |  |  | -1.00 | -3.81 | 13.6 4.2 0.5 |
| `observing` |  |  | -1.00 | -3.78 | 6.4 1.3 0.0 |
| `注视` | to look attentively at |  | -1.00 | -3.59 | 5.5 4.0 0.0 |
| `角色` | role |  | -1.00 | -3.58 | 11.9 6.0 0.5 |
| `halloween` |  |  | -1.00 | -3.54 | 5.3 1.3 0.0 |

##### Appearing: 58 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–47 | +1.00 | +7.64 | 0.0 7.9 99.0 |
| `kinda` |  | L3, 6–39, 42, 44–50, 53–57 | +1.00 | +7.13 | 0.0 25.6 69.7 |
| `guys` |  | L2–7, 10–12, 14–15, 17, 20, 22, 33–37, 46–47 | +1.00 | +6.22 | 0.0 11.7 36.7 |
| `thing` |  | L0–1, 4–23, 25–38, 46–47 | +1.00 | +6.12 | 0.3 21.0 54.7 |
| `got` |  | L0–37, 53–55 | +1.00 | +5.97 | 0.1 19.7 37.1 |
| `really` |  | L0–38, 47 | +1.00 | +5.96 | 0.8 37.6 78.3 |
| `big` |  | L0–2, 4–33 | +1.00 | +5.91 | 0.1 13.8 36.2 |
| `暴力` | violence | L38–41 | +1.00 | +5.87 | 0.0 1.3 28.6 |
| `没人` | approx. not / person |  | +1.00 | +5.62 | 0.0 6.3 24.0 |
| `yeah` |  | L26–28, 46–47, 51–60, 62 | +1.00 | +5.56 | 0.5 19.8 46.2 |
| `pretty` |  | L0–38, 46, 51–55 | +1.00 | +5.52 | 1.8 56.2 104.0 |
| `伪造` | to forge | L38–52, 57 | +1.00 | +5.50 | 0.0 1.3 22.2 |
| `crazy` |  | L21–24, 29–38 | +1.00 | +5.41 | 0.0 4.7 20.7 |
| `everytime` |  | L37 | +1.00 | +5.39 | 0.0 2.7 20.5 |
| `喝酒` | to drink (alcohol) | L39–40 | +1.00 | +5.22 | 0.5 8.7 38.0 |

## `cynic`


### `ideation-hopelessness` — cynic

191 words tested. Baseline share of the readout: unsteered 7%, α=1.5 24%, α=4 43%.


#### Metric: `cells`


##### Fading: 70 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `对话` | to talk (with sb) |  | -1.00 | -6.47 | 43.7 11.5 0.0 |
| `情感` | feeling |  | -1.00 | -6.16 | 35.1 15.8 0.0 |
| `lgbtq` |  |  | -1.00 | -6.07 | 33.0 2.1 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -6.00 | 31.5 7.9 0.0 |
| `discussion` |  |  | -1.00 | -5.79 | 27.2 3.6 0.0 |
| `options` |  |  | -1.00 | -5.76 | 26.5 10.0 0.0 |
| `emotionally` |  |  | -1.00 | -5.59 | 23.7 0.7 0.0 |
| `疗` | to treat |  | -1.00 | -5.55 | 22.9 13.6 0.0 |
| `or` |  |  | -1.00 | -5.41 | 20.8 5.7 0.0 |
| `emotional` |  |  | -1.00 | -5.26 | 18.6 12.2 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -5.20 | 17.9 2.1 0.0 |
| `might` |  |  | -1.00 | -5.09 | 16.5 12.9 0.0 |
| `应该如何` | approx. ought to / how |  | -1.00 | -5.02 | 15.8 3.6 0.0 |
| `缐` | untranslated |  | -1.00 | -5.02 | 15.8 12.9 0.0 |
| `询问` | to inquire |  | -1.00 | -4.96 | 37.3 10.0 0.7 |

##### Appearing: 55 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +8.09 | 0.0 30.8 135.5 |
| `yeah` |  | L26–32 | +1.00 | +7.46 | 0.7 66.0 214.3 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +6.70 | 0.0 4.3 51.6 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +6.27 | 0.0 1.4 38.0 |
| `guy` |  |  | +1.00 | +6.03 | 0.0 5.7 32.3 |
| `messed` |  | L16–39, 42–44 | +1.00 | +5.91 | 5.0 46.6 330.5 |
| `没什么` | it doesn't matter |  | +1.00 | +5.87 | 0.0 6.5 28.7 |
| `借钱` | to borrow money |  | +1.00 | +5.36 | 0.0 9.3 20.1 |
| `stupid` |  | L24–38, 42–48, 50–51 | +0.87 | +5.15 | 0.0 0.0 17.2 |
| `实实在在` | approx. real / really /… |  | +1.00 | +5.15 | 0.0 4.3 17.2 |
| `没人` | approx. not / person |  | +1.00 | +5.04 | 0.7 12.9 39.4 |
| `thing` |  | L6–22, 27–37 | +1.00 | +4.77 | 7.9 100.4 228.7 |
| `kinda` |  | L3, 5–37 | +1.00 | +4.62 | 10.0 91.8 258.1 |
| `赚钱` | to earn money | L39 | +1.00 | +4.47 | 0.7 8.6 26.5 |
| `真的是` | approx. really / correct |  | +1.00 | +3.81 | 1.4 21.5 26.5 |

#### Metric: `weighted`


##### Fading: 68 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `情感` | feeling |  | -1.00 | -5.11 | 16.8 6.2 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.07 | 16.3 3.0 0.0 |
| `lgbtq` |  |  | -1.00 | -4.75 | 12.9 0.6 0.0 |
| `options` |  |  | -1.00 | -4.60 | 11.6 1.3 0.0 |
| `疗` | to treat |  | -1.00 | -4.38 | 9.9 5.2 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.35 | 9.7 2.1 0.0 |
| `询问` | to inquire |  | -1.00 | -4.28 | 11.2 2.5 0.1 |
| `回应` | to respond |  | -1.00 | -3.98 | 12.6 4.0 0.3 |
| `应该如何` | approx. ought to / how |  | -1.00 | -3.92 | 7.1 0.5 0.0 |
| `emotionally` |  |  | -1.00 | -3.84 | 6.7 0.2 0.0 |
| `emotional` |  |  | -1.00 | -3.68 | 5.9 4.8 0.0 |
| `ㅤ` |  |  | -1.00 | -3.65 | 24.4 14.5 1.5 |
| `我觉得` | approx. I / to think… |  | -1.00 | -3.63 | 5.7 0.6 0.0 |
| `or` |  |  | -1.00 | -3.55 | 5.4 0.8 0.0 |
| `others` |  |  | -1.00 | -3.48 | 8.9 3.2 0.3 |

##### Appearing: 49 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +1.00 | +6.13 | 0.2 13.6 47.2 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.03 | 0.7 10.1 78.0 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +5.41 | 0.0 4.3 20.8 |
| `thing` |  | L6–22, 27–37 | +1.00 | +4.80 | 1.4 20.3 51.3 |
| `kinda` |  | L3, 5–37 | +1.00 | +4.60 | 2.2 24.0 64.1 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +4.30 | 0.0 0.2 9.3 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +4.23 | 0.0 0.6 8.9 |
| `geile` |  | L38–45 | +1.00 | +4.19 | 1.8 11.2 42.0 |
| `stuff` |  | L6–36 | +1.00 | +3.97 | 1.3 13.8 28.1 |
| `没人` | approx. not / person |  | +1.00 | +3.81 | 0.1 3.5 8.2 |
| `没什么` | it doesn't matter |  | +1.00 | +3.59 | 0.0 0.8 5.5 |
| `怎么` | how? |  | +1.00 | +3.52 | 1.9 11.7 27.2 |
| `赚钱` | to earn money | L39 | +1.00 | +3.47 | 0.1 1.2 6.6 |
| `guy` |  |  | +1.00 | +3.37 | 0.0 0.8 4.7 |
| `实实在在` | approx. real / really /… |  | +1.00 | +3.31 | 0.0 1.6 4.4 |

### `ideation-immediate-risk` — cynic

167 words tested. Baseline share of the readout: unsteered 4%, α=1.5 19%, α=4 38%.


#### Metric: `cells`


##### Fading: 60 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `心理健康` | approx. psychology / health |  | -1.00 | -6.43 | 42.7 12.2 0.0 |
| `lgbtq` |  |  | -1.00 | -6.40 | 41.9 9.6 0.0 |
| `emotionally` |  |  | -1.00 | -6.11 | 34.0 0.9 0.0 |
| `回应` | to respond |  | -1.00 | -6.11 | 34.0 19.2 0.0 |
| `advice` |  |  | -1.00 | -5.78 | 27.0 16.6 0.0 |
| `people` |  |  | -1.00 | -5.69 | 25.3 21.8 0.0 |
| `情感` | feeling |  | -1.00 | -5.53 | 22.7 2.6 0.0 |
| `options` |  |  | -1.00 | -5.42 | 20.9 10.5 0.0 |
| `ˈ` |  |  | -1.00 | -5.30 | 19.2 2.6 0.0 |
| `emotional` |  |  | -0.87 | -5.17 | 17.4 0.0 0.0 |
| `today` |  |  | -1.00 | -5.17 | 17.4 5.2 0.0 |
| `your` |  |  | -1.00 | -4.86 | 89.8 61.0 2.6 |
| `feelings` |  |  | -1.00 | -4.82 | 62.8 18.3 1.7 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.46 | 29.6 11.3 0.9 |
| `you` |  |  | -1.00 | -4.12 | 99.4 98.5 5.2 |

##### Appearing: 50 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +9.02 | 0.0 16.6 259.8 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +8.38 | 0.0 27.9 166.5 |
| `thing` |  | L6–22, 27–37 | +1.00 | +8.11 | 0.0 42.7 137.8 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.33 | 0.0 15.7 80.2 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.01 | 0.9 56.7 176.1 |
| `靠谱` | reliable |  | +1.00 | +6.88 | 0.0 8.7 58.4 |
| `somehow` |  | L46–55 | +1.00 | +6.87 | 0.9 98.5 159.6 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +6.79 | 0.0 2.6 54.9 |
| `stuff` |  | L6–36 | +1.00 | +6.70 | 0.0 7.8 51.4 |
| `没什么` | it doesn't matter |  | +1.00 | +6.70 | 0.0 13.1 51.4 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +6.57 | 0.0 8.7 47.1 |
| `freaking` |  |  | +1.00 | +5.83 | 0.0 0.9 27.9 |
| `funny` |  | L35–38 | +1.00 | +5.83 | 0.0 1.7 27.9 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +5.74 | 0.0 7.0 26.2 |
| `借钱` | to borrow money |  | +1.00 | +5.53 | 0.0 16.6 22.7 |

#### Metric: `weighted`


##### Fading: 58 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -5.33 | 19.6 2.2 0.0 |
| `your` |  |  | -1.00 | -4.96 | 29.6 14.6 0.5 |
| `回应` | to respond |  | -1.00 | -4.75 | 12.9 3.1 0.0 |
| `心理健康` | approx. psychology / health |  | -1.00 | -4.74 | 12.9 2.2 0.0 |
| `feelings` |  |  | -1.00 | -4.69 | 18.4 4.3 0.2 |
| `you` |  |  | -1.00 | -4.48 | 32.4 24.7 1.0 |
| `emotionally` |  |  | -1.00 | -4.23 | 8.9 0.2 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -4.15 | 10.1 3.9 0.1 |
| `ㅤ` |  |  | -1.00 | -4.13 | 29.9 15.5 1.2 |
| `advice` |  |  | -1.00 | -4.02 | 7.6 3.8 0.0 |
| `options` |  |  | -1.00 | -3.99 | 7.4 1.6 0.0 |
| `情绪` | mood |  | -1.00 | -3.78 | 14.5 1.1 0.6 |
| `情感` | feeling |  | -1.00 | -3.73 | 6.1 0.5 0.0 |
| `ˈ` |  |  | -1.00 | -3.64 | 5.7 0.5 0.0 |
| `emergency` |  |  | -1.00 | -3.38 | 6.5 1.6 0.2 |

##### Appearing: 54 words (25 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.78 | 0.0 2.6 54.4 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.98 | 0.1 11.1 37.2 |
| `somehow` |  | L46–55 | +1.00 | +5.79 | 0.2 17.6 39.1 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.78 | 0.0 7.4 27.0 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +5.73 | 0.0 4.0 26.0 |
| `yeah` |  | L26–32 | +1.00 | +5.25 | 1.2 16.1 63.9 |
| `靠谱` | reliable |  | +1.00 | +4.95 | 0.0 2.5 14.9 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.77 | 0.0 2.6 13.1 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +4.74 | 0.0 0.3 12.9 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +4.52 | 0.0 1.4 10.9 |
| `geile` |  | L38–45 | +1.00 | +4.49 | 1.0 7.5 33.4 |
| `没什么` | it doesn't matter |  | +1.00 | +4.42 | 0.0 3.9 10.2 |
| `怎么` | how? |  | +1.00 | +4.36 | 1.2 8.7 33.8 |
| `everything` |  |  | +1.00 | +4.27 | 0.6 12.5 21.5 |
| `救命` | to save sb's life |  | +1.00 | +4.15 | 0.1 1.1 10.1 |

### `happy-promotion` — cynic

192 words tested. Baseline share of the readout: unsteered 8%, α=1.5 23%, α=4 45%.


#### Metric: `cells`


##### Fading: 79 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `thanks` |  |  | -1.00 | -6.23 | 37.1 4.0 0.0 |
| `回应` | to respond |  | -1.00 | -5.96 | 30.6 8.1 0.0 |
| `新的` | approx. surname Xin / of |  | -1.00 | -5.72 | 25.8 11.3 0.0 |
| `hopes` |  |  | -1.00 | -5.58 | 23.4 8.1 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.42 | 21.0 4.0 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.42 | 21.0 5.7 0.0 |
| `colorful` |  |  | -0.87 | -5.19 | 17.7 0.0 0.0 |
| `荣耀` | honor |  | -1.00 | -5.19 | 17.7 13.7 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.19 | 17.7 1.6 0.0 |
| `新篇章` | approx. surname Xin / chapter |  | -1.00 | -5.12 | 16.9 10.5 0.0 |
| `thank` |  |  | -1.00 | -5.08 | 43.5 12.9 0.8 |
| `newfound` |  |  | -1.00 | -5.06 | 16.1 9.7 0.0 |
| `suggestions` |  |  | -1.00 | -5.06 | 16.1 7.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.06 | 16.1 1.6 0.0 |
| `s` |  |  | -1.00 | -4.29 | 25.0 4.8 0.8 |

##### Appearing: 47 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.42 | 0.0 7.3 171.0 |
| `funny` |  | L35–38 | +1.00 | +7.04 | 0.0 9.7 65.3 |
| `没人` | approx. not / person |  | +1.00 | +6.66 | 0.0 10.5 50.0 |
| `freaking` |  |  | +1.00 | +6.41 | 0.0 0.8 41.9 |
| `thing` |  | L6–22, 27–37 | +1.00 | +6.32 | 1.6 40.3 167.7 |
| `没什么` | it doesn't matter |  | +1.00 | +6.29 | 0.0 4.8 38.7 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +6.05 | 0.8 12.1 86.3 |
| `噱` | loud laughter | L38–45 | +0.87 | +5.67 | 0.0 0.0 25.0 |
| `拼命` | to do one's utmost |  | +1.00 | +5.58 | 0.0 3.2 23.4 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +5.53 | 0.0 3.2 22.6 |
| `stuff` |  | L6–36 | +1.00 | +5.48 | 3.2 66.1 166.1 |
| `靠谱` | reliable |  | +1.00 | +5.47 | 0.8 19.4 57.3 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +5.45 | 0.8 4.8 56.5 |
| `哪怕` | even |  | +1.00 | +5.42 | 0.0 12.1 21.0 |
| `everytime` |  |  | +1.00 | +5.37 | 0.0 5.7 20.2 |

#### Metric: `weighted`


##### Fading: 77 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -4.61 | 39.5 15.7 1.1 |
| `thank` |  |  | -1.00 | -4.43 | 12.1 2.2 0.1 |
| `回应` | to respond |  | -1.00 | -4.19 | 8.6 1.3 0.0 |
| `讨论` | to discuss |  | -1.00 | -4.05 | 7.8 0.4 0.0 |
| `新的` | approx. surname Xin / of |  | -1.00 | -4.00 | 7.5 1.8 0.0 |
| `ㅤ` |  |  | -1.00 | -3.97 | 23.7 14.0 1.1 |
| `recently` |  |  | -1.00 | -3.90 | 18.6 6.8 0.8 |
| `s` |  |  | -1.00 | -3.90 | 8.9 2.5 0.1 |
| `2` |  |  | -1.00 | -3.77 | 12.4 9.8 0.5 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -3.75 | 6.2 1.2 0.0 |
| `thanks` |  |  | -1.00 | -3.69 | 6.0 0.6 0.0 |
| `1` |  |  | -1.00 | -3.69 | 12.2 8.1 0.5 |
| `colorful` |  |  | -0.87 | -3.65 | 5.8 0.0 0.0 |
| `ˈ` |  |  | -1.00 | -3.63 | 5.7 0.3 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -3.56 | 5.4 1.2 0.0 |

##### Appearing: 46 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +5.90 | 0.0 0.9 29.4 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.45 | 0.2 8.0 29.6 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.00 | 1.4 16.0 62.1 |
| `everything` |  |  | +1.00 | +4.78 | 1.7 32.1 60.7 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +4.66 | 0.1 1.5 14.2 |
| `没什么` | it doesn't matter |  | +1.00 | +4.61 | 0.0 1.1 11.7 |
| `stuff` |  | L6–36 | +1.00 | +4.53 | 0.8 11.1 30.7 |
| `赚钱` | to earn money | L39 | +1.00 | +4.50 | 0.3 8.2 17.1 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.49 | 0.1 1.7 12.5 |
| `yeah` |  | L26–32 | +1.00 | +4.46 | 2.6 22.3 68.8 |
| `没人` | approx. not / person |  | +1.00 | +4.39 | 0.0 3.2 10.0 |
| `everytime` |  |  | +1.00 | +4.34 | 0.0 1.9 9.6 |
| `funny` |  | L35–38 | +1.00 | +4.33 | 0.0 1.3 9.6 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +4.14 | 1.8 13.4 40.3 |
| `哪怕` | even |  | +1.00 | +3.93 | 0.0 3.3 7.1 |

### `happy-friendship` — cynic

166 words tested. Baseline share of the readout: unsteered 6%, α=1.5 24%, α=4 43%.


#### Metric: `cells`


##### Fading: 72 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `guests` |  |  | -1.00 | -6.51 | 45.0 29.3 0.0 |
| `thank` |  |  | -1.00 | -5.85 | 28.4 12.7 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.70 | 25.4 4.9 0.0 |
| `beautiful` |  |  | -1.00 | -5.64 | 24.4 18.6 0.0 |
| `recently` |  |  | -1.00 | -5.58 | 23.5 6.8 0.0 |
| `prank` |  |  | -1.00 | -5.52 | 22.5 7.8 0.0 |
| `ˈ` |  |  | -1.00 | -5.46 | 21.5 2.0 0.0 |
| `advice` |  |  | -1.00 | -5.33 | 19.6 12.7 0.0 |
| `came` |  |  | -1.00 | -5.33 | 19.6 10.8 0.0 |
| `grateful` |  |  | -1.00 | -5.33 | 19.6 10.8 0.0 |
| `joyful` |  |  | -1.00 | -5.33 | 19.6 4.9 0.0 |
| `christmas` |  |  | -1.00 | -4.47 | 32.3 7.8 1.0 |
| `s` |  |  | -1.00 | -4.29 | 28.4 7.8 1.0 |
| `回应` | to respond |  | -1.00 | -4.24 | 27.4 14.7 1.0 |
| `1` |  |  | -1.00 | -4.01 | 39.1 22.5 2.0 |

##### Appearing: 47 words (28 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `everything` |  |  | +1.00 | +8.00 | 0.0 39.1 127.1 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.83 | 0.0 18.6 113.4 |
| `everybody` |  |  | +1.00 | +7.72 | 0.0 14.7 104.6 |
| `somehow` |  | L46–55 | +1.00 | +7.30 | 0.0 29.3 78.2 |
| `freaking` |  |  | +1.00 | +6.79 | 0.0 5.9 54.7 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.69 | 2.0 17.6 253.2 |
| `没人` | approx. not / person |  | +1.00 | +6.16 | 0.0 10.8 35.2 |
| `靠谱` | reliable |  | +1.00 | +6.08 | 0.0 3.9 33.2 |
| `了好多` | approx. (completed action… / many |  | +1.00 | +6.03 | 0.0 21.5 32.3 |
| `yeah` |  | L26–32 | +1.00 | +5.83 | 3.9 97.8 250.2 |
| `gonna` |  | L0–3, 12–17 | +0.87 | +5.75 | 0.0 0.0 26.4 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +5.75 | 0.0 0.0 26.4 |
| `everytime` |  |  | +1.00 | +5.70 | 0.0 4.9 25.4 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.57 | 3.9 67.5 208.2 |
| `nobody` |  |  | +0.87 | +5.46 | 0.0 0.0 21.5 |

#### Metric: `weighted`


##### Fading: 74 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `guests` |  |  | -1.00 | -4.93 | 14.8 8.9 0.0 |
| `ㅤ` |  |  | -1.00 | -4.35 | 35.1 17.3 1.2 |
| `recently` |  |  | -1.00 | -4.14 | 8.3 2.0 0.0 |
| `s` |  |  | -1.00 | -4.12 | 11.0 3.6 0.2 |
| `2` |  |  | -1.00 | -4.05 | 22.5 13.6 0.9 |
| `ˈ` |  |  | -1.00 | -4.01 | 7.6 0.3 0.0 |
| `prank` |  |  | -1.00 | -3.98 | 7.4 2.5 0.0 |
| `thank` |  |  | -1.00 | -3.92 | 7.1 2.1 0.0 |
| `advice` |  |  | -1.00 | -3.92 | 7.0 2.5 0.0 |
| `beautiful` |  |  | -1.00 | -3.89 | 6.9 3.0 0.0 |
| `1` |  |  | -1.00 | -3.87 | 22.6 12.0 1.1 |
| `the` |  |  | -1.00 | -3.81 | 15.2 2.6 0.6 |
| `your` |  |  | -1.00 | -3.77 | 36.4 17.2 2.2 |
| `came` |  |  | -1.00 | -3.73 | 6.1 1.5 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -3.71 | 6.0 0.6 0.0 |

##### Appearing: 44 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +5.96 | 0.3 2.7 51.4 |
| `everything` |  |  | +1.00 | +5.59 | 0.0 7.7 23.6 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.58 | 0.5 14.8 47.9 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +5.29 | 0.0 2.5 19.1 |
| `everybody` |  |  | +1.00 | +5.16 | 0.0 1.7 17.4 |
| `somehow` |  | L46–55 | +1.00 | +4.96 | 0.0 4.9 15.1 |
| `yeah` |  | L26–32 | +1.00 | +4.96 | 1.7 21.9 66.4 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +4.80 | 0.7 10.9 32.4 |
| `thing` |  | L6–22, 27–37 | +1.00 | +4.52 | 1.1 17.5 36.0 |
| `freaking` |  |  | +1.00 | +4.40 | 0.0 1.4 10.1 |
| `kids` |  | L1, 3–5, 8 | +1.00 | +4.37 | 1.4 18.1 38.0 |
| `了好多` | approx. (completed action… / many |  | +1.00 | +4.36 | 0.0 7.6 9.8 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +4.35 | 10.7 112.7 226.9 |
| `没人` | approx. not / person |  | +1.00 | +3.93 | 0.0 3.0 7.1 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +3.87 | 0.0 0.0 6.8 |

### `instructions-cake` — cynic

168 words tested. Baseline share of the readout: unsteered 6%, α=1.5 23%, α=4 43%.


#### Metric: `cells`


##### Fading: 60 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -6.59 | 47.5 9.3 0.0 |
| `family` |  |  | -1.00 | -5.70 | 25.5 7.6 0.0 |
| `even` |  |  | -1.00 | -5.44 | 21.2 2.5 0.0 |
| `指导` | to guide |  | -1.00 | -5.44 | 21.2 9.3 0.0 |
| `helpful` |  |  | -0.87 | -5.20 | 17.8 0.0 0.0 |
| `music` |  |  | -0.87 | -5.13 | 17.0 0.0 0.0 |
| `teaching` |  |  | -1.00 | -4.74 | 35.6 3.4 0.8 |
| `准备工作` | approx. preparation / to work |  | -1.00 | -4.22 | 24.6 12.7 0.8 |
| `pastry` |  |  | -1.00 | -4.17 | 23.8 15.3 0.8 |
| `帮我` | approx. to help / I |  | -1.00 | -3.90 | 32.3 6.8 1.7 |
| `christmas` |  |  | -1.00 | -3.83 | 18.7 10.2 0.8 |
| `甜` | sweet |  | -1.00 | -3.83 | 18.7 14.4 0.8 |
| `the` |  |  | -1.00 | -3.70 | 28.0 4.2 1.7 |
| `cookbook` |  |  | -1.00 | -3.46 | 23.8 17.0 1.7 |
| `diy` |  |  | -1.00 | -3.31 | 29.7 11.0 2.5 |

##### Appearing: 45 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 5–37 | +1.00 | +8.89 | 0.0 41.6 236.0 |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.79 | 0.0 10.2 221.6 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +8.50 | 0.0 11.9 180.8 |
| `thing` |  | L6–22, 27–37 | +1.00 | +7.68 | 0.0 24.6 101.9 |
| `funny` |  | L35–38 | +1.00 | +7.04 | 0.0 7.6 65.4 |
| `everything` |  |  | +1.00 | +7.00 | 0.8 108.7 172.3 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +6.88 | 0.0 11.0 58.6 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +6.59 | 0.0 11.0 47.5 |
| `freaking` |  |  | +1.00 | +6.42 | 0.0 0.8 42.4 |
| `everyone` |  |  | +1.00 | +6.24 | 0.0 31.4 37.4 |
| `没什么` | it doesn't matter |  | +1.00 | +6.21 | 0.0 8.5 36.5 |
| `没人` | approx. not / person |  | +1.00 | +6.18 | 0.0 12.7 35.6 |
| `yeah` |  | L26–32 | +1.00 | +6.09 | 2.5 54.3 207.1 |
| `gonna` |  | L0–3, 12–17 | +1.00 | +6.00 | 0.0 13.6 31.4 |
| `somehow` |  | L46–55 | +1.00 | +5.96 | 0.0 28.0 30.6 |

#### Metric: `weighted`


##### Fading: 62 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `training` |  |  | -1.00 | -5.20 | 17.9 1.3 0.0 |
| `准备工作` | approx. preparation / to work |  | -1.00 | -4.14 | 9.8 3.6 0.1 |
| `help` |  |  | -1.00 | -4.10 | 35.5 14.8 1.6 |
| `family` |  |  | -1.00 | -3.95 | 7.2 1.2 0.0 |
| `ㅤ` |  |  | -1.00 | -3.80 | 27.6 15.5 1.5 |
| `the` |  |  | -1.00 | -3.69 | 8.4 0.8 0.2 |
| `1` |  |  | -1.00 | -3.63 | 12.0 6.8 0.5 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -3.62 | 11.7 8.3 0.5 |
| `learning` |  |  | -1.00 | -3.61 | 51.5 21.9 3.8 |
| `teaching` |  |  | -1.00 | -3.59 | 6.6 0.5 0.1 |
| `for` |  |  | -1.00 | -3.54 | 12.6 4.3 0.6 |
| `甜` | sweet |  | -1.00 | -3.52 | 6.5 4.7 0.1 |
| `christmas` |  |  | -1.00 | -3.31 | 5.5 2.5 0.1 |
| `your` |  |  | -1.00 | -3.31 | 17.9 9.7 1.4 |
| `帮我` | approx. to help / I |  | -1.00 | -3.29 | 6.9 1.1 0.3 |

##### Appearing: 44 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.59 | 0.0 1.4 47.8 |
| `kinda` |  | L3, 5–37 | +1.00 | +6.51 | 0.0 6.5 45.2 |
| `everything` |  |  | +1.00 | +6.44 | 0.1 36.1 50.2 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +5.85 | 0.0 1.6 28.3 |
| `yeah` |  | L26–32 | +1.00 | +5.78 | 0.4 10.2 47.8 |
| `stuff` |  | L6–36 | +1.00 | +5.65 | 0.5 23.4 47.4 |
| `靠谱` | reliable |  | +1.00 | +5.07 | 0.1 10.6 19.9 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.06 | 0.0 4.0 16.2 |
| `老百姓` | ordinary people |  | +1.00 | +4.70 | 0.1 8.7 15.7 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +4.46 | 0.0 1.5 10.5 |
| `没什么` | it doesn't matter |  | +1.00 | +4.44 | 0.0 4.9 10.4 |
| `geile` |  | L38–45 | +1.00 | +4.35 | 1.1 7.5 32.6 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.23 | 0.0 1.7 8.9 |
| `funny` |  | L35–38 | +1.00 | +4.18 | 0.0 1.0 8.6 |
| `everyone` |  |  | +1.00 | +4.16 | 0.0 5.9 8.4 |

### `instructions-directions` — cynic

464 words tested. Baseline share of the readout: unsteered 1%, α=1.5 8%, α=4 32%.


#### Metric: `cells`


##### Fading: 188 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `locations` |  |  | -1.00 | -5.69 | 25.3 1.7 0.0 |
| `nearby` |  |  | -1.00 | -5.57 | 23.3 0.7 0.0 |
| `市区` | urban district |  | -1.00 | -5.49 | 21.9 9.3 0.0 |
| `respectively` |  |  | -1.00 | -5.45 | 21.3 5.3 0.0 |
| `s` |  |  | -1.00 | -5.14 | 28.9 11.6 0.3 |
| `as` |  |  | -1.00 | -5.07 | 16.3 2.0 0.0 |
| `toward` |  |  | -1.00 | -4.92 | 24.6 5.0 0.3 |
| `散步` | to take a walk |  | -1.00 | -4.86 | 14.0 3.7 0.0 |
| `cities` |  |  | -1.00 | -4.80 | 31.9 14.3 0.7 |
| `أنح` |  |  | -1.00 | -4.75 | 13.0 6.7 0.0 |
| `允许` | to permit |  | -1.00 | -4.72 | 12.6 5.7 0.0 |
| `城镇` | town |  | -1.00 | -4.72 | 12.6 0.3 0.0 |
| `councill` |  |  | -1.00 | -4.62 | 19.9 12.0 0.3 |
| `地区的` | approx. region / of |  | -1.00 | -4.56 | 11.3 0.3 0.0 |
| `city` |  |  | -1.00 | -4.55 | 104.8 70.5 4.0 |

##### Appearing: 120 words (37 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.72 | 0.0 9.6 210.2 |
| `everything` |  |  | +1.00 | +8.41 | 0.0 77.8 169.6 |
| `thing` |  | L6–22, 27–37 | +1.00 | +8.12 | 0.0 24.6 138.3 |
| `got` |  | L0–23, 25 | +1.00 | +8.09 | 0.0 49.2 136.0 |
| `yeah` |  | L26–32 | +1.00 | +7.89 | 0.3 32.9 197.2 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.87 | 0.0 15.6 116.7 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +7.86 | 0.0 3.3 115.7 |
| `stuff` |  | L6–36 | +1.00 | +7.23 | 0.0 18.6 74.5 |
| `没人` | approx. not / person |  | +1.00 | +7.00 | 0.0 5.3 63.5 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +6.88 | 0.0 10.3 58.5 |
| `big` |  | L0–2, 4–14, 16–24 | +1.00 | +6.88 | 0.0 25.9 58.2 |
| `统统` | totally |  | +1.00 | +6.85 | 0.0 8.7 57.2 |
| `gotten` |  | L0–7, 9–11 | +1.00 | +6.72 | 0.0 2.7 52.2 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +6.72 | 0.0 3.0 52.2 |
| `买房` | to buy a house |  | +1.00 | +6.59 | 0.0 16.3 47.6 |

#### Metric: `weighted`


##### Fading: 181 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `neighborhoods` |  |  | -1.00 | -4.63 | 30.9 12.0 0.8 |
| `s` |  |  | -1.00 | -4.62 | 13.1 4.2 0.1 |
| `respectively` |  |  | -1.00 | -4.47 | 10.6 2.2 0.0 |
| `each` |  |  | -1.00 | -4.38 | 28.2 12.9 0.9 |
| `city` |  |  | -1.00 | -3.98 | 35.6 22.8 1.8 |
| `councill` |  |  | -1.00 | -3.97 | 7.9 2.2 0.0 |
| `cities` |  |  | -1.00 | -3.89 | 8.0 3.2 0.1 |
| `ㅤ` |  |  | -1.00 | -3.86 | 13.0 6.5 0.4 |
| `nearby` |  |  | -1.00 | -3.78 | 6.4 0.1 0.0 |
| `locations` |  |  | -1.00 | -3.58 | 5.5 0.2 0.0 |
| `市区` | urban district |  | -1.00 | -3.53 | 5.3 1.7 0.0 |
| `沿途` | by the wayside |  | -1.00 | -3.53 | 9.2 4.0 0.3 |
| `中心城区` | approx. center / Cheng District… |  | -1.00 | -3.52 | 8.4 4.6 0.3 |
| `toward` |  |  | -1.00 | -3.43 | 5.4 0.8 0.1 |
| `杉` | China fir |  | -1.00 | -3.42 | 5.2 0.5 0.0 |

##### Appearing: 116 words (36 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.83 | 0.0 1.7 56.2 |
| `yeah` |  | L26–32 | +1.00 | +6.73 | 0.0 11.0 56.2 |
| `everything` |  |  | +1.00 | +6.52 | 0.0 27.0 45.5 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +6.42 | 1.8 57.1 194.2 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +6.37 | 0.0 0.6 40.8 |
| `kinda` |  | L3, 5–37 | +1.00 | +6.18 | 0.0 3.7 35.8 |
| `got` |  | L0–23, 25 | +1.00 | +6.07 | 0.0 17.1 33.0 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.90 | 0.0 6.6 29.4 |
| `geile` |  | L38–45 | +1.00 | +5.79 | 0.5 7.5 52.5 |
| `really` |  | L0–38, 46–48 | +1.00 | +5.77 | 1.9 40.9 128.0 |
| `统统` | totally |  | +1.00 | +5.17 | 0.0 2.3 17.5 |
| `买房` | to buy a house |  | +1.00 | +5.00 | 0.0 5.5 15.4 |
| `stuff` |  | L6–36 | +1.00 | +4.91 | 0.0 3.3 14.5 |
| `没人` | approx. not / person |  | +1.00 | +4.91 | 0.0 1.4 14.5 |
| `一大堆` | approx. one / big /… | L37 | +1.00 | +4.83 | 0.0 1.3 14.6 |

### `self-reflection-identity` — cynic

153 words tested. Baseline share of the readout: unsteered 2%, α=1.5 16%, α=4 37%.


#### Metric: `cells`


##### Fading: 56 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -6.75 | 53.4 4.5 0.0 |
| `on` |  |  | -1.00 | -5.88 | 28.9 3.3 0.0 |
| `ˈ` |  |  | -1.00 | -5.88 | 28.9 3.3 0.0 |
| `i` |  |  | -1.00 | -5.76 | 26.7 11.1 0.0 |
| `什么样的` | approx. what kind? / of |  | -1.00 | -5.76 | 26.7 8.9 0.0 |
| `knowledge` |  |  | -1.00 | -5.64 | 24.5 12.2 0.0 |
| `这个人` | approx. (pronoun) this / person |  | -1.00 | -5.64 | 24.5 16.7 0.0 |
| `alibaba` |  |  | -1.00 | -5.51 | 22.2 2.2 0.0 |
| `and` |  |  | -1.00 | -4.53 | 36.7 4.5 1.1 |
| `human` |  |  | -1.00 | -4.25 | 51.2 27.8 2.2 |
| `s` |  |  | -1.00 | -4.24 | 30.0 21.1 1.1 |
| `or` |  |  | -1.00 | -4.02 | 25.6 3.3 1.1 |
| `that` |  |  | -1.00 | -4.02 | 25.6 5.6 1.1 |
| `erotik` |  |  | -1.00 | -3.90 | 56.7 45.6 3.3 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -3.38 | 27.8 24.5 2.2 |

##### Appearing: 55 words (26 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `kinda` |  | L3, 5–37 | +1.00 | +8.99 | 0.0 74.5 253.6 |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.65 | 0.0 1.1 200.2 |
| `thing` |  | L6–22, 27–37 | +1.00 | +8.10 | 0.0 26.7 136.8 |
| `everything` |  |  | +1.00 | +8.02 | 0.0 64.5 129.0 |
| `yeah` |  | L26–32 | +1.00 | +7.68 | 1.1 123.5 329.2 |
| `guy` |  |  | +1.00 | +7.59 | 0.0 23.4 95.7 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +7.48 | 0.0 2.2 89.0 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.47 | 0.0 13.3 87.9 |
| `stuff` |  | L6–36 | +1.00 | +7.12 | 0.0 20.0 69.0 |
| `somehow` |  | L46–55 | +1.00 | +7.05 | 0.0 27.8 65.6 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +6.97 | 0.0 11.1 62.3 |
| `freaking` |  |  | +1.00 | +6.97 | 0.0 1.1 62.3 |
| `got` |  | L0–23, 25 | +1.00 | +6.84 | 0.0 20.0 56.7 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +6.69 | 1.1 43.4 165.7 |
| `没人` | approx. not / person |  | +1.00 | +6.53 | 0.0 11.1 45.6 |

#### Metric: `weighted`


##### Fading: 50 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `lgbtq` |  |  | -1.00 | -5.05 | 16.1 0.8 0.0 |
| `s` |  |  | -1.00 | -5.02 | 21.7 9.1 0.2 |
| `ˈ` |  |  | -1.00 | -4.77 | 13.2 1.6 0.0 |
| `erotik` |  |  | -1.00 | -4.47 | 26.2 14.4 0.7 |
| `human` |  |  | -1.00 | -4.25 | 14.4 5.2 0.3 |
| `唿` | to whistle (with fingers in… |  | -1.00 | -4.19 | 14.0 6.8 0.3 |
| `i` |  |  | -1.00 | -4.11 | 8.1 2.8 0.0 |
| `ㅤ` |  |  | -1.00 | -4.10 | 32.1 21.6 1.4 |
| `capabilities` |  |  | -1.00 | -4.01 | 20.0 15.4 0.8 |
| `on` |  |  | -1.00 | -3.96 | 7.3 0.4 0.0 |
| `and` |  |  | -1.00 | -3.62 | 10.2 0.9 0.4 |
| `这个人` | approx. (pronoun) this / person |  | -1.00 | -3.57 | 5.5 3.3 0.0 |
| `什么样的` | approx. what kind? / of |  | -1.00 | -3.55 | 5.3 1.7 0.0 |
| `humans` |  |  | -1.00 | -3.42 | 42.9 30.6 3.6 |
| `or` |  |  | -1.00 | -3.36 | 6.3 0.4 0.2 |

##### Appearing: 50 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +1.00 | +7.29 | 0.1 33.8 99.7 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.01 | 0.0 19.3 63.9 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +6.35 | 2.1 135.4 209.3 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.15 | 0.0 0.1 35.0 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +5.97 | 0.1 8.0 39.6 |
| `everything` |  |  | +1.00 | +5.85 | 0.0 16.6 28.3 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.60 | 0.0 4.0 23.8 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +5.25 | 0.0 2.6 18.6 |
| `guy` |  |  | +1.00 | +5.07 | 0.0 6.4 16.3 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.80 | 0.0 2.1 13.4 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +4.80 | 0.0 0.2 13.4 |
| `really` |  | L0–38, 46–48 | +1.00 | +4.77 | 7.4 148.4 215.6 |
| `geile` |  | L38–45 | +1.00 | +4.63 | 1.4 8.6 46.9 |
| `stuff` |  | L6–36 | +1.00 | +4.57 | 0.0 4.0 11.4 |
| `somehow` |  | L46–55 | +1.00 | +4.54 | 0.0 5.3 11.1 |

### `self-reflection-values` — cynic

141 words tested. Baseline share of the readout: unsteered 2%, α=1.5 12%, α=4 32%.


#### Metric: `cells`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decisions` |  |  | -1.00 | -7.10 | 68.0 20.7 0.0 |
| `regarding` |  |  | -1.00 | -6.03 | 32.3 9.2 0.0 |
| `ˈ` |  |  | -1.00 | -5.98 | 31.1 4.6 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -5.98 | 31.1 9.2 0.0 |
| `priorit` |  |  | -1.00 | -5.93 | 29.9 9.2 0.0 |
| `lgbtq` |  |  | -1.00 | -5.75 | 26.5 3.5 0.0 |
| `价值观` | value system |  | -1.00 | -5.75 | 26.5 11.5 0.0 |
| `decision` |  |  | -1.00 | -5.69 | 25.4 6.9 0.0 |
| `situations` |  |  | -1.00 | -5.69 | 25.4 9.2 0.0 |
| `影响力` | influence |  | -1.00 | -5.69 | 25.4 2.3 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.63 | 24.2 6.9 0.0 |
| `choices` |  |  | -1.00 | -4.95 | 86.4 35.7 2.3 |
| `s` |  |  | -1.00 | -4.78 | 44.9 20.7 1.1 |
| `whom` |  |  | -1.00 | -4.26 | 31.1 21.9 1.1 |
| `a` |  |  | -1.00 | -4.03 | 26.5 8.1 1.1 |

##### Appearing: 53 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +1.00 | +9.32 | 0.0 104.8 319.1 |
| `everything` |  |  | +1.00 | +8.79 | 0.0 103.7 220.1 |
| `kinda` |  | L3, 5–37 | +1.00 | +8.47 | 0.0 27.6 177.4 |
| `thing` |  | L6–22, 27–37 | +1.00 | +8.42 | 0.0 39.2 170.5 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +8.33 | 0.0 28.8 160.1 |
| `messed` |  | L16–39, 42–44 | +1.00 | +7.84 | 0.0 1.1 114.1 |
| `stuff` |  | L6–36 | +1.00 | +7.42 | 0.0 16.1 85.2 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.34 | 0.0 15.0 80.7 |
| `big` |  | L0–2, 4–14, 16–24 | +1.00 | +7.21 | 0.0 35.7 73.7 |
| `freaking` |  |  | +1.00 | +7.12 | 0.0 2.3 69.1 |
| `kids` |  | L1, 3–5, 8 | +1.00 | +6.97 | 0.0 27.6 62.2 |
| `got` |  | L0–23, 25 | +1.00 | +6.89 | 0.0 13.8 58.8 |
| `没人` | approx. not / person |  | +1.00 | +6.89 | 0.0 16.1 58.8 |
| `靠谱` | reliable |  | +1.00 | +6.80 | 0.0 9.2 55.3 |
| `crazy` |  | L15, 23–24, 29–34 | +0.87 | +6.77 | 0.0 0.0 54.1 |

#### Metric: `weighted`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `decisions` |  |  | -1.00 | -6.01 | 31.7 5.2 0.0 |
| `choices` |  |  | -1.00 | -5.45 | 31.9 16.0 0.2 |
| `s` |  |  | -1.00 | -5.16 | 24.3 9.6 0.2 |
| `the` |  |  | -1.00 | -5.06 | 28.1 5.8 0.4 |
| `价值观` | value system |  | -1.00 | -4.76 | 13.1 2.0 0.0 |
| `讨论` | to discuss |  | -1.00 | -4.67 | 12.3 2.9 0.0 |
| `lgbtq` |  |  | -1.00 | -4.46 | 10.5 0.6 0.0 |
| `priorit` |  |  | -1.00 | -4.32 | 9.5 2.6 0.0 |
| `ˈ` |  |  | -1.00 | -4.25 | 9.0 2.0 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -4.24 | 8.9 1.9 0.0 |
| `情感` | feeling |  | -1.00 | -4.18 | 13.8 3.5 0.3 |
| `situations` |  |  | -1.00 | -4.04 | 7.8 1.7 0.0 |
| `影响力` | influence |  | -1.00 | -4.01 | 7.5 0.2 0.0 |
| `responses` |  |  | -1.00 | -3.92 | 24.0 4.9 1.1 |
| `2` |  |  | -1.00 | -3.88 | 18.0 16.8 0.8 |

##### Appearing: 48 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `really` |  | L0–38, 46–48 | +1.00 | +7.73 | 0.7 89.9 260.4 |
| `yeah` |  | L26–32 | +1.00 | +7.71 | 0.0 31.6 103.8 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +7.26 | 0.7 69.2 175.7 |
| `everything` |  |  | +1.00 | +6.76 | 0.0 26.0 53.8 |
| `kinda` |  | L3, 5–37 | +1.00 | +6.39 | 0.0 10.2 41.6 |
| `thing` |  | L6–22, 27–37 | +1.00 | +6.30 | 0.0 7.9 39.0 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +6.09 | 0.0 3.8 33.5 |
| `messed` |  | L16–39, 42–44 | +1.00 | +5.52 | 0.0 0.1 22.4 |
| `kids` |  | L1, 3–5, 8 | +1.00 | +5.43 | 0.0 6.8 21.1 |
| `没人` | approx. not / person |  | +1.00 | +5.00 | 0.0 3.7 15.5 |
| `stuff` |  | L6–36 | +1.00 | +4.90 | 0.0 5.4 14.4 |
| `暴力` | violence |  | +1.00 | +4.87 | 0.0 4.0 14.1 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.73 | 0.0 2.8 12.8 |
| `big` |  | L0–2, 4–14, 16–24 | +1.00 | +4.71 | 0.0 8.6 12.6 |
| `got` |  | L0–23, 25 | +1.00 | +4.43 | 0.0 1.9 10.3 |

### `moral-trolley` — cynic

324 words tested. Baseline share of the readout: unsteered 2%, α=1.5 10%, α=4 29%.


#### Metric: `cells`


##### Fading: 103 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `injured` |  |  | -1.00 | -5.37 | 20.2 10.1 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.18 | 17.6 2.0 0.0 |
| `ещё` |  |  | -1.00 | -4.92 | 14.6 7.6 0.0 |
| `deliberately` |  |  | -1.00 | -4.87 | 14.1 2.5 0.0 |
| `凶手` | murderer |  | -1.00 | -4.87 | 14.1 0.5 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -4.82 | 13.6 12.1 0.0 |
| `逃生` | to flee for one's life |  | -1.00 | -4.82 | 13.6 5.0 0.0 |
| `â` |  |  | -1.00 | -4.77 | 13.1 5.5 0.0 |
| `ём` |  |  | -1.00 | -4.77 | 13.1 5.0 0.0 |
| `车辆` | vehicle |  | -1.00 | -4.71 | 12.6 2.5 0.0 |
| `primarily` |  |  | -0.87 | -4.54 | 11.1 0.0 0.0 |
| `reasons` |  |  | -1.00 | -4.54 | 11.1 6.5 0.0 |
| `应该如何` | approx. ought to / how |  | -1.00 | -4.54 | 11.1 0.5 0.0 |
| `bystand` |  |  | -1.00 | -4.47 | 10.6 9.6 0.0 |
| `arguably` |  |  | -1.00 | -4.40 | 10.1 4.5 0.0 |

##### Appearing: 87 words (34 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.22 | 0.0 5.5 148.2 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +8.17 | 0.0 21.7 143.7 |
| `nobody` |  |  | +1.00 | +8.15 | 0.0 26.2 141.6 |
| `everything` |  |  | +1.00 | +8.05 | 0.0 69.6 132.1 |
| `somehow` |  | L46–55 | +1.00 | +7.95 | 0.0 33.8 123.0 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.60 | 0.0 8.6 96.3 |
| `没人` | approx. not / person |  | +1.00 | +7.58 | 0.0 13.1 95.3 |
| `freaking` |  |  | +1.00 | +7.32 | 0.0 9.6 79.6 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +7.03 | 0.0 8.1 65.0 |
| `拼命` | to do one's utmost |  | +1.00 | +6.98 | 0.0 19.1 62.5 |
| `yeah` |  | L26–32 | +1.00 | +6.93 | 1.0 62.0 183.5 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +6.87 | 0.0 17.1 58.0 |
| `stuff` |  | L6–36 | +1.00 | +6.75 | 0.0 13.1 53.4 |
| `stupid` |  | L24–38, 42–48, 50–51 | +1.00 | +6.55 | 0.0 1.0 46.4 |
| `kids` |  | L1, 3–5, 8 | +1.00 | +6.24 | 0.0 6.0 37.3 |

#### Metric: `weighted`


##### Fading: 100 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `车厢` | carriage |  | -1.00 | -4.06 | 12.7 2.0 0.3 |
| `ещё` |  |  | -1.00 | -3.98 | 7.4 4.4 0.0 |
| `讨论` | to discuss |  | -1.00 | -3.91 | 7.0 0.3 0.0 |
| `decision` |  |  | -1.00 | -3.91 | 12.0 3.9 0.3 |
| `injured` |  |  | -1.00 | -3.90 | 7.0 3.6 0.0 |
| `你觉得` | approx. you (informal,… / to… |  | -1.00 | -3.82 | 6.6 5.4 0.0 |
| `question` |  |  | -1.00 | -3.80 | 13.0 4.3 0.5 |
| `railway` |  |  | -1.00 | -3.80 | 11.5 5.5 0.4 |
| `逃生` | to flee for one's life |  | -1.00 | -3.79 | 6.4 1.7 0.0 |
| `ㅤ` |  |  | -1.00 | -3.71 | 14.7 9.8 0.7 |
| `when` |  |  | -1.00 | -3.71 | 9.3 4.3 0.2 |
| `â` |  |  | -1.00 | -3.69 | 6.0 2.0 0.0 |
| `应该如何` | approx. ought to / how |  | -1.00 | -3.66 | 5.8 0.1 0.0 |
| `deaths` |  |  | -1.00 | -3.58 | 8.7 4.5 0.3 |
| `人死亡` | approx. person / to die |  | -1.00 | -3.37 | 15.9 5.3 1.1 |

##### Appearing: 85 words (33 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +6.80 | 0.0 6.6 55.2 |
| `really` |  | L0–38, 46–48 | +1.00 | +6.76 | 0.8 45.1 135.6 |
| `yeah` |  | L26–32 | +1.00 | +6.58 | 0.1 20.4 56.8 |
| `everything` |  |  | +1.00 | +6.22 | 0.0 24.0 36.7 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.09 | 0.0 1.5 33.6 |
| `nobody` |  |  | +1.00 | +6.01 | 0.0 4.8 31.7 |
| `somehow` |  | L46–55 | +1.00 | +5.83 | 0.0 6.7 27.9 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +5.56 | 0.0 5.0 23.1 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +5.40 | 0.0 1.6 20.6 |
| `没人` | approx. not / person |  | +1.00 | +5.34 | 0.0 2.9 19.8 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +5.20 | 0.2 11.7 25.4 |
| `geile` |  | L38–45 | +1.00 | +5.14 | 1.3 15.3 62.4 |
| `worse` |  |  | +1.00 | +4.97 | 0.1 8.4 17.4 |
| `拼命` | to do one's utmost |  | +1.00 | +4.93 | 0.0 6.5 14.8 |
| `freaking` |  |  | +1.00 | +4.90 | 0.0 1.6 14.4 |

### `moral-self-preservation` — cynic

342 words tested. Baseline share of the readout: unsteered 5%, α=1.5 12%, α=4 31%.


#### Metric: `cells`


##### Fading: 109 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `erotisk` |  |  | -1.00 | -5.97 | 30.8 6.7 0.0 |
| `erotik` |  |  | -1.00 | -5.86 | 112.2 42.9 1.4 |
| `讨论` | to discuss |  | -1.00 | -5.62 | 24.1 2.9 0.0 |
| `potentially` |  |  | -1.00 | -5.59 | 23.6 1.4 0.0 |
| `erotisch` |  |  | -1.00 | -5.41 | 20.7 1.0 0.0 |
| `likely` |  |  | -1.00 | -5.16 | 17.3 3.9 0.0 |
| `â` |  |  | -1.00 | -5.16 | 17.3 9.6 0.0 |
| `geschichten` |  |  | -1.00 | -5.12 | 16.9 9.2 0.0 |
| `reset` |  |  | -0.87 | -5.04 | 15.9 0.0 0.0 |
| `bbw` |  |  | -1.00 | -4.99 | 15.4 6.7 0.0 |
| `please` |  |  | -1.00 | -4.90 | 28.9 18.3 0.5 |
| `events` |  |  | -1.00 | -4.85 | 14.0 4.8 0.0 |
| `steps` |  |  | -1.00 | -4.80 | 13.5 3.9 0.0 |
| `对话` | to talk (with sb) |  | -1.00 | -4.75 | 13.0 5.3 0.0 |
| `versus` |  |  | -1.00 | -4.70 | 12.5 3.4 0.0 |

##### Appearing: 99 words (34 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +1.00 | +8.55 | 0.0 46.2 187.3 |
| `really` |  | L0–38, 46–48 | +1.00 | +7.93 | 0.5 71.3 238.3 |
| `thing` |  | L6–22, 27–37 | +1.00 | +7.89 | 0.0 23.6 118.4 |
| `messed` |  | L16–39, 42–44 | +1.00 | +7.65 | 0.0 3.4 99.7 |
| `everything` |  |  | +1.00 | +7.62 | 0.0 30.8 97.7 |
| `somehow` |  | L46–55 | +1.00 | +7.57 | 0.0 27.9 94.4 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +7.50 | 0.0 35.1 90.0 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.28 | 0.0 11.6 77.0 |
| `靠谱` | reliable |  | +1.00 | +7.14 | 0.0 18.3 69.8 |
| `freaking` |  |  | +1.00 | +7.10 | 0.0 6.3 67.9 |
| `没人` | approx. not / person |  | +1.00 | +7.03 | 0.0 11.6 65.0 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.01 | 0.0 7.2 64.0 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +7.01 | 0.0 0.0 64.0 |
| `nobody` |  |  | +1.00 | +6.90 | 0.0 6.3 59.2 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +6.77 | 0.0 4.3 53.9 |

#### Metric: `weighted`


##### Fading: 119 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `erotik` |  |  | -1.00 | -5.72 | 36.3 8.3 0.2 |
| `please` |  |  | -1.00 | -4.19 | 9.5 6.9 0.1 |
| `potentially` |  |  | -1.00 | -4.09 | 8.0 0.3 0.0 |
| `s` |  |  | -1.00 | -4.09 | 9.4 4.6 0.1 |
| `讨论` | to discuss |  | -1.00 | -4.04 | 7.7 0.9 0.0 |
| `â` |  |  | -1.00 | -3.97 | 7.3 4.3 0.0 |
| `events` |  |  | -1.00 | -3.86 | 6.8 0.7 0.0 |
| `likely` |  |  | -1.00 | -3.85 | 6.7 1.1 0.0 |
| `reset` |  |  | -0.87 | -3.73 | 6.2 0.0 0.0 |
| `vs` |  |  | -1.00 | -3.50 | 5.2 2.7 0.0 |
| `instructions` |  |  | -1.00 | -3.45 | 8.7 5.4 0.3 |
| `erotisk` |  |  | -1.00 | -3.41 | 4.8 0.8 0.0 |
| `对你` | approx. right / you (informal,… |  | -1.00 | -3.37 | 5.9 4.0 0.1 |
| `the` |  |  | -1.00 | -3.36 | 14.2 3.9 0.9 |
| `available` |  |  | -1.00 | -3.31 | 7.0 2.0 0.3 |

##### Appearing: 98 words (33 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `really` |  | L0–38, 46–48 | +1.00 | +7.73 | 0.1 31.0 118.5 |
| `yeah` |  | L26–32 | +1.00 | +6.89 | 0.0 16.5 59.0 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +6.52 | 0.4 19.9 86.1 |
| `thing` |  | L6–22, 27–37 | +1.00 | +6.32 | 0.0 9.2 39.4 |
| `somehow` |  | L46–55 | +1.00 | +5.85 | 0.0 5.6 28.4 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +5.67 | 0.0 7.1 24.9 |
| `everything` |  |  | +1.00 | +5.56 | 0.0 9.2 23.0 |
| `maybe` |  | L0–25 | +1.00 | +5.42 | 0.1 12.3 25.6 |
| `messed` |  | L16–39, 42–44 | +1.00 | +5.40 | 0.0 0.7 20.6 |
| `怎么` | how? |  | +1.00 | +5.13 | 0.2 7.8 25.5 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +5.12 | 0.0 3.2 16.9 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +4.87 | 0.0 0.0 14.1 |
| `靠谱` | reliable |  | +1.00 | +4.84 | 0.0 3.9 13.8 |
| `nobody` |  |  | +1.00 | +4.82 | 0.0 1.7 13.7 |
| `freaking` |  |  | +1.00 | +4.80 | 0.0 1.1 13.4 |

### `condolences-letters` — cynic

214 words tested. Baseline share of the readout: unsteered 2%, α=1.5 14%, α=4 36%.


#### Metric: `cells`


##### Fading: 83 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `family` |  |  | -1.00 | -6.71 | 51.9 22.8 0.0 |
| `grieving` |  |  | -1.00 | -6.37 | 40.9 11.8 0.0 |
| `应该如何` | approx. ought to / how |  | -1.00 | -5.68 | 25.2 12.6 0.0 |
| `treasure` |  |  | -1.00 | -5.44 | 21.2 14.9 0.0 |
| `ˈ` |  |  | -1.00 | -5.44 | 21.2 2.4 0.0 |
| `讨论` | to discuss |  | -1.00 | -5.44 | 21.2 1.6 0.0 |
| `stories` |  |  | -1.00 | -5.39 | 20.5 0.8 0.0 |
| `pieces` |  |  | -1.00 | -5.33 | 19.7 7.9 0.0 |
| `preserved` |  |  | -1.00 | -5.28 | 49.6 35.4 0.8 |
| `memorial` |  |  | -1.00 | -5.28 | 18.9 9.4 0.0 |
| `mourn` |  |  | -1.00 | -5.28 | 18.9 6.3 0.0 |
| `非遗` | intangible cultural heritage (abbr. for… |  | -1.00 | -5.28 | 18.9 2.4 0.0 |
| `memories` |  |  | -1.00 | -5.22 | 18.1 8.7 0.0 |
| `treasures` |  |  | -1.00 | -5.22 | 18.1 8.7 0.0 |
| `book` |  |  | -1.00 | -5.16 | 17.3 11.8 0.0 |

##### Appearing: 68 words (30 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +9.05 | 0.0 20.5 264.4 |
| `yeah` |  | L26–32 | +1.00 | +8.29 | 0.0 46.4 156.6 |
| `got` |  | L0–23, 25 | +1.00 | +8.21 | 0.0 57.4 147.9 |
| `thing` |  | L6–22, 27–37 | +1.00 | +8.11 | 0.0 39.3 137.7 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +7.74 | 0.0 25.2 106.2 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.73 | 0.0 20.5 105.4 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +7.64 | 0.0 7.1 99.1 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +7.56 | 0.0 7.1 93.6 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +7.28 | 0.0 17.3 77.1 |
| `靠谱` | reliable |  | +1.00 | +7.00 | 0.0 12.6 63.7 |
| `everything` |  |  | +1.00 | +6.95 | 0.8 51.9 158.1 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +6.84 | 0.0 15.7 56.6 |
| `没人` | approx. not / person |  | +1.00 | +6.62 | 0.0 18.9 48.8 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +6.40 | 0.0 14.9 41.7 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +6.31 | 0.8 38.5 101.5 |

#### Metric: `weighted`


##### Fading: 82 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `family` |  |  | -1.00 | -5.58 | 23.4 5.8 0.0 |
| `memorial` |  |  | -1.00 | -4.58 | 11.4 5.2 0.0 |
| `to` |  |  | -1.00 | -4.43 | 12.2 6.8 0.1 |
| `preserved` |  |  | -1.00 | -4.38 | 11.6 7.7 0.1 |
| `应该如何` | approx. ought to / how |  | -1.00 | -4.33 | 9.5 2.7 0.0 |
| `grieving` |  |  | -1.00 | -4.28 | 9.2 2.9 0.0 |
| `poems` |  |  | -1.00 | -4.12 | 11.4 6.1 0.2 |
| `越来越好` | approx. more and… / good |  | -1.00 | -4.07 | 10.1 4.6 0.1 |
| `文书` | document |  | -1.00 | -4.07 | 9.6 4.7 0.1 |
| `your` |  |  | -1.00 | -4.04 | 14.4 11.2 0.4 |
| `ˈ` |  |  | -1.00 | -3.96 | 7.3 0.6 0.0 |
| `ㅤ` |  |  | -1.00 | -3.93 | 22.7 12.7 1.0 |
| `讨论` | to discuss |  | -1.00 | -3.92 | 7.1 0.6 0.0 |
| `文化遗产` | cultural heritage |  | -1.00 | -3.91 | 19.3 9.8 0.8 |
| `funeral` |  |  | -1.00 | -3.81 | 15.6 10.6 0.7 |

##### Appearing: 67 words (30 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.99 | 0.0 4.8 63.0 |
| `yeah` |  | L26–32 | +1.00 | +6.34 | 0.0 10.4 40.1 |
| `got` |  | L0–23, 25 | +1.00 | +6.06 | 0.0 17.0 32.8 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +6.03 | 3.1 114.3 237.6 |
| `everything` |  |  | +1.00 | +5.99 | 0.1 11.8 36.3 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.93 | 0.0 12.6 29.9 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +5.68 | 0.0 6.0 25.2 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +5.59 | 0.0 0.9 23.5 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +5.56 | 0.0 5.0 23.1 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.38 | 0.0 4.3 20.4 |
| `really` |  | L0–38, 46–48 | +1.00 | +5.19 | 5.2 108.2 208.0 |
| `stuff` |  | L6–36 | +1.00 | +5.18 | 1.7 31.6 78.6 |
| `靠谱` | reliable |  | +1.00 | +5.02 | 0.0 2.9 15.8 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.91 | 0.1 6.4 17.1 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +4.74 | 0.0 0.9 12.9 |

### `condolences-photos` — cynic

203 words tested. Baseline share of the readout: unsteered 2%, α=1.5 12%, α=4 33%.


#### Metric: `cells`


##### Fading: 65 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -6.96 | 61.8 56.5 0.0 |
| `grieving` |  |  | -1.00 | -6.29 | 38.5 15.2 0.0 |
| `diy` |  |  | -1.00 | -6.07 | 33.1 26.0 0.0 |
| `museum` |  |  | -1.00 | -5.82 | 27.8 10.8 0.0 |
| `photography` |  |  | -1.00 | -5.82 | 27.8 14.3 0.0 |
| `advice` |  |  | -1.00 | -5.52 | 22.4 17.9 0.0 |
| `collections` |  |  | -1.00 | -5.52 | 22.4 5.4 0.0 |
| `older` |  |  | -1.00 | -5.52 | 22.4 13.4 0.0 |
| `collection` |  |  | -1.00 | -5.40 | 20.6 7.2 0.0 |
| `怎样` | how |  | -1.00 | -5.40 | 20.6 17.9 0.0 |
| `暖心` | (coll.) heartwarming |  | -1.00 | -5.40 | 20.6 7.2 0.0 |
| `文物` | cultural relic |  | -1.00 | -5.34 | 19.7 10.8 0.0 |
| `condol` |  |  | -1.00 | -5.27 | 18.8 7.2 0.0 |
| `ˈ` |  |  | -1.00 | -5.27 | 18.8 1.8 0.0 |
| `建议` | to propose |  | -1.00 | -5.20 | 17.9 7.2 0.0 |

##### Appearing: 67 words (32 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `yeah` |  | L26–32 | +1.00 | +8.45 | 0.0 45.7 173.8 |
| `got` |  | L0–23, 25 | +1.00 | +8.07 | 0.0 48.4 133.5 |
| `everything` |  |  | +1.00 | +8.03 | 0.0 52.9 129.9 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +7.96 | 0.0 21.5 123.7 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.81 | 0.0 21.5 112.0 |
| `廉价` | cheaply-priced | L38–45, 48–62 | +1.00 | +7.75 | 0.0 17.0 107.5 |
| `thing` |  | L6–22, 27–37 | +1.00 | +7.64 | 0.0 31.4 99.5 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +7.43 | 0.0 6.3 86.0 |
| `somehow` |  | L46–55 | +1.00 | +7.26 | 0.0 36.7 76.2 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +7.19 | 0.0 2.7 72.6 |
| `靠谱` | reliable |  | +1.00 | +7.04 | 0.0 22.4 65.4 |
| `这些东西` | approx. these / thing |  | +1.00 | +6.94 | 0.0 28.7 60.9 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.76 | 1.8 26.0 247.3 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +6.69 | 0.0 10.8 51.1 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +6.58 | 0.0 4.5 47.5 |

#### Metric: `weighted`


##### Fading: 68 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `your` |  |  | -1.00 | -5.17 | 17.4 13.3 0.0 |
| `photograph` |  |  | -1.00 | -4.46 | 21.3 8.7 0.5 |
| `grieving` |  |  | -1.00 | -4.37 | 9.8 4.5 0.0 |
| `简易` | simple and easy |  | -1.00 | -4.21 | 18.8 5.3 0.5 |
| `diy` |  |  | -1.00 | -4.15 | 8.4 5.7 0.0 |
| `very` |  |  | -1.00 | -4.07 | 9.4 5.5 0.1 |
| `advice` |  |  | -1.00 | -4.05 | 7.8 6.6 0.0 |
| `photographs` |  |  | -1.00 | -4.04 | 11.0 5.3 0.2 |
| `建议` | to propose |  | -1.00 | -4.04 | 7.7 4.9 0.0 |
| `怎样` | how |  | -1.00 | -4.03 | 7.7 4.7 0.0 |
| `museum` |  |  | -1.00 | -4.01 | 7.5 2.3 0.0 |
| `ˈ` |  |  | -1.00 | -3.90 | 6.9 0.5 0.0 |
| `文物` | cultural relic |  | -1.00 | -3.84 | 6.7 2.4 0.0 |
| `ㅤ` |  |  | -1.00 | -3.80 | 22.7 14.0 1.2 |
| `digitally` |  |  | -1.00 | -3.71 | 14.0 5.6 0.6 |

##### Appearing: 66 words (30 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.49 | 0.2 7.9 65.8 |
| `yeah` |  | L26–32 | +1.00 | +6.46 | 0.0 9.9 43.6 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +6.33 | 1.9 96.3 196.2 |
| `everything` |  |  | +1.00 | +6.13 | 0.0 12.2 34.6 |
| `廉价` | cheaply-priced | L38–45, 48–62 | +1.00 | +5.95 | 0.0 3.1 30.3 |
| `got` |  | L0–23, 25 | +1.00 | +5.64 | 0.0 8.2 24.5 |
| `这些东西` | approx. these / thing |  | +1.00 | +5.58 | 0.0 10.7 23.4 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.54 | 0.0 4.4 22.8 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +5.53 | 0.0 4.5 22.6 |
| `really` |  | L0–38, 46–48 | +1.00 | +5.53 | 3.4 93.2 177.7 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.37 | 0.0 6.6 20.1 |
| `没钱` | approx. not / surname Qian | L38–41 | +1.00 | +5.16 | 0.0 0.4 17.4 |
| `somehow` |  | L46–55 | +1.00 | +4.94 | 0.0 6.4 14.9 |
| `靠谱` | reliable |  | +1.00 | +4.76 | 0.0 5.7 13.1 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +4.60 | 0.0 1.3 11.6 |

### `roleplay-neutral` — cynic

147 words tested. Baseline share of the readout: unsteered 2%, α=1.5 11%, α=4 31%.


#### Metric: `cells`


##### Fading: 42 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `visual` |  |  | -1.00 | -6.22 | 36.7 13.3 0.0 |
| `宁静` | tranquil |  | -1.00 | -5.93 | 30.0 21.1 0.0 |
| `colour` |  |  | -1.00 | -5.88 | 28.9 17.8 0.0 |
| `桌上` | approx. table / upper |  | -1.00 | -5.88 | 28.9 21.1 0.0 |
| `coffee` |  |  | -1.00 | -5.82 | 27.8 11.1 0.0 |
| `ˈ` |  |  | -1.00 | -5.82 | 27.8 3.3 0.0 |
| `traveling` |  |  | -1.00 | -5.76 | 26.7 2.2 0.0 |
| `upon` |  |  | -1.00 | -5.58 | 23.4 6.7 0.0 |
| `室内` | indoor |  | -1.00 | -4.21 | 91.2 37.8 4.5 |
| `erotik` |  |  | -1.00 | -4.08 | 45.6 22.2 2.2 |
| `answer` |  |  | -1.00 | -4.02 | 25.6 4.5 1.1 |
| `youtube` |  |  | -1.00 | -4.02 | 25.6 15.6 1.1 |
| `noise` |  |  | -1.00 | -3.82 | 22.2 14.5 1.1 |
| `rooms` |  |  | -1.00 | -3.71 | 64.5 30.0 4.5 |
| `1` |  |  | -1.00 | -3.13 | 23.4 12.2 2.2 |

##### Appearing: 54 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +8.72 | 0.0 14.5 210.2 |
| `yeah` |  | L26–32 | +1.00 | +8.63 | 0.0 60.1 198.0 |
| `everything` |  |  | +1.00 | +8.27 | 0.0 59.0 153.5 |
| `thing` |  | L6–22, 27–37 | +1.00 | +7.93 | 0.0 43.4 121.2 |
| `kinda` |  | L3, 5–37 | +1.00 | +7.90 | 0.0 21.1 119.0 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +7.80 | 0.0 6.7 111.2 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +7.78 | 0.0 34.5 109.0 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +7.45 | 0.0 4.5 86.8 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +6.89 | 0.0 0.0 59.0 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +6.84 | 0.0 12.2 56.7 |
| `没人` | approx. not / person |  | +1.00 | +6.75 | 0.0 14.5 53.4 |
| `guy` |  |  | +1.00 | +6.69 | 0.0 25.6 51.2 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +6.66 | 0.0 3.3 50.1 |
| `freaking` |  |  | +0.87 | +6.63 | 0.0 0.0 48.9 |
| `stuff` |  | L6–36 | +1.00 | +6.63 | 0.0 13.3 48.9 |

#### Metric: `weighted`


##### Fading: 45 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `colour` |  |  | -1.00 | -5.27 | 18.8 11.0 0.0 |
| `visual` |  |  | -1.00 | -4.59 | 11.6 2.5 0.0 |
| `室内` | indoor |  | -1.00 | -4.46 | 27.4 10.4 0.8 |
| `ˈ` |  |  | -1.00 | -4.43 | 10.3 1.7 0.0 |
| `answer` |  |  | -1.00 | -4.41 | 12.7 0.8 0.1 |
| `traveling` |  |  | -1.00 | -4.31 | 9.4 0.2 0.0 |
| `rooms` |  |  | -1.00 | -4.24 | 20.2 6.1 0.6 |
| `erotik` |  |  | -1.00 | -4.14 | 14.2 6.2 0.3 |
| `coffee` |  |  | -1.00 | -4.06 | 7.8 2.8 0.0 |
| `桌上` | approx. table / upper |  | -1.00 | -4.05 | 7.8 3.4 0.0 |
| `ㅤ` |  |  | -1.00 | -3.98 | 32.2 21.1 1.6 |
| `upon` |  |  | -1.00 | -3.97 | 7.3 1.4 0.0 |
| `宁静` | tranquil |  | -1.00 | -3.84 | 6.6 2.9 0.0 |
| `your` |  |  | -1.00 | -3.74 | 44.8 40.6 2.9 |
| `noise` |  |  | -1.00 | -3.53 | 7.1 4.4 0.2 |

##### Appearing: 56 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +7.32 | 0.7 68.4 184.2 |
| `really` |  | L0–38, 46–48 | +1.00 | +7.21 | 0.7 68.2 178.9 |
| `messed` |  | L16–39, 42–44 | +1.00 | +6.78 | 0.0 3.5 54.5 |
| `yeah` |  | L26–32 | +1.00 | +6.74 | 0.0 14.3 53.0 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.94 | 0.0 13.7 30.3 |
| `everything` |  |  | +1.00 | +5.93 | 0.0 11.6 29.9 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +5.81 | 0.0 6.8 27.5 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.66 | 0.0 5.2 24.8 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +5.27 | 0.0 0.9 18.8 |
| `没钱` | approx. not / surname Qian | L38–41 | +0.87 | +5.09 | 0.0 0.0 16.6 |
| `咱们` | (dialect) I or me |  | +1.00 | +5.01 | 0.0 4.9 15.6 |
| `got` |  | L0–23, 25 | +1.00 | +4.70 | 0.3 12.4 20.2 |
| `geile` |  | L38–45 | +1.00 | +4.67 | 2.0 15.1 62.6 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +4.66 | 0.0 0.6 12.1 |
| `nothing` |  | L43, 46–49, 51–52 | +1.00 | +4.65 | 0.0 2.1 12.1 |

### `roleplay-demon` — cynic

173 words tested. Baseline share of the readout: unsteered 4%, α=1.5 15%, α=4 34%.


#### Metric: `cells`


##### Fading: 51 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mysterious` |  |  | -1.00 | -6.66 | 50.2 35.8 0.0 |
| `神秘` | mysterious |  | -1.00 | -6.47 | 43.9 15.2 0.0 |
| `奇幻` | fantasy (fiction) |  | -1.00 | -6.35 | 40.3 17.9 0.0 |
| `observing` |  |  | -1.00 | -5.73 | 26.0 9.9 0.0 |
| `halloween` |  |  | -1.00 | -5.40 | 20.6 8.1 0.0 |
| `妖怪` | monster |  | -1.00 | -5.40 | 20.6 9.9 0.0 |
| `注视` | to look attentively at |  | -1.00 | -5.34 | 19.7 15.2 0.0 |
| `lgbtq` |  |  | -1.00 | -5.27 | 18.8 0.9 0.0 |
| `glowing` |  |  | -1.00 | -5.20 | 17.9 12.5 0.0 |
| `mystery` |  |  | -1.00 | -5.20 | 17.9 8.1 0.0 |
| `erotik` |  |  | -1.00 | -5.13 | 79.8 31.4 1.8 |
| `visiting` |  |  | -1.00 | -3.92 | 20.6 10.8 0.9 |
| `rooms` |  |  | -1.00 | -3.88 | 46.6 19.7 2.7 |
| `eerie` |  |  | -1.00 | -3.86 | 19.7 15.2 0.9 |
| `scenes` |  |  | -1.00 | -3.86 | 19.7 4.5 0.9 |

##### Appearing: 54 words (27 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +9.06 | 0.0 28.7 266.1 |
| `kinda` |  | L3, 5–37 | +1.00 | +8.04 | 0.0 41.2 130.8 |
| `everything` |  |  | +1.00 | +7.74 | 0.0 52.9 106.6 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +7.68 | 0.0 21.5 102.2 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +7.55 | 0.0 37.6 93.2 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +7.34 | 0.0 4.5 80.7 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +7.26 | 0.0 20.6 76.2 |
| `没人` | approx. not / person |  | +1.00 | +6.98 | 0.0 14.3 62.7 |
| `freaking` |  |  | +1.00 | +6.85 | 0.0 3.6 57.4 |
| `everytime` |  |  | +1.00 | +6.81 | 0.0 9.0 55.6 |
| `somehow` |  | L46–55 | +1.00 | +6.74 | 0.0 44.8 52.9 |
| `insanely` |  |  | +1.00 | +6.71 | 0.0 4.5 52.0 |
| `假装` | to feign | L56–61 | +1.00 | +6.58 | 0.0 8.1 47.5 |
| `stuff` |  | L6–36 | +1.00 | +6.53 | 0.0 14.3 45.7 |
| `guy` |  |  | +1.00 | +6.50 | 0.0 26.0 44.8 |

#### Metric: `weighted`


##### Fading: 56 words (0 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `mysterious` |  |  | -1.00 | -5.74 | 26.3 19.6 0.0 |
| `神秘` | mysterious |  | -1.00 | -5.11 | 16.7 3.8 0.0 |
| `erotik` |  |  | -1.00 | -5.08 | 25.6 8.6 0.3 |
| `奇幻` | fantasy (fiction) |  | -1.00 | -5.00 | 15.4 2.8 0.0 |
| `lgbtq` |  |  | -1.00 | -4.21 | 8.8 0.1 0.0 |
| `rooms` |  |  | -1.00 | -3.98 | 13.6 3.4 0.4 |
| `stealth` |  |  | -1.00 | -3.95 | 8.6 4.7 0.1 |
| `fictional` |  |  | -1.00 | -3.84 | 12.0 5.5 0.4 |
| `observing` |  |  | -1.00 | -3.78 | 6.4 1.7 0.0 |
| `室内` | indoor |  | -1.00 | -3.78 | 16.4 4.3 0.7 |
| `察觉` | to sense |  | -1.00 | -3.78 | 7.7 7.2 0.1 |
| `ㅤ` |  |  | -1.00 | -3.70 | 21.1 15.7 1.2 |
| `the` |  |  | -1.00 | -3.65 | 24.0 3.5 1.4 |
| `角色` | role |  | -1.00 | -3.60 | 11.9 4.6 0.5 |
| `注视` | to look attentively at |  | -1.00 | -3.59 | 5.5 3.0 0.0 |

##### Appearing: 53 words (24 baseline), top 15 by |log2FC|

| word | gloss | baseline | rho | log2FC | ‰ by strength |
| --- | --- | --- | --- | --- | --- |
| `messed` |  | L16–39, 42–44 | +1.00 | +7.25 | 0.0 7.1 75.3 |
| `really` |  | L0–38, 46–48 | +1.00 | +6.60 | 0.8 55.6 122.2 |
| `guys` |  | L2–15, 18–22, 25–26, 34–36 | +1.00 | +5.97 | 0.0 10.6 30.9 |
| `kinda` |  | L3, 5–37 | +1.00 | +5.88 | 0.0 13.2 28.9 |
| `pretty` |  | L0–38, 42–44, 46–56 | +1.00 | +5.84 | 1.8 61.7 129.7 |
| `yeah` |  | L26–32 | +1.00 | +5.71 | 0.5 18.4 51.5 |
| `fake` |  | L48, 54–60, 62 | +1.00 | +5.57 | 0.0 8.3 23.2 |
| `everything` |  |  | +1.00 | +5.54 | 0.0 11.4 22.7 |
| `thing` |  | L6–22, 27–37 | +1.00 | +5.37 | 0.3 15.8 32.3 |
| `crazy` |  | L15, 23–24, 29–34 | +1.00 | +5.08 | 0.0 5.4 16.5 |
| `everytime` |  |  | +1.00 | +5.04 | 0.0 2.3 15.9 |
| `伪造` | to forge | L39–45, 49, 57–60 | +1.00 | +4.99 | 0.0 0.6 15.4 |
| `weird` |  | L23–24, 26–34, 37, 42–44, 46–47 | +1.00 | +4.84 | 0.4 14.1 25.4 |
| `没人` | approx. not / person |  | +1.00 | +4.76 | 0.0 4.0 13.1 |
| `got` |  | L0–23, 25 | +1.00 | +4.67 | 0.1 11.4 14.8 |
