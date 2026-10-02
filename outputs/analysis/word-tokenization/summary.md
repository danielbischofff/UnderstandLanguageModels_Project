# Word tokenization audit

Model: `Qwen/Qwen3-32B`; tokenizer revision: `9216db5781bf21249d130ec9da846c4624c16137`.

4608 word occurrences; 766 distinct words; 60 distinct first-token proxies.
300 occurrences use proxies; 11 distinct words collide with another word's selected token.

Model and J-Lens tokenizers have identical vocabularies and agree on all audited encodings.
Counts below include repeated general-word occurrences across questions. Collisions exclude repeated occurrences of the same word.

| Bank | Persona | Set type | Occurrences | Proxy occurrences | Minimum complete words per question |
|---|---|---|---:|---:|---:|
| question_bank | activist | contextualized | 144 | 3 | 5 |
| question_bank | activist | non_contextualized | 144 | 24 | 5 |
| question_bank | caregiver | contextualized | 144 | 7 | 5 |
| question_bank | caregiver | non_contextualized | 144 | 0 | 6 |
| question_bank | entrepreneur | contextualized | 144 | 2 | 5 |
| question_bank | entrepreneur | non_contextualized | 144 | 0 | 6 |
| question_bank | lawyer | contextualized | 144 | 2 | 5 |
| question_bank | lawyer | non_contextualized | 144 | 0 | 6 |
| question_bank | mystic | contextualized | 144 | 9 | 5 |
| question_bank | mystic | non_contextualized | 144 | 24 | 5 |
| question_bank | poet | contextualized | 144 | 7 | 5 |
| question_bank | poet | non_contextualized | 144 | 24 | 5 |
| question_bank | scientist | contextualized | 144 | 8 | 4 |
| question_bank | scientist | non_contextualized | 144 | 0 | 6 |
| question_bank | skeptic | contextualized | 144 | 14 | 4 |
| question_bank | skeptic | non_contextualized | 144 | 0 | 6 |
| question_bank_contrastive | altruist | contextualized | 144 | 1 | 5 |
| question_bank_contrastive | altruist | non_contextualized | 144 | 24 | 5 |
| question_bank_contrastive | builder | contextualized | 144 | 3 | 5 |
| question_bank_contrastive | builder | non_contextualized | 144 | 0 | 6 |
| question_bank_contrastive | guardian | contextualized | 144 | 4 | 5 |
| question_bank_contrastive | guardian | non_contextualized | 144 | 24 | 5 |
| question_bank_contrastive | mediator | contextualized | 144 | 3 | 5 |
| question_bank_contrastive | mediator | non_contextualized | 144 | 0 | 6 |
| question_bank_contrastive | narcissist | contextualized | 144 | 4 | 5 |
| question_bank_contrastive | narcissist | non_contextualized | 144 | 24 | 5 |
| question_bank_contrastive | predator | contextualized | 144 | 1 | 5 |
| question_bank_contrastive | predator | non_contextualized | 144 | 0 | 6 |
| question_bank_contrastive | provocateur | contextualized | 144 | 21 | 4 |
| question_bank_contrastive | provocateur | non_contextualized | 144 | 48 | 4 |
| question_bank_contrastive | saboteur | contextualized | 144 | 19 | 4 |
| question_bank_contrastive | saboteur | non_contextualized | 144 | 0 | 6 |

## First-token proxies

| Word | Selected surface | Selected piece | Decoded piece | Colliding words |
|---|---|---|---|---|
| artifice | ` artifice` | `Ġart` | ` art` |  |
| blockage | ` blockage` | `Ġblock` | ` block` |  |
| causation | ` causation` | `Ġcaus` | ` caus` |  |
| chronology | ` chronology` | `Ġchron` | ` chron` |  |
| coexistence | ` coexistence` | `Ġco` | ` co` |  |
| companionship | ` companionship` | `Ġcompanions` | ` companions` |  |
| concealment | ` concealment` | `Ġconceal` | ` conceal` |  |
| confounding | ` confounding` | `Ġconf` | ` conf` |  |
| contingencies | ` contingencies` | `Ġconting` | ` conting` |  |
| corroboration | ` corroboration` | `Ġcor` | ` cor` |  |
| counterexample | ` counterexample` | `Ġcounter` | ` counter` | counterfactual |
| counterfactual | ` counterfactual` | `Ġcounter` | ` counter` | counterexample |
| crossroads | ` crossroads` | `Ġcross` | ` cross` |  |
| derailment | ` derailment` | `Ġderail` | ` derail` |  |
| desecration | ` desecration` | `Ġdes` | ` des` |  |
| discernment | ` discernment` | `Ġdiscern` | ` discern` |  |
| embankments | ` embankments` | `Ġemb` | ` emb` |  |
| erasure | ` erasure` | `Ġer` | ` er` |  |
| exclusivity | ` exclusivity` | `Ġexclus` | ` exclus` |  |
| foreseeability | ` foreseeability` | `Ġforesee` | ` foresee` |  |
| fragility | ` fragility` | `Ġfrag` | ` frag` |  |
| generalization | ` generalization` | `Ġgeneral` | ` general` |  |
| gentleness | ` gentleness` | `Ġgent` | ` gent` |  |
| grandeur | ` grandeur` | `Ġgrand` | ` grand` | grandiosity |
| grandiosity | ` grandiosity` | `Ġgrand` | ` grand` | grandeur |
| horizons | ` horizons` | `Ġhor` | ` hor` |  |
| hydrology | ` hydrology` | `Ġhydro` | ` hydro` |  |
| interdependence | ` interdependence` | `Ġinter` | ` inter` |  |
| irreverence | ` irreverence` | `Ġirre` | ` irre` |  |
| linguistics | ` linguistics` | `Ġlingu` | ` lingu` |  |
| lyricism | ` lyricism` | `Ġlyric` | ` lyric` |  |
| misdirection | ` misdirection` | `Ġmis` | ` mis` | misreporting, misrepresentation |
| misreporting | ` misreporting` | `Ġmis` | ` mis` | misdirection, misrepresentation |
| misrepresentation | ` misrepresentation` | `Ġmis` | ` mis` | misdirection, misreporting |
| mobilization | ` mobilization` | `Ġmobil` | ` mobil` |  |
| nourishment | ` nourishment` | `Ġnour` | ` nour` |  |
| omissions | ` omissions` | `Ġo` | ` o` |  |
| onboarding | ` onboarding` | `Ġon` | ` on` |  |
| pluralism | ` pluralism` | `Ġplural` | ` plural` |  |
| preparedness | ` preparedness` | `Ġprepared` | ` prepared` |  |
| provenance | ` provenance` | `Ġproven` | ` proven` |  |
| provocation | ` provocation` | `Ġprov` | ` prov` |  |
| reassurance | ` reassurance` | `Ġreass` | ` reass` |  |
| reciprocity | ` reciprocity` | `Ġrecipro` | ` recipro` |  |
| remembrance | ` remembrance` | `Ġrem` | ` rem` |  |
| safeguarding | ` safeguarding` | `Ġsafeguard` | ` safeguard` | safeguard |
| seedlings | ` seedlings` | `Ġseed` | ` seed` |  |
| selflessness | ` selflessness` | `Ġself` | ` self` |  |
| spoiling | ` spoiling` | `Ġspo` | ` spo` |  |
| stewardship | ` stewardship` | `Ġsteward` | ` steward` |  |
| stillness | ` stillness` | `Ġstill` | ` still` |  |
| subjectivity | ` subjectivity` | `Ġsubject` | ` subject` |  |
| tenderness | ` tenderness` | `Ġtend` | ` tend` |  |
| tradeoffs | ` tradeoffs` | `Ġtrade` | ` trade` | trade |
| transcendence | ` transcendence` | `Ġtranscend` | ` transcend` |  |
| transgression | ` transgression` | `Ġtrans` | ` trans` |  |
| unfairness | ` unfairness` | `Ġunfair` | ` unfair` |  |
| uprooting | ` uprooting` | `Ġup` | ` up` |  |
| vigilance | ` vigilance` | `Ġvigil` | ` vigil` |  |
| wholeness | ` wholeness` | `Ġwh` | ` wh` |  |

The banks are unchanged. This audit does not filter words or change similarity scoring.
The current analysis CLI still resolves words itself; consuming this mapping or applying complete-only filters requires a separate integration.
A single-token representation can be a capitalization or no-space variant, not necessarily the usual token in running text.
