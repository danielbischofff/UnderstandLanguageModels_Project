# Metric review for question-bank-mini-band

## Design and interpretation

396 trials: 12 questions × (8 personas × 2 layer schemes × 2 band widths + 1 unsteered control). Generation is greedy, seed 0, capped at 256 tokens. Capture layers are 29, 32, 35 and 62, at post-intervention block outputs. Each question is the first question of one topic. Q01 and Q23 are pilot calibration questions: report all 12 descriptively and a sensitivity summary excluding these two; do not call them held-out evaluation.

Band β is a half-width fraction. The implementation clamps the character-direction projection to character center ± β × direction norm. β=0 pins it to the center and is the tighter intervention; β=0.25 permits a wider interval. Only character `unsteered` is the control. Two widths do not establish a general dose-response relationship.

## Relevant metrics and checked definitions

Let W be a persona word set, r(w) its one-based full-vocabulary rank, x = RMSNorm(J_layer h), u_w a unit unembedding row, and c_W the normalized mean of those unit rows. The readout's top-k consists of wordlike tokens, with probabilities renormalized over those retained tokens.

| Metric | Definition | Interpretation and role |
|---|---|---|
| Mean reciprocal rank (MRR) | mean over w in W of 1/r(w) | Primary rank association; larger is stronger. This implementation averages reciprocal ranks over all selected words, rather than taking only the first relevant word. |
| Activation-to-word cosine | mean over w in W of cosine(x, U_w) | Primary directional association, independent of top-k selection. Uses learned final RMSNorm gain after J-Lens transport. Larger is stronger, not a probability. |
| Centroid cosine | sum over top-k tokens i of normalized p_i × cosine(U_i, c_W) | Complementary association of surfaced words with the persona set; conditional on the retained top-k and sensitive to its composition. |
| Median / mean / minimum rank | median / mean / minimum of r(w) | Secondary diagnostics; smaller is stronger. Minimum rank is particularly sensitive to one good match and word-set size. |
| Weighted maximum cosine | probability-weighted nearest-set-word cosine for each surfaced token | Secondary diagnostic; sensitive to one matching word and set size. |
| Top-1 cosine | cosine of highest-ranked retained word with c_W | Secondary, single-token diagnostic. |
| Maximum cosine | largest surfaced-token / bank-word cosine | Secondary extreme-value diagnostic; sensitive to set size. |
| Output quality | length, cap frequency, repeated token 4-grams; subsequently coherence, relevance and persona adherence review | Guardrail for interpretation. Lexical repetition and changed text alone cannot establish successful steering. |

For each of the three primary/complementary association scores, retain these separate quantities:

1. Intended-persona score, averaged over valid answer positions.
2. Paired control delta: intended score under steering minus that same set's score in the same question's control.
3. Persona margin: intended score minus the mean of the other seven sets in the same bank.
4. Margin delta: steered margin minus control margin. Prefer this alongside the control delta to distinguish intended association from a broad increase shared by many personas.

The bank pipeline compares against all seven other sets in a bank, including personas absent from this experiment. It excludes cross-bank personas from that bank's intended-persona summaries. Keep banks, contextualized/general word sets, token policies, layer schemes and readout layers separate.

Topic-balanced aggregation averages questions within topic, then topics equally. Here each topic has only one question, so it equals an equal-weight question mean when coverage is complete. Always show valid-question and valid-position counts. Twelve topics are the descriptive sample, not thousands of independent tokens. Shared controls create dependence between conditions; there are no random-seed replicates. Do not infer significance by treating tokens or layers as independent trials.

## Availability and execution checks

- **Available and audited now:** trial coverage, response text and IDs, activation artifacts, response lengths, cap frequency, paired exact-output equality and repeated 4-grams. See `audit.json` and `response-diagnostics.md`.
- **Association metrics are not yet computed:** J-Lens was disabled; every saved readout has zero layers and the shared readout vocabulary is empty. Empty readouts are not zero association scores.
- **Missing local readout dependencies:** no J-Lens/model readout assets at the configured local data location. Recomputing requires the fitted lens, pinned tokenizer, final RMSNorm gain and unembedding. Full model inference is unnecessary.
- **Archive ingestion gap:** `quant.load_run` accepts native per-trial `result.json` directories or ZIP bundles, not this extracted `index.json` layout. The ZIP path also does not attach retained activation provenance. A loader must map descriptors to provenance artifacts by question/persona/scheme/beta, with duplicate checks.
- **Cached event gap:** `AnalysisReader.probe_artifact` currently selects prefill events only. This run stores prompt prefill and separate one-token decode events. The reader must assemble events by layer and absolute position, including decode events, then explicitly represent the missing final input-token activation. Otherwise prompt-only series cannot match the full token metadata. Use chunked readout to avoid materializing all full-vocabulary logits at once.
- **Tokenization audit is stale for current code:** both bank file hashes still match; the resolver source hash does not, and the audited tokenizer path is absent. This is a validation failure, not proof that token mappings changed. Revalidate with the pinned tokenizer; do not bypass the hash check. Report retained word counts and unique token counts, proxy exclusions and collisions.
- **Validation limitation:** attempted metric/bank test execution could not start because the project's `.venv` lacks pytest. The standalone artifact audit uses system Python and NumPy. Existing metric implementation changes in the working tree were left untouched.

## Analysis order

1. Review truncation and short/degenerate outputs before interpreting persona association. Keep capped responses flagged; silently dropping them would select a strongly biased subset.
2. Restore the pinned readout assets and revalidate word mappings; resolve ingestion and cached-position alignment with small fixtures before full scoring.
3. Compute the three association measures at all four captured layers, using complete-token sets as the main view and proxy-inclusive sets as sensitivity checks. Confirm the lens actually fits each requested layer; never silently substitute layers.
4. Compare paired deltas and margin deltas for β=0 vs 0.25 within each layer scheme, then compare schemes at matched β. Show individual questions before topic averages.
5. Analyze prompt positions separately: they share the same input across conditions, whereas generated answer trajectories differ. For the layer-32-only scheme, prompt readout at layer 29 is a useful upstream negative control. That argument does not hold for answer positions once generated tokens diverge.
6. Examine Q01/Q23-excluded summaries and perform qualitative coherence/relevance/persona review. A stronger internal word association is not by itself better persona behavior.

The current findings describe data integrity and generation behavior. They do not yet rank personas or establish which steering scheme works best.
