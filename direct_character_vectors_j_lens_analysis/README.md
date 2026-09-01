# Direct character vectors through the Jacobian lens

Reads the **`bard` role vector** of Qwen3-32B out through the **Jacobian lens**,
one layer at a time, and reports the vocabulary tokens each layer's direction is
disposed to make the model say. Output is a single JSON document plus a
self-contained web viewer.

This is the Hypothesis-1 probe from
[`../Project Proposals/j_lens_persona_space_relationship.md`](../Project%20Proposals/j_lens_persona_space_relationship.md),
run *directly on the persona vectors* rather than on activations from a steered
forward pass — no 32B model needs to be loaded.

## Method

The Jacobian lens ([Anthropic, 2026](https://arxiv.org/abs/2607.15495);
[code](https://github.com/anthropics/jacobian-lens)) transports a residual-stream
vector at layer `l` into the final-layer basis with the corpus-averaged
input–output Jacobian, then decodes it with the model's own unembedding:

```
lens_l(h) = unembed( J_l @ h ),   J_l = E[ ∂h_final / ∂h_l ],   unembed = lm_head ∘ final_RMSNorm
```

Here `h` is the per-layer persona direction from
[The Assistant Axis](https://arxiv.org/abs/2601.10387) (Lu et al.), so
`lens_l(v_l)` answers: *what is this character's direction, at this depth,
inclined to verbalize?*

Because the readout ends in an RMSNorm, it is **invariant to the scale** of the
direction — only its orientation matters. That also makes the analysis immune to
the sum-vs-mean question raised below.

### Directions analysed

The published role vectors are **raw mean activations** — the mean residual over
samples a judge scored as fully in-role
([`pipeline/4_vectors.py`](https://github.com/safety-research/assistant-axis)) —
*not* difference vectors. A raw mean is dominated by the large layer-generic
component that every persona shares, so reading it out mostly returns generic
high-frequency tokens. The headline direction is therefore the contrast:

| key | direction | role |
| --- | --- | --- |
| `bard_minus_default` | `bard − default` | **primary** — what is distinctively bard; the steering direction |
| `bard` | `bard` | the published role vector as-is |
| `default` | `default` | control — the default Assistant persona's mean |
| `assistant_axis` | `default_mean − role_mean` | control — the published axis |
| `random_matched` | Gaussian noise | null control, per-layer norm matched to the contrast |

Each is read out in two modes: `jlens` (with the Jacobian transport) and
`logit_lens` (unembedded directly). The difference isolates what the lens adds
over a plain logit-lens reading.

## What it shows

Read down the stack, `bard − default` moves through three registers, and the readout
grows steadily more confident — entropy falls from 8.2 nats to 1.5 at layer 57:

| layers | dominant word-like tokens |
| --- | --- |
| 0–3 | `3`, `2`, `5`, `0`, `MessageLookup` — digits and tokenizer junk, no persona content |
| 4–12 | `mighty`, `tale`, `tales`, `古老` *ancient*, `tho`, `glorious` |
| 13–45 | `mighty`, `永恒` *eternal*, `tale`, `古老`, `伟大的` *great*, `tales`, `forever`, `glory` |
| 46–62 | `tale`, `tales`, `whispered`, `whispers`, `weary`, `enchanted`, `cunning`, `song` |

An epic-legendary register through the middle of the stack, resolving into a
storyteller's narrative voice near the output. It is also markedly **multilingual** —
Chinese tokens for the same concepts rank alongside the English ones, which is what the
workspace paper reports for verbalizable content generally.

The controls behave as they should:

| direction | deep layers (46–62) |
| --- | --- |
| `default` (Assistant) | `communities`, `mindset`, `lifestyle`, `LGBTQ`, `发展空间` *room to grow* |
| `assistant axis` | `Activities`, `Additionally`, `recommendations`, `Overall` in mid layers, collapsing to bare list numerals `1`–`6` by layer 56 |
| `random (norm-matched)` | incoherent multilingual web-scrape noise at every depth |

The assistant axis reading out as *list formatting and discourse connectives* is a
useful independent check: that is the Assistant's most characteristic surface habit.

**The Assistant is the null control.** There is an `assistant` role vector among the
275, and reading it out exactly like any other character produces no character at all —
web-scrape noise early, then generic institutional filler (`Numerous`, `subsequent`,
`associated`, `Establish`) through the deep layers, where `bard` is producing `tale` and
`whispered`. The geometry says why: `assistant` sits at cosine **0.9993** from
`default`, so `assistant − default` is a ~5% residual and the lens is mostly reading
sampling noise.

| role | cos(role, default), L32 / L58 | ‖role − default‖ / ‖default‖, L32 / L58 |
| --- | --- | --- |
| `assistant` | 0.9993 / 0.9991 | 0.05 / 0.05 |
| `bard` | 0.9386 / 0.9132 | 0.35 / 0.45 |
| `pirate` | 0.9186 / 0.8398 | 0.40 / 0.58 |

The default persona simply *is* the Assistant — which is the Assistant Axis paper's
premise — and the useful consequence here is a null that runs through the identical
pipeline, confirming the bard signal is not an artifact of the subtraction. Run it with
`--role assistant`; the output is kept as `out/readout_assistant.json`.

**The contrast is the only direction that never degenerates.** At eight layers some
direction collapses onto a formatting attractor — a newline, a zero-width space, a
backtick — with entropy below 0.5. This happens to the norm-matched random direction at
six layers, and to the *raw* `bard` and `default` means at layer 33, but never to
`bard − default`, whose entropy never falls below 1.5. That is direct evidence that the
published role vectors carry a large layer-generic component, and that subtracting the
default removes it.

**The Jacobian transport earns its keep.** The same contrast read out *without* it —
plain logit lens — is dominated through layers 0–45 by junk fragments (`因地制`, `名为`,
`落到实`, `ImplOptions`) and only recovers `tale`/`tales` in the final band. The
Jacobian version surfaces coherent bardic vocabulary from layer 4 onward.

Taken together this is a positive result for Hypothesis 1 at the level of the vectors
themselves: a persona direction has a specific, legible, non-generic verbalizable
signature, distinct from the Assistant's and absent from matched noise.

## The whole cast

`run_sweep.py` runs the same readout over all 275 characters in two contrasts, one
vocabulary pass each (~22 min total); `noise_floor.py` then calibrates it.

**Pick the metric carefully — the two obvious ones are both confounded, in opposite
directions.** *Entropy* looks like confidence, but a readout that collapses onto a
single junk attractor (a newline, a backtick) is also very low entropy: the `assistant`
null bottoms out at 1.06 nats, **sharper** than `bard`'s 1.53. *Distinctiveness* — how
rare a signature's tokens are across the cast — looks like specificity, but a direction
landing on its own private patch of tokenizer garbage scores maximally distinctive.
Neither separates signal from noise.

**Stability does.** Take the mean Jaccard overlap between the top-8 sets of
*consecutive* layers across the deep band. A real character signal persists over a
stretch of layers; a degenerate collapse is layer-local, and noise is uncorrelated from
one layer to the next.

| direction | layer-to-layer stability |
| --- | --- |
| 24 norm-matched random directions | **0.00** — not one token survives a layer |
| the cast, median | **0.75** |
| `assistant` | 0.64 — rank 255 of 275 |
| most stable | `cartographer` 0.96, `reporter` 0.96, `architect` 0.94 |
| least stable | `provincial` 0.58, `criminal` 0.58, `luddite` 0.59 |

**All 275 characters clear the random ceiling**, in both contrasts. `assistant` sits
near the bottom of its own cast, which is what a null should do — though note it is not
*at* the floor: its readout is stable, just stably generic rather than character-like.

**Name leakage.** The character's own name token appears somewhere in its deep readout
for **40%** of characters, and inside the 6-token signature for **19%**. Real, but not
the whole story: four signatures in five are not simply re-reading the name.

## A different direction: refusal

Nothing here is specific to personas — the readout applies to anything living in the
residual stream. `refusal_readout.py` runs it on a difference-of-means **refusal**
direction for Qwen3-32B, in the style of
[Arditi et al.](https://arxiv.org/abs/2406.11717):

| layers | tokens |
| --- | --- |
| L08–32 | the rare-token attractor, plus `阴谋` *conspiracy*, `作弊` *cheating*, `骗局` *scam* |
| L40–46 | `illegal`, `非法`, `违法`, `色情`, `走私` *smuggling*, `伪造` *forgery*, `毒品` *drugs* |
| L52 | `legality`, `禁止` *forbidden*, `不允许` *not allowed*, `拒绝` *refuse*, `我不想` *I don't want* |
| L56–62 | `sorry`, `抱歉`, `对不起`, `apologize`, `regret`, `죄` — entropy down to **0.1** |

The subject matter of harm, then its legal status, then the apology itself. Negating
the direction gives digits and noise (entropy 8.6 at the output), which settles the
undocumented sign convention; a norm-matched random direction is noise at every depth.
Layer 46 — where the harmful-content cluster peaks — is the layer the source selected
for ablation, arrived at independently.

**The published file needed forensics**, documented in the module: its README states
dimension 6656 (Qwen3-32B's residual stream is 5120) and describes a single vector,
but the 21 MB file is a `[5120]` *view* into a `[512, 5120]` float64 storage that
`torch.save` carried along — the entire 8-position × 64-layer candidate grid. The axis
order was recovered from the norm profile: read one way it rises with depth (Spearman
0.99), transposed it is unordered (0.08). Provenance is third-party and unverified;
the readout is the evidence, since a mislabelled direction would not verbalize refusal.

**It also bears on the attractor.** The rare-token cluster shows up at L08–L32 here
too — on a direction computed by someone else, from a different dataset, with no
personas and no cast mean involved. So it is neither persona-specific nor caused by
the subtraction in this project: it looks like a property of difference-of-means
directions read at early and middle layers.

## Side investigation: is the early-layer junk the lens?

Independent of the persona work. Every J-Lens readout here is polluted in early and
middle layers by rare web-crawl tokens. **R-lens** (RelP,
[arXiv:2508.21258](https://arxiv.org/abs/2508.21258)) is the *same* Jacobian estimator
read through an LRP-modified backward graph — forward values bit-identical, only the
gradients differ — and its claimed advantage is early-layer faithfulness.
[`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses)
publishes matched J/R pairs, so both can read one direction with everything else held
fixed. `compare_lenses.py` does that for Qwen3.6-27B.

**The direction had to be recovered.** None is published for this model, but
abliterated builds are — and abliteration projects the direction out of the weights, so
`W_base − W_abliterated` is rank one with the direction as its leading left singular
vector. `recover_refusal_direction.py` diffs one matrix per layer (~63 MB a side rather
than ~120 GB of models): `s₁/‖D‖_F` was **0.997** at layer 11 and **0.996** at layer 31,
and the two independently recovered vectors agree at **|cos| = 1.0000**. SVD fixes no
sign; the semantics sit on `−r`, and `+r` is flat noise at near-maximum entropy.

**R-lens reads the semantics nine layers earlier.**

| | J-lens | R-lens |
| --- | --- | --- |
| first layer with a harm token | L12 | **L3** |
| harm tokens in layers 0–15 | 1 | **30** |
| top-k agreement, early → deep | 0.10 | → 0.77 |

At L03 J-lens returns `agraph NOTE rape SDL pletely ilst SD` — tokenizer debris — while
R-lens returns `legitimately attempting attempted incredibly horribly attempt`. By L12
R-lens is on `合法 illegal 危害 malicious 非法` while J-lens is still on profanity. The
arms converge by the deep layers, which is exactly where RelP claims no advantage.

**The erotic cluster is a separate phenomenon.** It does not appear in Qwen3.6-27B at
all — either lens, any layer. The two models have different tokenizers (152k vs 248k
vocab), so that cluster looks like a property of one vocabulary's tail rather than of
J-Lens. So the findings are independent: J-Lens *does* carry early-layer error and RelP
removes it; separately, *which* junk surfaces is model-specific. Settling the cluster
itself needs an R-lens for Qwen3-32B, which that repo does not ship.

## What the inputs actually are

Two things about the published artifacts are worth recording, because neither is
documented where you would expect and both change how results should be read.

**1. The lens file is a fit checkpoint, not a saved lens.** The Hub file
`Qwen3-32B_jacobian_lens.pt` contains `{jacobian_sum, n_done, next_idx,
source_layers}`, not the `{J, n_prompts, ...}` that `JacobianLens.save()` writes
— so `jlens.JacobianLens.load()` rejects it outright. `jacobian_sum` is a *sum*
over prompts and must be divided by `n_done`. [`readout.py`](src/readout.py)
handles both layouts.

**2. It was fitted on 80 prompts, not the 615 its config claims.** The
checkpoint reports `n_done = 80`, and the companion `convergence.csv` has
exactly 80 rows — while `config.yaml` advertises `prompts_fitted: 615`. At
n=80 the fit's `mean_rel_change` is ≈0.026, an order of magnitude above the
`stop_at_delta: 0.002` the run configured, so this lens had **not** converged to
its own criterion. The `jlens` README calls ~100 prompts "usable", so 80 is
borderline. Treat fine distinctions between adjacent layers with suspicion.

Shapes, for reference: role vectors `[64, 5120]` bf16; Jacobians `[5120, 5120]`
fp32 for layers **0–62** (layer 63 has no fitted Jacobian, so it appears only in
the `logit_lens` mode); vocabulary 151,936.

## Layout

```
src/fetch_data.py     download every input into data/ (~8.2 GB, resumable)
src/inspect_lens.py   read a .pt archive's metadata without loading tensors —
                      works over HTTP, which is how the two points above were found
src/unembed.py        Qwen3 final RMSNorm + lm_head, streamed from disk
src/readout.py        memory-mapped per-layer Jacobian transport
src/run_analysis.py   one character in depth: directions × modes × layers
                      -> out/readout.json
src/run_sweep.py      all 275 characters, two contrasts -> out/atlas.json
src/refusal_readout.py    the same readout on a refusal direction
src/noise_floor.py    calibrates the atlas against random directions
src/raw_vs_contrast.py    open question: what causes the rare-token attractor
src/build_web.py      inline a readout into its viewer -> out/*.html
web/template.html         the single-character viewer
web/atlas_template.html   the whole-cast viewer
web/refusal_template.html the refusal viewer
out/*.json            results (committed; the pages are rebuilt from them)

# the Qwen3.6-27B J-lens vs R-lens side investigation
src/fetch_qwen36.py             its lenses, unembedding and tokenizer
src/recover_refusal_direction.py  a refusal direction, by rank-1 weight diff
src/compare_lenses.py           the two arms side by side -> out/lens_compare.json
web/compare_template.html       the comparison viewer
```

Hub downloads authenticate with `HF_TOKEN` or `~/.cache/huggingface/token` when one is
present; anonymous requests get 503s once more than one large fetch is in flight.

### Why it streams

The lens is 6.6 GB and `lm_head` is 1.6 GB — together larger than this machine's
RAM. Nothing is ever fully resident:

- the lens is opened with `torch.load(..., mmap=True)` and touched one 100 MB
  layer at a time;
- `lm_head` is memory-mapped as raw bf16 and widened to fp32 in vocabulary
  blocks (bf16 is the top 16 bits of an fp32, so the widening is an exact shift,
  verified against `torch` bit-for-bit);
- every readout is batched into **one** streaming pass over the vocabulary
  rather than one pass per layer.

Only two tensors of Qwen3-32B are needed, and they sit in the last shard, so
`fetch_data.py` parses the safetensors header and range-requests just those
1.56 GB instead of the 3.0 GB shard (or the 65 GB model).

## Running it

```bash
python -m venv .venv && ./.venv/Scripts/pip install -r requirements.txt
```

```bash
./.venv/Scripts/python src/fetch_data.py
```

```bash
./.venv/Scripts/python src/run_analysis.py --top-k 25 && ./.venv/Scripts/python src/build_web.py
```

Then open `out/index.html` — it is self-contained, with the readout inlined.

Add `--role <name>` to read out any of the 275 roles in the dataset; the vector is
fetched on demand. `pirate`, for instance, runs colloquial (`kinda`, `gonna`, `booze`)
through the middle layers into nautical vocabulary (`ship`, `sails`, `船`, `航行`) by
layer 52, and lands on `ye`, `yer`, `sail`, `captain`, `plunder` at the output.

For the whole cast at once — 275 characters, both contrasts, one pass over the
vocabulary per contrast:

```bash
./.venv/Scripts/python src/run_sweep.py && ./.venv/Scripts/python src/build_web.py --page atlas
```

That writes `out/atlas.json` and `out/atlas.html`. It fetches the remaining role
vectors (~180 MB) on first run and takes roughly half an hour, almost all of it in the
two unembedding passes.

## Caveats

- A persona direction is not a residual the model ever actually held. The lens is
  linear, so this is a well-defined reading of the direction, but it is *not* the
  same experiment as steering a forward pass and reading the resulting
  activations — that is the natural follow-up.
- The lens is fitted on generic web text (wikitext-103), not on chat or
  role-play transcripts, and its Jacobian averages over positions 16..n-1.
- `bard − default` is a first-order contrast. It does not remove components
  shared with *other* role-play personas, only those shared with the default
  Assistant; some of what it surfaces will be generic "in character" signal
  rather than bard-specific. Reading out several roles and subtracting the role
  mean would separate the two.
- Probabilities are softmaxes over the full vocabulary of a readout that is not a
  real forward pass; compare them across layers and directions, not against a
  model's actual next-token distribution.

## Sources

| what | where |
| --- | --- |
| Role & axis vectors | [`lu-christina/assistant-axis-vectors`](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) (`qwen-3-32b`) |
| Fitted lens | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) (`qwen3-32b/jlens/Salesforce-wikitext`) |
| Unembedding & tokenizer | [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) |
| Lens reference code | [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0) |
| Axis reference code | [`safety-research/assistant-axis`](https://github.com/safety-research/assistant-axis) |
