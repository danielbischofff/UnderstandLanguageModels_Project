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
src/run_analysis.py   the driver: directions × modes × layers -> out/readout.json
src/build_web.py      inline the readout into the viewer -> out/index.html
web/template.html     the viewer, with a __DATA__ placeholder
out/readout.json      the result (committed; the page is rebuilt from it)
```

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
