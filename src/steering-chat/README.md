# Steering chat

Start an interactive Qwen3-32B chat with direct persona-vector additions:

```bash
uv run steering-chat /bard "Hey, bard, what's your name?"
```

Use `-neutral` to subtract the raw default vector automatically on every turn:

```bash
uv run steering-chat -neutral /bard "Hey, bard, what's your name?"
```

In this mode, the example above applies `bard - default_vector`; later untagged
messages still apply `-default_vector`.

Tags must precede the message and apply only to that turn. `/bard` adds the raw
published Bard vector, `/-bard` subtracts it, and `/5bard /-2default_vector`
applies `5 * bard - 2 * default_vector` at the middle transformer block. The
reserved names are `default_vector` (also `default`) and `assistant_axis`; other
names are resolved as published role or trait vectors and downloaded on demand.
For example, `/evil` uses `role_vectors/evil.pt` when present and otherwise
falls back to `trait_vectors/evil.pt`; a role wins if both have the same name.
Enter `/stop` to unload the model and exit.

Responses default to 300 visible tokens. After each response, the CLI prints
top-10 J-Lens tokens for the mean, first, and last post-steering output-token
residuals. Missing J-Lens assets are downloaded automatically (they require
several GB); use `--no-j-lens` to skip them. Run `uv run steering-chat --help`
for model, cache, quantization, and path options.
