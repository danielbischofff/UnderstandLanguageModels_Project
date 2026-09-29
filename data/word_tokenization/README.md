# Word-token mappings

Generated from both question banks using the tokenizer revision pinned in
`configs/experiments/question-bank-pilot.yaml` and the existing
`quantitive_analysis.contrastive.resolve_word` function.

- `word_tokenization.csv`: one row per source word occurrence, including all
  tested forms, token IDs, tokenizer pieces, decoded pieces, selected token,
  complete/proxy status, and collisions with other distinct words.
- `word_tokenization.json`: the same rows plus model/tokenizer provenance,
  tokenizer file hashes, source-bank hashes, and resolver source hash.
- Human-readable summary: `outputs/analysis/word-tokenization/summary.md`.

Reproduce from the repository root, without a GPU or network:

```bash
PYTHONPATH=src runtime/venv/bin/python -m quantitive_analysis.tokenization_audit \
  --config configs/experiments/question-bank-pilot.yaml \
  --banks configs/question_bank.yaml configs/question_bank_contrastive.yaml
```

This replaces the generated audit files. The source banks remain unchanged.
The validated mapping files and audit summary are explicitly versioned snapshots.
Other data/ and outputs/ artifacts remain ignored by Git. The generator lives in
src/quantitive_analysis/tokenization_audit.py.

Token strings such as a leading-space marker are raw tokenizer pieces;
`selected_token_decoded` is the decoded token. The resolver's legacy `token_text`
field contains the surface form even for proxies, so this audit explicitly
uses the selected ID to obtain the actual piece and its decoded representation.

The analysis CLI does not yet consume this audit or filter incomplete words.
It continues to run the same resolver itself. Verify provenance before reusing
these mappings with another tokenizer, word bank, or resolver version.
