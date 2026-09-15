# Write-up

Final project paper, formatted with the official ACL style files
(<https://github.com/acl-org/acl-style-files>), as the course grading criteria require.

## Structure

```text
main.tex             Preamble, title/authors, and the section order
sections/            One file per paper section, in the required order:
  abstract.tex         Abstract (max. 150 words, includes the conclusion)
  introduction.tex     Motivation, research question, hypotheses
  methods.tex          Models, data, analyses: enough to replicate
  results.tex          Results and their interpretation
  discussion.tex       Reflection, limitations, future work
  conclusion.tex       Optional; can be merged into the discussion
  ai_usage.tex         AI usage declaration (placed after the references)
references.bib       Bibliography
figures/, tables/    Figure files and table sources
acl.sty, acl_natbib.bst  ACL style files (unmodified)
```

Each section file starts with a comment summarising what the grading criteria
expect from it.

## Build

```bash
latexmk -pdf main.tex
```

`latexmk -c` removes the intermediate files. Until the first `\cite` is added,
BibTeX finds no citations and LaTeX reports "perhaps a missing \item" for the
empty reference list; the PDF is still produced and the error disappears once
something is cited.
