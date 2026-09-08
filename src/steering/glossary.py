"""Offline Chinese dictionary glosses for display, never changes to lens scores."""

from __future__ import annotations

import gzip
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping


HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
KANA = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
ENTRY = re.compile(r"^(\S+) (\S+) \[[^\]]+\] /(.+)/$")


@lru_cache(maxsize=1)
def dictionary() -> dict[str, str]:
    """Keep a few dictionary senses, merging alternate readings of each word."""
    raw = gzip.decompress(files("steering").joinpath("data/cedict.txt.gz").read_bytes())
    senses: dict[str, list[str]] = {}
    for line in raw.decode("utf-8").splitlines():
        match = ENTRY.match(line)
        if match is None:
            continue
        for word in {match[1], match[2]}:
            entries = senses.setdefault(word, [])
            for meaning in match[3].split("/"):
                meaning = re.sub(r"\s*\(CL:[^)]*\)", "", meaning).strip()
                if not meaning.startswith("CL:") and meaning not in entries:
                    entries.append(meaning)
    return {word: "; ".join(entries[:3]) for word, entries in senses.items() if entries}


def gloss(text: str, words: Mapping[str, str]) -> dict[str, str] | None:
    """Exact word lookup first; segmented fallbacks explicitly remain approximate."""
    text = text.strip()
    if not HAN.search(text):
        return None
    if KANA.search(text) or "\ufffd" in text:
        return {"kind": "untranslated", "text": "untranslated: mixed script or byte fragment"}
    if text in words:
        return {"kind": "dictionary", "text": words[text]}
    parts = []
    index = 0
    found = False
    while index < len(text):
        if not HAN.fullmatch(text[index]):
            parts.append(text[index])
            index += 1
            continue
        # Token strings are short. Longest dictionary match is deterministic,
        # but does not establish a context-sensitive translation of the whole.
        end = next((end for end in range(len(text), index, -1) if text[index:end] in words), index + 1)
        word = text[index:end]
        meaning = words.get(word)
        found = found or meaning is not None
        parts.append(f"{word}: {meaning or 'unknown'}")
        index = end
    return {
        "kind": "parts" if found else "untranslated",
        "text": "approx. parts — " + " | ".join(parts) if found else "untranslated: no dictionary entry",
    }


def report_glossary(document: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Embed only tokens present in this report, keeping the results immutable."""
    tokens: set[str] = set()
    for trial in document.get("trials", []):
        lens = trial["j_lens"]
        tokens.update(token["text"].strip() for token in lens["tokens"])
        for layer in lens["layers"]:
            for position in layer["positions"]:
                tokens.update(token["text"].strip() for token in position)
    words = dictionary() if any(HAN.search(token) for token in tokens) else {}
    return {token: value for token in sorted(tokens) if (value := gloss(token, words)) is not None}
