"""
Text normalization utilities for contamination detection.

All evaluation instances and training corpus passages are normalized through
the same pipeline before comparison, following standard practices from
GPT-4 and Llama-3 contamination analyses.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(
    text: str,
    *,
    unicode_form: str = "NFKC",
    lowercase: bool = True,
    strip_whitespace: bool = True,
    strip_punctuation: bool = False,
    strip_accents: bool = False,
) -> str:
    """Apply a deterministic normalization pipeline to *text*.

    Parameters
    ----------
    text : str
        Raw input string.
    unicode_form : str
        Unicode normalization form (NFC, NFKC, NFD, NFKD).
    lowercase : bool
        Convert to lowercase.
    strip_whitespace : bool
        Collapse all whitespace runs to a single space and strip edges.
    strip_punctuation : bool
        Remove all characters in the Unicode *Punctuation* category.
    strip_accents : bool
        Remove combining diacritical marks (use with caution for Portuguese).

    Returns
    -------
    str
        Normalized text.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize(unicode_form, text)

    # 2. Optional accent removal (NFD decomposition + strip combining chars)
    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize(unicode_form, text)

    # 3. Lowercasing
    if lowercase:
        text = text.lower()

    # 4. Punctuation removal
    if strip_punctuation:
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "P")

    # 5. Whitespace normalization
    if strip_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def ngrams(text: str, n: int) -> list[str]:
    """Extract word-level n-grams from *text*.

    Parameters
    ----------
    text : str
        Already-normalized text.
    n : int
        N-gram size.

    Returns
    -------
    list[str]
        List of n-gram strings (space-joined tokens).
    """
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
