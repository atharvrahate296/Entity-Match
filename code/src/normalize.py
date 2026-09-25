"""
Text normalization for business names and addresses.

Kept deliberately rule-based and language/country-agnostic: no hard-coded
country list, no external lookups. Works the same way regardless of which
country string appears on a record (US, India, France, or anything else in
the test set).
"""

import re

# --- Legal-form / abbreviation tables -------------------------------------
# Order matters: longer / more specific tokens first so we don't clobber
# substrings of later replacements.
NAME_SUFFIX_MAP = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "company": "co", "&": "and", "llp": "llp",
    "llc": "llc", "pvt.": "pvt", "ltd.": "ltd", "inc.": "inc",
    "corp.": "corp", "co.": "co",
}

ADDRESS_ABBR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "court": "ct", "circle": "cir",
    "place": "pl", "square": "sq", "highway": "hwy", "apartment": "apt",
    "building": "bldg", "floor": "fl", "suite": "ste", "unit": "unit",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "near": "near", "opposite": "opp",
}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b\d{4,6}\b")
_STOPWORDS_ADDR = {"the", "of", "in", "a", "an"}


def _base_clean(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name):
    """Lowercase, strip punctuation, collapse common legal-suffix abbreviations."""
    text = _base_clean(name)
    tokens = text.split()
    out = []
    for tok in tokens:
        out.append(NAME_SUFFIX_MAP.get(tok, tok))
    return " ".join(out)


def normalize_address(address):
    """Lowercase, strip punctuation, collapse common street-type abbreviations."""
    text = _base_clean(address)
    tokens = text.split()
    out = []
    for tok in tokens:
        tok = ADDRESS_ABBR_MAP.get(tok, tok)
        if tok not in _STOPWORDS_ADDR:
            out.append(tok)
    return " ".join(out)


def extract_postal_code(raw_address):
    """Best-effort postal/PIN/ZIP code extraction (4-6 digit token), or None."""
    if raw_address is None:
        return None
    matches = _POSTAL_RE.findall(str(raw_address))
    return matches[-1] if matches else None


def token_set(text):
    """Return the set of tokens in an already-normalized string."""
    return set(text.split()) if text else set()
