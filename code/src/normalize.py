"""
Text normalization for business names and addresses.

Kept deliberately rule-based and language/country-agnostic: no hard-coded
country list, no external lookups. Works the same way regardless of which
country string appears on a record (US, India, France, or anything else in
the test set).

Changes over the original version, aimed at the noise patterns the Problem
Statement calls out explicitly:
- Unicode/accent normalization ("Café" / "Cafe" / "Café" all collapse to
  the same string) instead of the accented character just being stripped
  to a blank by the punctuation filter, which used to silently destroy
  part of the name/address for transliterated records.
- A few common multi-word legal-suffix phrases ("limited liability
  company" -> "llc") that the previous token-by-token mapping couldn't
  catch, since it only ever substituted one word at a time.
- Landmark-based references ("near SBI ATM", "opposite the mall") are
  explicitly called out in the Problem Statement as a noise pattern, and
  they're actively harmful to keep: the landmark itself is often described
  completely differently between two records for the SAME business, so
  keeping "near"/"opposite"/"behind"/etc. as tokens can make two addresses
  for the same place look LESS similar, not more. These are now dropped
  entirely rather than partially abbreviated and kept.
- A broader address-abbreviation table (sector/phase/cross/nagar-style
  Indian locality terms, PO/post office, etc.).
"""

import re
import unicodedata

# --- Legal-form / abbreviation tables -------------------------------------
# Order matters: longer / more specific tokens first so we don't clobber
# substrings of later replacements.
NAME_SUFFIX_MAP = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "company": "co", "&": "and", "llp": "llp",
    "llc": "llc", "pvt.": "pvt", "ltd.": "ltd", "inc.": "inc",
    "corp.": "corp", "co.": "co",
}

# Multi-word phrases collapsed BEFORE tokenization -- token-by-token
# mapping can't catch these since each word would map independently
# ("limited" -> "ltd", "liability" -> unchanged, "company" -> "co", never
# converging on "llc").
NAME_PHRASE_MAP = {
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "private limited": "pvt ltd",
    "public limited": "plc",
}

ADDRESS_ABBR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "court": "ct", "circle": "cir",
    "place": "pl", "square": "sq", "highway": "hwy", "apartment": "apt",
    "building": "bldg", "floor": "fl", "suite": "ste", "unit": "unit",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "sector": "sec", "phase": "ph", "cross": "x",
    "post office": "po",  # matched as a phrase below, before tokenization
}

# Landmark / relative-reference words: dropped entirely (see module
# docstring) rather than kept or abbreviated -- the landmark that follows
# is exactly the noisy, inconsistently-worded part of the address.
_LANDMARK_WORDS = {
    "near", "opposite", "opp", "behind", "beside", "adjacent", "next",
    "front", "above", "below", "besides", "close", "nearby",
}
_STOPWORDS_ADDR = {"the", "of", "in", "a", "an"} | _LANDMARK_WORDS

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b\d{4,6}\b")


def _strip_accents(text):
    """Fold accented/transliterated Latin characters to their plain-ASCII
    form (e.g. "café" -> "cafe", "Ecole" stays "Ecole"). Non-Latin scripts
    fall through unchanged rather than being mangled."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _replace_phrases(text, phrase_map):
    for phrase, repl in phrase_map.items():
        if phrase in text:
            text = text.replace(phrase, repl)
    return text


def _base_clean(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = _strip_accents(text)
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name):
    """Lowercase, strip accents/punctuation, collapse legal-suffix
    abbreviations (including a few multi-word phrases)."""
    text = _base_clean(name)
    text = _replace_phrases(text, NAME_PHRASE_MAP)
    tokens = text.split()
    out = [NAME_SUFFIX_MAP.get(tok, tok) for tok in tokens]
    return " ".join(out)


def normalize_address(address):
    """Lowercase, strip accents/punctuation, collapse street-type
    abbreviations, and drop landmark/relative-reference noise words."""
    text = _base_clean(address)
    text = _replace_phrases(text, {"post office": "po"})
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