"""Text normalization and the JP/EN script filter shared by data build and eval."""

from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")

# Scripts we never train or evaluate on (see docs/plan-title-encoder.md scope).
_FORBIDDEN = re.compile(
    "["
    "Ͱ-Ͽ"  # Greek
    "Ѐ-ԯ"  # Cyrillic
    "֐-׿"  # Hebrew
    "؀-ۿݐ-ݿ"  # Arabic
    "ऀ-෿"  # Indic
    "฀-໿"  # Thai, Lao
    "ᄀ-ᇿ㄰-㆏가-힯"  # Hangul
    "㄀-ㄯㆠ-ㆿ"  # Bopomofo
    "]"
)
_KANA = re.compile("[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")
_LATIN = re.compile("[A-Za-zÀ-ɏ]")


def norm_key(text: str) -> str:
    """Dedup key: NFKC, casefold, collapsed whitespace."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def allowed(text: str) -> bool:
    return bool(text.strip()) and not _FORBIDDEN.search(text)


def script_class(text: str) -> str:
    """Coarse script of a title: 'kana' (any kana), 'han' (kanji, no kana),
    'latin', or 'other' (digits/symbols only)."""
    if _KANA.search(text):
        return "kana"
    if _HAN.search(text):
        return "han"
    if _LATIN.search(text):
        return "latin"
    return "other"


def cross_script(a: str, b: str) -> bool:
    ca, cb = script_class(a), script_class(b)
    cjk = {"kana", "han"}
    return (ca in cjk) != (cb in cjk) and "other" not in (ca, cb)
