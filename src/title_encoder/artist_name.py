"""Deterministic artist-name parse: strip platform wrappers, split into name fragments.

Library artist aliases carry source-specific wrappers that dominate the
embedding (`Takane Lui - Topic` lands next to every other `- Topic` channel):

- YouTube: `X - Topic`, `XVEVO`, `X Official`, `X channel` / `Xチャンネル`,
  `A Ch. B - Group`, `name【affiliation】`, `name / romaji`.
- Discogs: `X (2)` disambiguation, `X様`.
- SoundCloud: `X (All)`.

`fragments()` returns the cleaned name parts; an item is matched by its best
fragment pair. Affiliation fragments (`hololive`, `にじさんじ`, `FLOW GLOW`)
can't be recognized from one name alone: `affiliations()` finds the fragments
that sit next to many different names in a corpus, and callers drop them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from .text import norm_key, script_class

_TOPIC = re.compile(r"\s*-\s*Topic\s*$")
_VEVO = re.compile(r"(?:Official)?VEVO$")
_TRAIL = re.compile(r"\s*(?:\((?:All|\d+)\)|様)\s*$")
_CV = re.compile(r"[(（]\s*CV\s*[.:：．]?\s*([^)）]*)[)）]", re.IGNORECASE)
_AT = re.compile(r"@.*$")
# Separators between name parts. `Ch.` is hololive-style `Given Ch. Full Name`.
_SEP = re.compile(r"\s+/\s*|\s*/\s+|//|\s+[-ー]\s+|\s*[‐–—]\s*|\s*-\s*$|^\s*-\s*"
                  r"|[【】《》『』「」\[\]()（）|｜\x1f]|\s*(?<![A-Za-z])[Cc]h\.\s*")
_WRAP = re.compile(
    r"(?i)official\s+youtube\s+(?:channel|チャンネル)|youtube\s+(?:channel|チャンネル)"
    r"|official\s+channel|公式チャンネル|(?<![a-z])official(?![a-z])|official$|公式"
    r"|(?<![a-z])channel(?![a-z])|channel$|チャンネル|ちゃんねる|所属"
)
_EDGE = re.compile(r"^[\s\-‐ー–—☆★♪・.,:;~〜]+|[\s\-‐ー–—☆★♪・,:;~〜]+$")


def _clean(text: str) -> str:
    return _EDGE.sub("", re.sub(r"\s+", " ", text)).strip()


def fragments(name: str) -> list[str]:
    """Cleaned name parts, in order. Falls back to `[name]` if nothing survives."""
    s = _TOPIC.sub("", name)
    for _ in range(2):
        s = _TRAIL.sub("", s)
    s = _VEVO.sub("", s)
    # `キャラ (CV:声優)`: the voice actor becomes a later fragment.
    s = _CV.sub(lambda m: "\x1f" + m.group(1) + "\x1f", s)
    s = _AT.sub("", s) or s
    # Wrapper words also separate (`Mel Channel 夜空メル`, `ひま食堂channel -HIMASHOKUDO`).
    s = _WRAP.sub("\x1f", s)
    parts = []
    for raw in _SEP.split(s):
        if not raw:
            continue
        # A bare `/` splits only across scripts (`キノシタ/kinoshita`), not `AC/DC`.
        sub = raw.split("/")
        if len(sub) == 2 and script_class(sub[0]) != script_class(sub[1]):
            parts.extend(sub)
        else:
            parts.append(raw)
    out, seen = [], set()
    for p in parts:
        p = _clean(p)
        key = norm_key(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out or [name.strip() or name]


def affiliations(names: Iterable[str], min_names: int = 3) -> set[str]:
    """norm_keys of fragments that trail >= min_names distinct leading names.

    Only non-first positions count: an affiliation follows the name
    (`Koyori ch. 博衣こより - holoX -`), so a name that merely leads many
    variants (`Koyori ...`) isn't one."""
    partners: dict[str, set[str]] = defaultdict(set)
    for name in names:
        keys = [norm_key(f) for f in fragments(name)]
        for k in keys[1:]:
            if k != keys[0]:
                partners[k].add(keys[0])
    return {k for k, v in partners.items() if len(v) >= min_names}


def name_fragments(name: str, drop: set[str]) -> list[str]:
    """fragments() minus affiliation fragments (keeps the first if all are dropped)."""
    frags = fragments(name)
    kept = [f for f in frags if norm_key(f) not in drop]
    return kept or frags[:1]


def platform_strip(name: str) -> str:
    """The lossless subset: suffixes that are fixed platform conventions, not
    language (YouTube `- Topic` / `VEVO`, Discogs `(n)` / `様`, SoundCloud
    `(All)`). Everything fuzzier is left to the encoder (v7)."""
    s = _TOPIC.sub("", name)
    for _ in range(2):
        s = _TRAIL.sub("", s)
    s = _VEVO.sub("", s).strip()
    return s or name
