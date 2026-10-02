"""Leak-free retrieval eval on the musiclib library.

Every entry with >= 2 distinct aliases yields queries: an alias of the entry
must retrieve a *different-text* alias of the same entry among all aliases of
that entry type (optionally padded with distractor titles). The training data
excludes every library entity and artist, so this is held out by construction.

Reports recall@k per entry type and per stratum (cross-script, video-vs-clean,
lexically hard).

--artist mode attaches each alias's own source's primary artist and keeps
same-title/different-artist aliases as separate pool items (covers no longer
collapse into one). A model given as `path@cond` then embeds
`title [A] artist`; plain models still embed the title alone. Absolute numbers are a lower bound: the library still has
true duplicate entries, which count as misses.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .text import allowed, cross_script, norm_key

VIDEO_SOURCES = {"youtube", "nicovideo", "soundcloud"}
ENTRY_TYPES = ("track", "artist", "release", "release_group")
KS = (1, 5, 10, 20, 50)


@dataclass
class Pool:
    texts: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    entries: list[set[int]] = field(default_factory=list)
    artists: list[str | None] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)

    def add(self, text: str, entry: int | None, artist: str | None = None,
            by_artist: bool = False) -> int:
        key = norm_key(text) + (f"\x00{norm_key(artist or '')}" if by_artist else "")
        i = self.index.get(key)
        if i is None:
            i = self.index[key] = len(self.texts)
            self.texts.append(text)
            self.keys.append(key)
            self.entries.append(set())
            self.artists.append(artist)
        if entry is not None:
            self.entries[i].add(entry)
        return i


@dataclass
class Query:
    pool_idx: int
    entry: int
    entry_type: str
    strata: tuple[str, ...]


def trigram_jaccard(a: str, b: str) -> float:
    ga = {a[i : i + 3] for i in range(max(1, len(a) - 2))}
    gb = {b[i : i + 3] for i in range(max(1, len(b) - 2))}
    return len(ga & gb) / max(1, len(ga | gb))


def pair_artists(con: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Primary artist name per (source, identifier): main listed_artist at
    index 0 first, then uploader, then any main credit; primary alias first."""
    best: dict[tuple[str, str], tuple[tuple, str]] = {}
    for source, ident, role, extra, name, primary in con.execute(
        """select c.source, c.identifier, c.role, c.extra, a.name, a."primary"
           from contribution c
           join entry_alias a on a.source = c.artist_source and a.identifier = c.artist_identifier
           where c.main_artist = 1"""
    ):
        try:
            meta = json.loads(extra or "{}")
        except ValueError:
            meta = None
        index = meta.get("index", 9) if isinstance(meta, dict) else 9
        rank = (0 if role == "listed_artist" else 1 if role == "uploader" else 2, index,
                0 if primary else 1)
        key = (source, ident)
        if allowed(name) and (key not in best or rank < best[key][0]):
            best[key] = (rank, name)
    return {k: v[1] for k, v in best.items()}


def load_library(db: Path, max_queries_per_entry: int, seed: int, artist_mode: bool = False):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        """select e.id, e.entry_type, al.name, al.source, al.identifier
           from entry e
           join entry_source es on es.entry_id = e.id
           join entry_alias al on al.source = es.source and al.identifier = es.identifier
           where e.entry_type in ('track', 'artist', 'release', 'release_group')"""
    ).fetchall()
    artists = pair_artists(con) if artist_mode else {}
    con.close()
    by_entry: dict[int, dict[str, tuple[str, set[str], str | None]]] = defaultdict(dict)
    types: dict[int, str] = {}
    for entry, etype, name, source, ident in rows:
        if not allowed(name):
            continue
        artist = artists.get((source, ident)) if etype == "track" else None
        key = norm_key(name) + (f"\x00{norm_key(artist or '')}" if artist_mode else "")
        if not norm_key(name):
            continue
        types[entry] = etype
        text, sources, _ = by_entry[entry].setdefault(key, (name, set(), artist))
        sources.add(source)

    rng = random.Random(seed)
    pools = {t: Pool() for t in ENTRY_TYPES}
    queries: list[Query] = []
    for entry, aliases in by_entry.items():
        etype = types[entry]
        idx = {key: pools[etype].add(text, entry, artist, artist_mode)
               for key, (text, _, artist) in aliases.items()}
        if len(aliases) < 2:
            continue
        keys = sorted(aliases)
        rng.shuffle(keys)
        for key in keys[:max_queries_per_entry]:
            text, sources, _ = aliases[key]
            others = [aliases[k] for k in aliases if k != key]
            strata = ["all"]
            if all(cross_script(text, o[0]) for o in others):
                strata.append("cross_script")
            q_video = bool(sources & VIDEO_SOURCES) and not sources - VIDEO_SOURCES
            if q_video and any(o[1] - VIDEO_SOURCES for o in others):
                strata.append("video_to_clean")
            if max(trigram_jaccard(norm_key(text), norm_key(o[0])) for o in others) < 0.3:
                strata.append("lexically_hard")
            queries.append(Query(idx[key], entry, etype, tuple(strata)))
    return pools, queries


def add_distractors(pools: dict[str, Pool], path: Path | None, per_type: int, seed: int,
                    artist_mode: bool = False) -> None:
    """Pad pools with MB names (JSONL: {"entry_type", "text", "artist"?})."""
    if path is None or per_type <= 0:
        return
    by_type: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_type[row["entry_type"]].append((row["text"], row.get("artist")))
    rng = random.Random(seed)
    for etype, texts in by_type.items():
        if etype not in pools:
            continue
        rng.shuffle(texts)
        added = 0
        for text, artist in texts:
            if added >= per_type:
                break
            artist = artist if artist_mode else None
            key = norm_key(text) + (f"\x00{norm_key(artist or '')}" if artist_mode else "")
            if key not in pools[etype].index:
                pools[etype].add(text, None, artist, artist_mode)
                added += 1


def search(emb: np.ndarray, query_idx: np.ndarray, k: int, device: str) -> np.ndarray:
    """Top-(k+1) inner-product neighbours of emb[query_idx] within emb."""
    import torch

    pool = torch.from_numpy(emb).to(device=device, dtype=torch.float16 if device != "cpu" else torch.float32)
    out = []
    # Keep each score matrix around 256 MB regardless of pool size.
    chunk = max(16, (1 << 27) // max(1, len(emb)))
    for start in range(0, len(query_idx), chunk):
        q = pool[torch.from_numpy(query_idx[start : start + chunk]).to(device)]
        out.append(torch.topk(q @ pool.T, k=min(k + 1, len(emb)), dim=1).indices.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, k + 1), dtype=np.int64)


def evaluate(pools, queries, embed_fn, device: str) -> dict:
    kmax = max(KS)
    hits: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * len(KS))
    totals: dict[tuple[str, str], int] = defaultdict(int)
    ranks_by_type: dict[str, list[int]] = defaultdict(list)
    timing = {}
    for etype, pool in pools.items():
        qs = [q for q in queries if q.entry_type == etype]
        if not qs:
            continue
        start = time.time()
        emb = embed_fn(pool.texts, pool.artists)
        timing[etype] = {"texts": len(pool.texts), "embed_seconds": round(time.time() - start, 2)}
        nn = search(emb, np.array([q.pool_idx for q in qs]), kmax, device)
        for q, row in zip(qs, nn):
            rank = None
            pos = 0
            for j in row:
                if j == q.pool_idx:
                    continue
                pos += 1
                if q.entry in pool.entries[j]:
                    rank = pos
                    break
                if pos >= kmax:
                    break
            ranks_by_type[etype].append(rank or kmax + 1)
            for stratum in q.strata:
                for key in ((etype, stratum), ("*", stratum)):
                    totals[key] += 1
                    for i, k in enumerate(KS):
                        if rank is not None and rank <= k:
                            hits[key][i] += 1
    metrics = {
        f"{etype}/{stratum}": {
            "n": totals[(etype, stratum)],
            **{f"r@{k}": round(hits[(etype, stratum)][i] / totals[(etype, stratum)], 4)
               for i, k in enumerate(KS)},
        }
        for etype, stratum in sorted(totals)
    }
    return {"metrics": metrics, "timing": timing}


def make_embed_fn(spec: str, device: str, batch_size: int, truncate_dim: int | None = None):
    """`spec` is a model path/name, 'lexical', or `path@cond` for an
    artist-conditioned model (input `title [A] artist` when the artist is known)."""
    model_name, _, mode = spec.partition("@")
    conditioned = mode == "cond"

    def fmt(texts: list[str], artists: list[str | None] | None) -> list[str]:
        if not conditioned or artists is None:
            return list(texts)
        return [f"{t} [A] {a}" if a else t for t, a in zip(texts, artists)]

    if model_name == "lexical":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        def lexical(texts: list[str], artists=None) -> np.ndarray:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1,
                                  preprocessor=norm_key, dtype=np.float32)
            m = vec.fit_transform(texts)
            # Dense projection keeps search() uniform; SVD keeps it cheap.
            from sklearn.decomposition import TruncatedSVD

            dense = TruncatedSVD(n_components=256, random_state=0).fit_transform(m)
            return normalize(dense).astype(np.float32)

        return lexical

    from sentence_transformers import SentenceTransformer

    prefix = ""
    if model_name.startswith("intfloat/multilingual-e5"):
        prefix = "query: "
    model = SentenceTransformer(model_name, device=device, truncate_dim=truncate_dim)
    model.max_seq_length = 64

    def encode(texts: list[str], artists: list[str | None] | None = None) -> np.ndarray:
        return model.encode([prefix + t for t in fmt(texts, artists)], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="HF/sentence-transformers path, or 'lexical'")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--distractors", type=Path)
    parser.add_argument("--distractors-per-type", type=int, default=0)
    parser.add_argument("--max-queries-per-entry", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--truncate-dim", type=int,
                        help="keep the first N embedding dims (Matryoshka), renormalized")
    parser.add_argument("--artist", action="store_true",
                        help="attach per-source primary artists; keep covers as separate items")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pools, queries = load_library(args.library_db, args.max_queries_per_entry, args.seed, args.artist)
    add_distractors(pools, args.distractors, args.distractors_per_type, args.seed, args.artist)
    if args.artist:
        tracks = pools["track"]
        print(f"track pool items with an artist: {sum(a is not None for a in tracks.artists)}"
              f"/{len(tracks.artists)}")
    print("pool sizes:", {t: len(p.texts) for t, p in pools.items()}, "queries:", len(queries))

    report = {"library_db": str(args.library_db), "pool_sizes": {t: len(p.texts) for t, p in pools.items()},
              "queries": len(queries), "models": {}}
    for name in args.models:
        result = evaluate(pools, queries, make_embed_fn(name, args.device, args.batch_size, args.truncate_dim), args.device)
        report["models"][name] = result
        m = result["metrics"]
        print(f"\n== {name}")
        for key in sorted(m):
            if key.startswith("*/") or key.endswith("/all"):
                row = m[key]
                print(f"  {key:32s} n={row['n']:6d}  " +
                      "  ".join(f"r@{k}={row[f'r@{k}']:.3f}" for k in (1, 10, 20, 50)))
        import torch

        torch.cuda.empty_cache()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
