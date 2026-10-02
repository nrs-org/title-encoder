"""Export JP/EN name groups from the local MusicBrainz mirror.

Writes the leakage inputs derived from the eval library (musiclib.db), runs
sql/export_mb.sql inside the MB db container (TEMP tables only), and copies
the CSVs back to data/title-encoder/<build>/raw/.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path

from .paths import build_dir

SQL = Path(__file__).resolve().parents[2] / "sql" / "export_mb.sql"
CONTAINER = "musicbrainz-docker-db-1"
MB_PREFIX = "https://musicbrainz.org/"


def library_inputs(db: Path, out: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    mbids = set()
    for (identifier,) in con.execute(
        "select identifier from entry_source where source = 'musicbrainz'"
    ):
        if identifier.startswith(MB_PREFIX):
            kind, _, gid = identifier[len(MB_PREFIX) :].partition("/")
            if len(gid) == 36:
                mbids.add((kind, gid))
    names = set()
    for (name,) in con.execute(
        """select al.name from entry_alias al
           join entry_source es on es.source = al.source and es.identifier = al.identifier
           join entry e on e.id = es.entry_id
           where e.entry_type = 'artist'"""
    ):
        # Postgres lower() and Python lower() agree for the scripts we keep.
        name = name.strip().lower()
        if len(name) >= 2 and "\t" not in name and "\n" not in name and "\\" not in name:
            names.add(name)
    con.close()
    (out / "lib_mbid.tsv").write_text(
        "".join(f"{k}\t{g}\n" for k, g in sorted(mbids)), encoding="utf-8"
    )
    (out / "lib_artist_names.tsv").write_text(
        "".join(f"{n}\n" for n in sorted(names)), encoding="utf-8"
    )
    return {"lib_mbid": len(mbids), "lib_artist_names": len(names)}


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--build", default="mb-v1")
    args = parser.parse_args()

    out = build_dir(args.build) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    stats = library_inputs(args.library_db, out)
    print("library inputs:", stats)

    run("docker", "exec", CONTAINER, "sh", "-c", "rm -rf /tmp/te && mkdir -p /tmp/te/out")
    for name in ("lib_mbid.tsv", "lib_artist_names.tsv"):
        run("docker", "cp", str(out / name), f"{CONTAINER}:/tmp/te/{name}")
    run("docker", "cp", str(SQL), f"{CONTAINER}:/tmp/te/export_mb.sql")

    start = time.time()
    run("docker", "exec", CONTAINER, "psql", "-U", "musicbrainz", "-d", "musicbrainz_db",
        "-X", "-q", "-f", "/tmp/te/export_mb.sql")
    elapsed = time.time() - start
    run("docker", "cp", f"{CONTAINER}:/tmp/te/out/.", str(out))
    run("docker", "exec", CONTAINER, "rm", "-rf", "/tmp/te")

    replication = subprocess.run(
        ["docker", "exec", CONTAINER, "psql", "-U", "musicbrainz", "-d", "musicbrainz_db",
         "-XAtc", "select current_replication_sequence from replication_control"],
        capture_output=True, text=True,
    ).stdout.strip()
    card = {
        "build": args.build,
        "library_db": str(args.library_db),
        "mb_replication_sequence": replication or None,
        "export_seconds": round(elapsed, 1),
        **stats,
        "counts": dict(
            line.split(",", 1) for line in (out / "counts.csv").read_text().splitlines()
        ),
    }
    (out.parent / "export_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
