from __future__ import annotations
import re
from pathlib import Path
from .models import Song
from .utils import normalize_key


def _leading_track_no(name: str) -> int | None:
    m = re.match(r"^\s*0*(\d{1,3})(?:\D|$)", name)
    return int(m.group(1)) if m else None


def match_audio(songs: list[Song], audio_files: list[Path]) -> list[Song]:
    remaining = list(audio_files)
    # 1) track number
    for song in songs:
        for p in list(remaining):
            if _leading_track_no(p.stem) == song.track_no:
                song.audio_path = p
                remaining.remove(p)
                break

    # 2) title contains / normalized similarity
    for song in songs:
        if song.audio_path:
            continue
        skey = normalize_key(song.title)
        candidates = [p for p in remaining if skey and (skey in normalize_key(p.stem) or normalize_key(p.stem) in skey)]
        if candidates:
            song.audio_path = candidates[0]
            remaining.remove(candidates[0])

    # 3) remaining in order
    unassigned = [s for s in songs if not s.audio_path]
    for song, p in zip(unassigned, remaining):
        song.audio_path = p
    return songs
