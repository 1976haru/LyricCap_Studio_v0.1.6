from __future__ import annotations
import wave
from pathlib import Path


def audio_duration(path: Path) -> float:
    ext = path.suffix.lower()
    if ext == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass
    try:
        from mutagen import File as MutagenFile
        obj = MutagenFile(str(path))
        if obj is not None and getattr(obj, "info", None):
            return float(obj.info.length)
    except Exception:
        pass
    return 0.0
