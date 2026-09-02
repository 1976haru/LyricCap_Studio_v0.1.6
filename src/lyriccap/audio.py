from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _from_wave(path: Path) -> float:
    """표준 PCM WAV 헤더에서 길이를 읽습니다."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = float(w.getframerate())
            if rate > 0:
                return w.getnframes() / rate
    except Exception:
        pass
    return 0.0


def _from_soundfile(path: Path) -> float:
    """float32 WAV 등 wave 모듈이 못 읽는 형식을 처리합니다."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return 0.0


def _from_mutagen(path: Path) -> float:
    try:
        from mutagen import File as MutagenFile
        obj = MutagenFile(str(path))
        if obj is not None and getattr(obj, "info", None):
            length = float(obj.info.length)
            if length > 0:
                return length
    except Exception:
        pass
    return 0.0


def _from_ffprobe(path: Path) -> float:
    """마지막 보루. Demucs를 쓰는 환경이면 ffmpeg/ffprobe가 대개 함께 있습니다."""
    exe = shutil.which("ffprobe")
    if not exe:
        return 0.0
    try:
        proc = subprocess.run(
            [
                exe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            creationflags=NO_WINDOW,
            check=False,
        )
        value = float((proc.stdout or "").strip() or 0.0)
        return value if value > 0 else 0.0
    except Exception:
        return 0.0


def audio_duration(path: Path) -> float:
    """음원 길이를 초 단위로 반환합니다. 실패하면 0.0.

    기존 버전은 WAV 헤더와 mutagen만 시도한 뒤 조용히 0.0을 돌려줬습니다.
    통합 SRT의 곡 오프셋이 이 값의 누적합이기 때문에, 한 곡이라도 0이 되면
    그 뒤의 모든 자막이 통째로 밀립니다. 그래서 가능한 경로를 모두 시도합니다.
    """
    path = Path(path)
    if not path.exists():
        return 0.0

    if path.suffix.lower() == ".wav":
        order = (_from_wave, _from_soundfile, _from_ffprobe, _from_mutagen)
    else:
        order = (_from_mutagen, _from_soundfile, _from_ffprobe, _from_wave)

    for probe in order:
        value = probe(path)
        if value > 0:
            return value
    return 0.0


def audio_duration_detailed(path: Path) -> dict:
    """진단용. 각 방법이 각각 어떤 값을 냈는지 보여줍니다."""
    path = Path(path)
    return {
        "path": str(path),
        "wave": _from_wave(path),
        "soundfile": _from_soundfile(path),
        "mutagen": _from_mutagen(path),
        "ffprobe": _from_ffprobe(path),
        "resolved": audio_duration(path),
    }
