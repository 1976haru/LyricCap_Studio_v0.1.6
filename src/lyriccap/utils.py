from __future__ import annotations
import re
import unicodedata
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^a-z0-9가-힣ぁ-んァ-ン一-龯]+", "", text)
    return text


def strip_section_tag(line: str) -> str:
    line = line.strip()
    if re.fullmatch(r"\[[^\]]+\]", line):
        return ""
    return line


def clean_lyrics(raw: str) -> list[str]:
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = strip_section_tag(line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def discover_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )
