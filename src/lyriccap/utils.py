from __future__ import annotations
import re
import unicodedata
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


# 라틴 문자에 붙는 결합 악센트(U+0300~U+036F)만 골라냅니다. 일본어 탁점
# (U+3099, U+309A)은 여기 들어가지 않으므로 が와 か는 계속 구분됩니다.
_LATIN_ACCENTS = re.compile(r"[̀-ͯ]")


def normalize_key(text: str) -> str:
    """곡 제목과 파일 이름을 비교할 때 쓰는 키.

    허용 문자를 a-z와 한글/일본어 범위로만 잡아 두어서, 악센트 글자가 통째로
    지워졌습니다. 그래서 "La dernière nacelle"은 'ladernirenacelle'이 되고
    악센트 없이 저장된 "La derniere nacelle.mp3"와 연결되지 않았습니다.
    이제는 악센트를 떼어 e로 만들어 두 표기가 같은 키가 됩니다.
    """
    text = unicodedata.normalize("NFKD", text)
    text = _LATIN_ACCENTS.sub("", text)
    text = unicodedata.normalize("NFC", text).casefold()
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
