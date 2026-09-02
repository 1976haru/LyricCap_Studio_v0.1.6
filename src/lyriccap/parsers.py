from __future__ import annotations
import json
import re
from pathlib import Path
from .models import Song
from .utils import clean_lyrics

# 가사가 비어 직전 파싱에서 제외된 곡 목록. app.py가 사용자에게 보여줍니다.
SKIPPED_TRACKS: list[str] = []

LANG_MAP = {
    "english": "en", "en": "en",
    "korean": "ko", "ko": "ko", "한국어": "ko",
    "japanese": "ja", "ja": "ja", "일본어": "ja",
}


def parse_lyrics_file(path: Path, source_language: str = "auto") -> list[Song]:
    SKIPPED_TRACKS.clear()
    if path.suffix.lower() == ".json":
        return parse_json(path, source_language)
    if path.suffix.lower() in {".txt", ".text"}:
        return parse_txt(path, source_language)
    raise ValueError("지원 형식은 JSON 또는 TXT입니다.")


def parse_json(path: Path, source_language: str = "auto") -> list[Song]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    meta_lang = "en"
    if isinstance(data, dict):
        raw_lang = str(data.get("meta", {}).get("lyricLanguage", "english")).lower()
        meta_lang = LANG_MAP.get(raw_lang, "en")
    lang = meta_lang if source_language == "auto" else source_language

    songs_data = data.get("songs") if isinstance(data, dict) else None
    if isinstance(songs_data, list):
        songs: list[Song] = []
        for i, item in enumerate(songs_data, 1):
            if not isinstance(item, dict):
                continue
            raw = item.get("lyrics", "")
            lines = clean_lyrics(raw if isinstance(raw, str) else "\n".join(map(str, raw)))
            if not lines:
                # 조용히 건너뛰면 곡이 사라진 걸 눈치채기 어렵습니다.
                no = item.get("trackNo") or i
                title = str(item.get("title") or f"Track {i:02d}")
                SKIPPED_TRACKS.append(f"{no}. {title}")
                continue
            songs.append(Song(
                track_no=int(item.get("trackNo") or i),
                title=str(item.get("title") or f"Track {i:02d}"),
                localized_title=str(item.get("titleLocalized") or ""),
                lyrics=lines,
                source_language=lang,
            ))
        if songs:
            return songs

    # Flexible single-song JSON fallbacks
    if isinstance(data, dict):
        raw = data.get("lyrics") or data.get("text") or data.get("lyric")
        if raw:
            lines = clean_lyrics(raw if isinstance(raw, str) else "\n".join(map(str, raw)))
            return [Song(1, str(data.get("title") or path.stem), lines, lang)]

    raise ValueError("JSON에서 songs[].lyrics 또는 lyrics/text 필드를 찾지 못했습니다.")


def parse_txt(path: Path, source_language: str = "auto") -> list[Song]:
    text = path.read_text(encoding="utf-8-sig")
    lang = "en" if source_language == "auto" else source_language

    # Multi-track TXT format: lines like ### 01. Title or === 01 Title ===
    marker = re.compile(r"^(?:#{2,}|={2,})\s*(\d{1,3})[.)\-\s]+(.+?)(?:\s*=+)?$", re.M)
    matches = list(marker.finditer(text))
    if matches:
        songs: list[Song] = []
        for idx, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            lines = clean_lyrics(text[body_start:body_end])
            if lines:
                songs.append(Song(int(m.group(1)), m.group(2).strip(), lines, lang))
        if songs:
            return songs

    lines = clean_lyrics(text)
    if not lines:
        raise ValueError("TXT에 가사가 없습니다.")
    return [Song(1, path.stem, lines, lang)]
