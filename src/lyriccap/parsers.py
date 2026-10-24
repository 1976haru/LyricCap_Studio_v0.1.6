from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .languages import LANG_ALIASES, detect_language_from_text, normalize_language, resolve_language
from .models import Song
from .utils import clean_lyrics_with_metadata, clean_lyrics_with_structure

SKIPPED_TRACKS: list[str] = []
LANG_MAP = LANG_ALIASES

ARRAY_KEYS = ("songs", "tracks", "items", "playlist", "entries")
LYRICS_KEYS = ("lyrics", "text", "lyric", "lyricText", "lyricsText")
TRACK_NUMBER_KEYS = ("trackNo", "trackNumber", "track_no", "number", "no", "index")
TITLE_KEYS = ("title", "songTitle", "trackTitle", "name")
LOCALIZED_TITLE_KEYS = ("titleLocalized", "localizedTitle", "title_localized")
TRACK_LANGUAGE_KEYS = ("sungLanguage", "lyricLanguage", "language", "sourceLanguage")


@dataclass(frozen=True)
class NormalizedLyricsDocument:
    items: list[dict[str, Any]]
    schema: str
    language_hint: str
    top_level_keys: tuple[str, ...]


@dataclass(frozen=True)
class ParseInfo:
    schema: str
    source_language: str
    track_count: int


@dataclass(frozen=True)
class ParsedLyricsDocument:
    songs: list[Song]
    info: ParseInfo


def _first_value(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _lyrics_value(item: dict[str, Any]) -> Any:
    return _first_value(item, LYRICS_KEYS)


def _lyrics_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(part) for part in raw)
    return ""


def _document_language_hint(data: dict[str, Any]) -> str:
    meta = data.get("meta")
    if isinstance(meta, dict):
        for key in ("lyricLanguage", "language"):
            value = meta.get(key)
            if value is not None and str(value).strip():
                return str(value)
    for key in ("lyricLanguage", "language", "sourceLanguage", "sungLanguage"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _extract_track_items(data: Any) -> NormalizedLyricsDocument:
    if not isinstance(data, dict):
        return NormalizedLyricsDocument([], "unknown", "", ())
    keys = tuple(str(key) for key in data.keys())
    language_hint = _document_language_hint(data)
    for schema in ARRAY_KEYS:
        candidate = data.get(schema)
        if not isinstance(candidate, list):
            continue
        items = [item for item in candidate if isinstance(item, dict)]
        # Presence, rather than truthiness, lets an all-empty lyric collection
        # produce a useful skipped/error report without accepting arbitrary lists.
        if any(any(key in item for key in LYRICS_KEYS) for item in items):
            return NormalizedLyricsDocument(items, schema, language_hint, keys)
    if any(key in data for key in LYRICS_KEYS):
        return NormalizedLyricsDocument([data], "single", language_hint, keys)
    return NormalizedLyricsDocument([], "unknown", language_hint, keys)


def _track_number(item: dict[str, Any], position: int) -> int:
    raw = _first_value(item, TRACK_NUMBER_KEYS)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return position


def _title(item: dict[str, Any], position: int) -> str:
    raw = _first_value(item, TITLE_KEYS)
    return str(raw).strip() if raw is not None and str(raw).strip() else f"Track {position:02d}"


def _localized_title(item: dict[str, Any]) -> str:
    raw = _first_value(item, LOCALIZED_TITLE_KEYS)
    return str(raw).strip() if raw is not None else ""


def _resolve_song_language(source_language: str, track_hint: str, document_hint: str, text: str) -> str:
    if source_language != "auto":
        return normalize_language(source_language, source_language)
    hinted = resolve_language(track_hint or document_hint, text, default="en")
    # Kana/Hangul are stronger evidence than conflicting Latin metadata.
    scripted = detect_language_from_text(text, default="")
    return scripted if scripted and scripted != hinted else hinted


def _no_lyrics_error(document: NormalizedLyricsDocument) -> ValueError:
    detected = ", ".join(document.top_level_keys) or "(없음)"
    return ValueError(
        "JSON에서 자막으로 사용할 가사를 찾지 못했습니다.\n"
        "지원 구조:\n- songs[].lyrics\n- tracks[].lyrics\n- 단일곡 lyrics/text/lyric\n\n"
        "선택한 JSON이 Suno/음악 생성용 파일이라면\n"
        "각 곡에 title과 lyrics가 있는지 확인하세요.\n\n"
        f"감지된 키: {detected}"
    )


def parse_lyrics_document(path: Path, source_language: str = "auto") -> ParsedLyricsDocument:
    SKIPPED_TRACKS.clear()
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        songs = parse_txt(path, source_language)
        languages = {song.source_language for song in songs}
        language = next(iter(languages)) if len(languages) == 1 else "mixed"
        return ParsedLyricsDocument(songs, ParseInfo("txt", language, len(songs)))
    if suffix != ".json":
        raise ValueError("지원 형식은 JSON 또는 TXT입니다.")

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    document = _extract_track_items(data)
    if not document.items:
        raise _no_lyrics_error(document)

    songs: list[Song] = []
    for position, item in enumerate(document.items, 1):
        raw_text = _lyrics_text(_lyrics_value(item))
        # Keep the long-standing cleanup contract as the source of lyric lines;
        # the structural pass additionally preserves section/speaker details.
        lines, instrumental_after = clean_lyrics_with_metadata(raw_text)
        _structured_lines, _structured_breaks, speaker_roles, sections = clean_lyrics_with_structure(raw_text)
        track_no, title = _track_number(item, position), _title(item, position)
        if not lines:
            SKIPPED_TRACKS.append(f"{track_no}. {title}")
            continue
        raw_hint = _first_value(item, TRACK_LANGUAGE_KEYS)
        language = _resolve_song_language(
            source_language,
            str(raw_hint).strip() if raw_hint is not None else "",
            document.language_hint,
            raw_text,
        )
        songs.append(Song(
            track_no=track_no, title=title, lyrics=lines, source_language=language,
            localized_title=_localized_title(item), instrumental_after=instrumental_after,
            speaker_roles=speaker_roles, sections=sections,
        ))
    if not songs:
        raise _no_lyrics_error(document)
    languages = {song.source_language for song in songs}
    language = next(iter(languages)) if len(languages) == 1 else "mixed"
    return ParsedLyricsDocument(songs, ParseInfo(document.schema, language, len(songs)))


def parse_lyrics_file(path: Path, source_language: str = "auto") -> list[Song]:
    return parse_lyrics_document(path, source_language).songs


def parse_json(path: Path, source_language: str = "auto") -> list[Song]:
    return parse_lyrics_document(path, source_language).songs


def parse_txt(path: Path, source_language: str = "auto") -> list[Song]:
    text = path.read_text(encoding="utf-8-sig")
    lang = detect_language_from_text(text, default="en") if source_language == "auto" else normalize_language(source_language, source_language)
    marker = re.compile(r"^(?:#{2,}|={2,})\s*(\d{1,3})[.)\-\s]+(.+?)(?:\s*=+)?$", re.M)
    matches = list(marker.finditer(text))
    if matches:
        songs: list[Song] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            lines, instrumental_after, speaker_roles, sections = clean_lyrics_with_structure(text[match.end():end])
            if lines:
                songs.append(Song(int(match.group(1)), match.group(2).strip(), lines, lang,
                                  instrumental_after=instrumental_after, speaker_roles=speaker_roles,
                                  sections=sections))
        if songs:
            return songs
    lines, instrumental_after, speaker_roles, sections = clean_lyrics_with_structure(text)
    if not lines:
        raise ValueError("TXT에 가사가 없습니다.")
    return [Song(1, path.stem, lines, lang, instrumental_after=instrumental_after,
                 speaker_roles=speaker_roles, sections=sections)]
