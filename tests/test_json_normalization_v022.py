from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyriccap.job_queue import QueueJob, QueueManager
from lyriccap.parsers import SKIPPED_TRACKS, parse_lyrics_document, parse_lyrics_file


FIXTURE = ROOT / "tests" / "fixtures" / "tracks_schema_english.json"


def write_json(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "lyrics.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_tracks_schema_fixture_normalizes_fifteen_songs():
    parsed = parse_lyrics_document(FIXTURE, "auto")
    assert parsed.info.schema == "tracks"
    assert parsed.info.track_count == 15
    assert parsed.info.source_language == "en"
    assert len(parsed.songs) == 15
    assert parsed.songs[0].track_no == 1
    assert parsed.songs[0].title == "Leave the Radio On"
    assert parsed.songs[0].source_language == "en"
    assert "The radio had a silver dial" in parsed.songs[0].lyrics
    assert parsed.songs[1].track_no == 2
    assert not any(line.startswith("[") for line in parsed.songs[0].lyrics)


@pytest.mark.parametrize("schema", ["songs", "tracks", "items", "playlist", "entries"])
def test_supported_array_schemas(tmp_path, schema):
    path = write_json(tmp_path, {schema: [{"trackNo": "01", "title": "One", "lyrics": "Hello"}]})
    parsed = parse_lyrics_document(path)
    assert parsed.info.schema == schema
    assert parsed.songs[0].track_no == 1


@pytest.mark.parametrize("field", ["lyrics", "text", "lyric", "lyricText", "lyricsText"])
def test_lyrics_aliases_and_list_values(tmp_path, field):
    path = write_json(tmp_path, {"tracks": [{field: ["line one", "line two"]}]})
    song = parse_lyrics_file(path)[0]
    assert song.lyrics == ["line one", "line two"]
    assert song.title == "Track 01"


def test_number_title_and_localized_aliases(tmp_path):
    path = write_json(tmp_path, {"tracks": [
        {"trackNumber": "07", "songTitle": "Alias", "localizedTitle": "별칭", "lyrics": "Line"},
        {"track_no": "bad", "trackTitle": "Fallback", "lyrics": "Line"},
    ]})
    songs = parse_lyrics_file(path)
    assert (songs[0].track_no, songs[0].title, songs[0].localized_title) == (7, "Alias", "별칭")
    assert songs[1].track_no == 2


@pytest.mark.parametrize(("hint", "text", "expected"), [
    ("English sung lyrics 100%", "Hello again", "en"),
    ("French sung lyrics 100%", "Je garde ton nom", "fr"),
    ("Japanese lyrics", "さよなら、また明日", "ja"),
    ("한국어", "오늘도 너를 기다려", "ko"),
])
def test_document_language_hints(tmp_path, hint, text, expected):
    path = write_json(tmp_path, {"language": hint, "tracks": [{"lyrics": text}]})
    assert parse_lyrics_file(path, "auto")[0].source_language == expected


def test_track_language_precedes_document_and_manual_precedes_track(tmp_path):
    path = write_json(tmp_path, {"language": "English", "tracks": [
        {"sungLanguage": "French chanson", "lyrics": "Je garde ton nom"}
    ]})
    assert parse_lyrics_file(path, "auto")[0].source_language == "fr"
    assert parse_lyrics_file(path, "ja")[0].source_language == "ja"


def test_clear_script_conflict_overrides_metadata(tmp_path):
    path = write_json(tmp_path, {"language": "English", "tracks": [{"lyrics": "今日は空がきれいです"}]})
    assert parse_lyrics_file(path)[0].source_language == "ja"


def test_empty_track_is_skipped_but_all_empty_is_error(tmp_path):
    path = write_json(tmp_path, {"tracks": [
        {"trackNo": 1, "title": "Empty", "lyrics": ""},
        {"trackNo": 2, "title": "Good", "lyrics": "A real line"},
    ]})
    assert [song.title for song in parse_lyrics_file(path)] == ["Good"]
    assert SKIPPED_TRACKS == ["1. Empty"]
    empty = write_json(tmp_path, {"tracks": [{"title": "Empty", "lyrics": ""}]})
    with pytest.raises(ValueError, match="자막으로 사용할 가사를"):
        parse_lyrics_file(empty)


@pytest.mark.parametrize("field", ["lyrics", "text", "lyric"])
def test_single_song_json_remains_supported(tmp_path, field):
    path = write_json(tmp_path, {"title": "Single", field: "Only line"})
    assert parse_lyrics_file(path)[0].lyrics == ["Only line"]


def test_txt_parser_remains_supported(tmp_path):
    path = tmp_path / "lyrics.txt"
    path.write_text("### 01. First\nLine one\n### 02. Second\nLine two", encoding="utf-8")
    assert [song.title for song in parse_lyrics_file(path)] == ["First", "Second"]


def test_unrelated_list_is_not_treated_as_tracks_and_error_lists_keys(tmp_path):
    path = write_json(tmp_path, {"packageName": "x", "qa": [{"status": "ok"}]})
    with pytest.raises(ValueError, match="packageName, qa"):
        parse_lyrics_file(path)


def test_tracks_fixture_can_be_registered_as_queue_job(tmp_path):
    songs = parse_lyrics_file(FIXTURE)
    audio = tmp_path / "01.wav"
    audio.write_bytes(b"RIFF")
    job = QueueJob(lyrics_path=FIXTURE, output_folder=tmp_path, source_language="auto",
                   model_name="base", sync_mode="music_precise", fallback_enabled=True,
                   profiles=["en"], translation_engine="gemini", translation_model="test",
                   gap_seconds=0, title_seconds=0, refine_timestamps=False, audio_files=[audio])
    queue = QueueManager()
    queue.add(job)
    assert len(queue.jobs) == 1
    assert len(songs) == 15


def test_ui_load_and_match_accepts_tracks_and_displays_schema(monkeypatch):
    sys.path.insert(0, str(ROOT))
    import app

    class Variable:
        def __init__(self, value=""):
            self.value = value
        def get(self):
            return self.value
        def set(self, value):
            self.value = value

    class Tree:
        def get_children(self):
            return ()
        def delete(self, *_items):
            pass
        def insert(self, *_args, **_kwargs):
            pass

    view = object.__new__(app.App)
    view.lyrics_path = Variable(str(FIXTURE))
    view.audio_folder = Variable("")
    view.selected_audio_files = []
    view.source_lang = Variable("auto")
    view.detected_language = Variable()
    view.status = Variable()
    view.tree = Tree()
    view.songs = []
    view._apply_profile_recommendation = lambda _language: None
    monkeypatch.setattr(app, "match_audio", lambda songs, _audio: songs)
    view.load_and_match()
    assert len(view.songs) == 15
    assert "구조 tracks" in view.status.get()
    assert "원문 영어" in view.status.get()
