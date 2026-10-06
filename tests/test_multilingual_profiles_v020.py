from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyriccap.aligners.base import AlignmentResult
from lyriccap.batch import default_profiles_for_source, process_songs, profiles_for_source, required_languages
from lyriccap.job_queue import QueueJob, QueueManager
from lyriccap.models import Cue, Song
from lyriccap.parsers import parse_lyrics_file
from lyriccap.srt import profile_text, render

sys.path.insert(0, str(ROOT))
from app import get_visible_profile_specs, recommended_profiles


def test_french_source_profiles_render_in_declared_order():
    cue = Cue(0, 1, "Je t'attendrai sous la pluie", "fr")
    cue.ja = "雨の中でも、あなたを待っている"
    cue.ko = "비 속에서도 당신을 기다릴게"
    assert profile_text(cue, "fr") == cue.source
    assert profile_text(cue, "fr_ja").splitlines() == [cue.source, cue.ja]
    assert profile_text(cue, "fr_ko").splitlines() == [cue.source, cue.ko]
    assert cue.source in render([cue], "fr")


def test_visible_profile_specs_expose_french_profiles_with_chanson_label():
    visible = dict(get_visible_profile_specs())
    assert visible["fr"] == "프랑스어(샹송)"
    assert "fr_ko" in visible
    assert visible["fr_ja"] == "프랑스어 + 일본어"


@pytest.mark.parametrize(("preset", "expected"), [
    ("일본 채널", {"fr_ja"}),
    ("한국 채널", {"fr_ko"}),
    ("원문만", {"fr"}),
])
def test_french_channel_preset_recommendations(preset, expected):
    assert recommended_profiles("fr", preset) == expected


def test_existing_english_channel_profiles_remain_supported():
    cue = Cue(0, 1, "Hello", "en", ko="안녕", ja="こんにちは")
    assert profile_text(cue, "en_ko") == "Hello\n안녕"
    assert profile_text(cue, "en_ja") == "Hello\nこんにちは"


@pytest.mark.parametrize(("profile", "expected"), [
    ("fr_ja", {"fr", "ja"}), ("fr_ko", {"fr", "ko"}),
    ("ja_en", {"ja", "en"}), ("ko_ja", {"ko", "ja"}),
])
def test_required_languages_is_generic(profile, expected):
    assert required_languages([profile]) == expected


def test_invalid_profile_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported subtitle profile"):
        required_languages(["fr_xx"])


def test_auto_detected_french_builds_french_ui_options(tmp_path):
    lyrics = tmp_path / "lyrics.json"
    lyrics.write_text(json.dumps({
        "meta": {"lyricLanguage": "French chanson"},
        "songs": [{"trackNo": 1, "title": "La nuit", "lyrics": ["La nuit garde ton nom"]}],
    }, ensure_ascii=False), encoding="utf-8")
    songs = parse_lyrics_file(lyrics, "auto")
    assert {song.source_language for song in songs} == {"fr"}
    assert profiles_for_source("fr")[:3] == ["fr", "fr_ko", "fr_ja"]
    assert {"ko", "ja"} <= set(profiles_for_source("fr"))
    assert default_profiles_for_source("fr") == ["fr", "fr_ko", "fr_ja"]


class GoodAligner:
    engine = "test-exact"

    def align(self, audio_path, lyric_lines, language, bounds=None):
        return AlignmentResult([Cue(i * 1.2, i * 1.2 + 1, text, language)
                                for i, text in enumerate(lyric_lines)], self.engine)


class RecordingTranslator:
    def __init__(self): self.calls = []

    def translate_multi(self, texts, source, targets, title=""):
        self.calls.append((source, tuple(targets)))
        values = {"ja": ["夜はまだ君の名を抱いている" for _ in texts],
                  "ko": ["밤은 아직도 당신의 이름을 간직하고 있어" for _ in texts]}
        return {target: values[target] for target in targets}


def _wav(path: Path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000)
        handle.writeframes(b"\0\0" * 24000)


def test_french_translates_directly_and_writes_track_and_playlist_files(tmp_path, monkeypatch):
    audio = tmp_path / "01.wav"; _wav(audio)
    song = Song(1, "Chanson", ["La nuit garde encore ton nom"], "fr", audio_path=audio)
    translator = RecordingTranslator()
    monkeypatch.setattr("lyriccap.batch.build_translator", lambda *args, **kwargs: translator)
    aligner = GoodAligner(); output = tmp_path / "output"
    process_songs([song], output, ["fr", "fr_ko", "fr_ja"], strict=False,
                  primary_aligner=aligner, retry_aligner=aligner, fallback_aligner=aligner,
                  title_seconds=0)
    assert translator.calls == [("fr", ("ja", "ko"))]
    assert all("en" not in targets for _source, targets in translator.calls)
    for profile in ("fr", "fr_ko", "fr_ja"):
        assert (output / "tracks" / f"01_Chanson_{profile}.srt").exists()
        assert (output / "combined" / f"PLAYLIST_{profile}.srt").exists()
    lines = (output / "combined" / "PLAYLIST_fr_ja.srt").read_text(encoding="utf-8-sig").splitlines()
    assert lines[2:4] == [song.lyrics[0], "夜はまだ君の名を抱いている"]


def test_queue_job_preserves_and_executes_independent_french_profile(tmp_path):
    lyrics = tmp_path / "lyrics.json"; audio = tmp_path / "01.wav"
    lyrics.write_text("{}", encoding="utf-8"); _wav(audio)
    job = QueueJob(lyrics_path=lyrics, output_folder=tmp_path, source_language="fr",
                   model_name="base", sync_mode="music_precise", fallback_enabled=True,
                   profiles=["fr_ja"], translation_engine="gemini",
                   translation_model="gemini-2.5-flash", gap_seconds=0, title_seconds=0,
                   refine_timestamps=False, audio_files=[audio])
    assert job.profiles == ["fr_ja"]
    seen = []
    report = QueueManager([job]).run(lambda queued, _progress: seen.append(list(queued.profiles)))
    assert seen == [["fr_ja"]]
    assert report["completed"] == 1
