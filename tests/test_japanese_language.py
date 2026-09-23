import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyriccap.languages import normalize_language, resolve_language, detect_language_from_text
from lyriccap.models import Cue
from lyriccap.parsers import parse_lyrics_file


def test_descriptor_japanese_dominant():
    assert normalize_language("Japanese-dominant with sparse English code-switch") == "ja"


def test_descriptor_chooses_first_dominant_language():
    assert normalize_language("Japanese dominant / sparse English") == "ja"
    assert normalize_language("English dominant / sparse Japanese") == "en"


def test_script_fallback_japanese():
    text = "窓ぎわに座る君がいた\n目が合っただけ I know"
    assert resolve_language("mixed multilingual lyrics", text) == "ja"


def test_script_fallback_korean():
    assert detect_language_from_text("창가에 앉은 너를 보았어\n오늘도 같은 자리") == "ko"


def test_japanese_json_preserves_exact_source(tmp_path):
    path = tmp_path / "jp.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "lyricLanguage": "Japanese-dominant with sparse English code-switch"
                },
                "songs": [
                    {
                        "trackNo": 1,
                        "title": "窓ぎわの君",
                        "lyrics": "[Verse 1]\n向かいじゃない　少し斜め\n窓ぎわに座る君がいた",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    song = parse_lyrics_file(path)[0]
    assert song.source_language == "ja"
    assert song.lyrics[0] == "向かいじゃない 少し斜め"

    cue = Cue(0, 1, song.lyrics[0], song.source_language)
    cue.set_text("ja", cue.source)
    assert cue.text("ja") == "向かいじゃない 少し斜め"


def test_unknown_meta_uses_lyrics_script(tmp_path):
    path = tmp_path / "jp_unknown_meta.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"lyricLanguage": "dominant language with code-switch"},
                "songs": [
                    {
                        "trackNo": 1,
                        "title": "試験",
                        "lyrics": "窓の反射に君の横顔\nまだ見ないふり",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert parse_lyrics_file(path)[0].source_language == "ja"


def test_japanese_txt_auto_detection(tmp_path):
    path = tmp_path / "jp.txt"
    path.write_text("窓の反射に君の横顔\nまだ見ないふり", encoding="utf-8")
    assert parse_lyrics_file(path)[0].source_language == "ja"
