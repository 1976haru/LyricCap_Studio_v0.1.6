from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyriccap.parsers import parse_lyrics_file
from lyriccap.srt import srt_time, render
from lyriccap.models import Cue


def test_json():
    songs = parse_lyrics_file(ROOT / "examples" / "sample.json")
    assert len(songs) == 2
    assert songs[0].lyrics[0] == "The morning opens wide"


def test_txt():
    songs = parse_lyrics_file(ROOT / "examples" / "sample_multi.txt")
    assert len(songs) == 2


def test_srt():
    assert srt_time(1.234) == "00:00:01,234"
    cue = Cue(1, 2, "Hello", "en", en="Hello", ko="안녕", ja="こんにちは")
    text = render([cue], "en_ko")
    assert "Hello\n안녕" in text
    text2 = render([cue], "en_ja")
    assert "Hello\nこんにちは" in text2


def test_non_english_source():
    cue = Cue(1, 2, "안녕", "ko", en="Hello", ko="안녕", ja="こんにちは")
    assert "Hello\n안녕" in render([cue], "en_ko")


if __name__ == "__main__":
    test_json(); test_txt(); test_srt(); test_non_english_source(); print("OK")


def test_translator_pair_installed_handles_translation_objects(tmp_path, monkeypatch):
    """Regression: IdentityTranslation objects do not have a .code attribute."""
    from lyriccap.translator import ArgosTranslator

    class FakeLanguage:
        def __init__(self, code):
            self.code = code
            self.translations_from = []
            self.translations_to = []
        def get_translation(self, to):
            for tr in self.translations_from:
                if tr.to_lang is to:
                    return tr
            return None

    class IdentityTranslation:
        def __init__(self, lang):
            self.from_lang = lang
            self.to_lang = lang

    class PackageTranslation:
        def __init__(self, src, dst):
            self.from_lang = src
            self.to_lang = dst

    en = FakeLanguage('en')
    ko = FakeLanguage('ko')
    ident = IdentityTranslation(en)
    en_ko = PackageTranslation(en, ko)
    en.translations_from = [ident, en_ko]
    en.translations_to = [ident]
    ko.translations_to = [en_ko]

    class FakeTranslate:
        @staticmethod
        def get_installed_languages():
            return [en, ko]
        @staticmethod
        def get_translation_from_codes(src, dst):
            if src == 'en' and dst == 'ko':
                return en_ko
            raise ValueError('not installed')
        @staticmethod
        def get_language_from_code(code):
            return {'en': en, 'ko': ko}.get(code)

    tr = ArgosTranslator(tmp_path / 'cache.json')
    monkeypatch.setattr(tr, '_imports', lambda: (object(), FakeTranslate))
    assert tr._pair_installed('en', 'ko') is True
    assert tr._pair_installed('ko', 'en') is False
    assert tr._pair_installed('en', 'en') is True


def test_gemini_translation_parser():
    from lyriccap.translator import GeminiNaturalTranslator
    raw = '{"translations":[{"id":1,"text":"자연스러운 첫 줄"},{"id":2,"text":"자연스러운 둘째 줄"}]}'
    out = GeminiNaturalTranslator._parse_translations(raw, 2)
    assert out == ["자연스러운 첫 줄", "자연스러운 둘째 줄"]


def test_gemini_translation_parser_reorders_ids():
    from lyriccap.translator import GeminiNaturalTranslator
    raw = '```json\n{"translations":[{"id":2,"text":"둘째"},{"id":1,"text":"첫째"}]}\n```'
    out = GeminiNaturalTranslator._parse_translations(raw, 2)
    assert out == ["첫째", "둘째"]


def test_aligner_uses_real_segment_word_times():
    from types import SimpleNamespace
    from lyriccap.aligner import StableTSAligner

    w1 = SimpleNamespace(word='Hello', start=4.2, end=4.8, probability=0.9)
    w2 = SimpleNamespace(word='world', start=4.9, end=5.5, probability=0.9)
    w3 = SimpleNamespace(word='Stay', start=9.1, end=9.6, probability=0.9)
    w4 = SimpleNamespace(word='near', start=9.7, end=10.1, probability=0.9)
    result = SimpleNamespace(segments=[
        SimpleNamespace(start=4.0, end=5.7, text='Hello world', words=[w1, w2]),
        SimpleNamespace(start=9.0, end=10.3, text='Stay near', words=[w3, w4]),
    ])
    aligner = StableTSAligner()
    cues, rebuilt = aligner._cues_from_result(result, ['Hello world', 'Stay near'], 'en')
    assert rebuilt is False
    assert cues[0].start == 4.2 and cues[0].end == 5.5
    assert cues[1].start == 9.1 and cues[1].end == 10.1


def test_aligner_rebuild_uses_aligned_word_times_not_proportional_track_time():
    from types import SimpleNamespace
    from lyriccap.aligner import StableTSAligner

    words = [
        SimpleNamespace(word='Hello', start=12.0, end=12.4, probability=0.9),
        SimpleNamespace(word='world', start=12.5, end=12.9, probability=0.9),
        SimpleNamespace(word='Stay', start=30.0, end=30.4, probability=0.9),
        SimpleNamespace(word='near', start=30.5, end=31.0, probability=0.9),
    ]
    result = SimpleNamespace(segments=[SimpleNamespace(start=0, end=100, text='merged', words=words)])
    aligner = StableTSAligner()
    cues, rebuilt = aligner._cues_from_result(result, ['Hello world', 'Stay near'], 'en')
    assert rebuilt is True
    assert cues[0].start == 12.0
    assert cues[0].end == 12.9
    assert cues[1].start == 30.0
    assert cues[1].end == 31.0


def test_aligner_never_falls_back_to_fake_even_timing():
    from types import SimpleNamespace
    from lyriccap.aligner import StableTSAligner, AlignmentError

    result = SimpleNamespace(segments=[])
    aligner = StableTSAligner()
    try:
        aligner._cues_from_result(result, ['one', 'two'], 'en')
    except AlignmentError:
        pass
    else:
        raise AssertionError('Expected AlignmentError instead of proportional fake timing')
