from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyriccap.aligner import AlignmentError
from lyriccap.aligners.whisperx_exact_aligner import WhisperXExactAligner
from lyriccap.parsers import parse_lyrics_file


def chars(text, start=1.0):
    out = []
    time = start
    for char in text:
        out.append({"char": char, "start": time, "end": time + .05, "score": .9})
        time += .05
    return {"char_segments": out}


def test_exact_source_char_mapping_japanese_and_repeated_order():
    lines = ["同じ歌", "同じ歌"]
    result = WhisperXExactAligner.result_from_aligned(chars("同じ歌\n同じ歌"), lines, "ja")
    assert [cue.source for cue in result.cues] == lines
    assert result.cues[0].start < result.cues[1].start
    assert result.evidence.direct_line_ratio == 1


def test_english_apostrophe_is_not_evidence_unit():
    lines = ["Don't stop"]
    result = WhisperXExactAligner.result_from_aligned(chars("Don't stop"), lines, "en")
    assert result.evidence.source_unit_count == len("Dontstop")
    assert result.evidence.unit_coverage == 1


def test_missing_line_fails_instead_of_interpolating():
    with pytest.raises(AlignmentError):
        WhisperXExactAligner.result_from_aligned(chars("first"), ["first", "missing"], "en")


def test_speaker_prefix_removed_and_metadata_retained(tmp_path):
    path = tmp_path / "lyrics.json"
    path.write_text(json.dumps({"songs": [{"lyrics": "Male: I don't run\nFemale Vocal: 私は歌う"}]}),
                    encoding="utf-8")
    song = parse_lyrics_file(path, "en")[0]
    assert song.lyrics == ["I don't run", "私は歌う"]
    assert song.speaker_roles == {0: "Male", 1: "Female Vocal"}
