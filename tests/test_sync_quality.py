from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyriccap.models import Cue
from lyriccap.sync_quality import evaluate_sync_quality


def _seconds(value: str) -> float:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _fixture_cues():
    text = (Path(__file__).parent / "fixtures" / "broken_mid_playlist.srt").read_text(encoding="utf-8")
    pattern = re.compile(r"\d+\n([^ ]+) --> ([^\n]+)\n([^\n]+)")
    return [Cue(_seconds(start), _seconds(end), lyric) for start, end, lyric in pattern.findall(text)]


def test_broken_mid_playlist_is_a_hard_failure():
    cues = _fixture_cues()
    report = evaluate_sync_quality(cues, [cue.source for cue in cues], 180.0)
    codes = {issue.code for issue in report.issues if issue.severity == "HARD"}
    assert not report.passed
    assert {"EXTREME_DURATION", "START_GAP", "INSTANT_CUE"} <= codes
    assert report.stats["count_over_10"] == 2
    assert report.stats["max_consecutive_start_gap"] == 43.12


def test_one_or_two_fast_rap_cues_are_not_hard_failures():
    cues = [Cue(0, 0.45, "a"), Cue(0.6, 1.05, "b"), Cue(1.2, 2.0, "c")]
    report = evaluate_sync_quality(cues, ["a", "b", "c"], 3.0)
    assert not report.hard_failed


def test_declared_instrumental_gap_is_allowed_but_not_a_stretched_lyric():
    gap = [Cue(0, 2, "a"), Cue(22, 24, "b")]
    allowed = evaluate_sync_quality(gap, ["a", "b"], 30, instrumental_after={1})
    assert not allowed.hard_failed
    stretched = [Cue(0, 22, "a"), Cue(22, 24, "b")]
    rejected = evaluate_sync_quality(stretched, ["a", "b"], 30, instrumental_after={1})
    assert any(issue.code == "EXTREME_DURATION" for issue in rejected.issues)


def test_lyric_order_and_count_are_invariants():
    report = evaluate_sync_quality([Cue(0, 1, "b")], ["a", "b"], 3)
    codes = {issue.code for issue in report.issues}
    assert "LYRIC_COUNT_MISMATCH" in codes
    assert "LYRIC_ORDER_CHANGED" in codes
