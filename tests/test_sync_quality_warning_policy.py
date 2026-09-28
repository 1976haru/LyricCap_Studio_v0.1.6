from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyriccap.aligners.base import AlignmentEvidence, LineAlignmentEvidence
from lyriccap.models import Cue
from lyriccap.sync_quality import evaluate_sync_quality


def evidence(cues, coverage=1.0, direct=None, interpolated=0):
    direct = [True] * len(cues) if direct is None else direct
    lines = [LineAlignmentEvidence(i, value, 1 if value else 0, 1,
             coverage if value else 0.0, 0.95 if value else None,
             cue.start if value else None, cue.end if value else None,
             "fixture", not value and interpolated > 0)
             for i, (cue, value) in enumerate(zip(cues, direct))]
    count = sum(direct)
    return AlignmentEvidence(len(cues), count, coverage, count, count / len(cues),
                             interpolated, interpolated / len(cues), interpolated,
                             direct[0], direct[-1], lines)


def test_long_direct_cue_is_pass_with_warning():
    cues = [Cue(0, 8.5, "sustained")]
    report = evaluate_sync_quality(cues, ["sustained"], 10, evidence=evidence(cues), strict=True)
    assert report.status == "PASS_WITH_WARNINGS"
    assert report.passed


def test_two_short_direct_cues_are_pass_with_warning():
    cues = [Cue(0, .45, "a"), Cue(.6, 1.05, "b")]
    report = evaluate_sync_quality(cues, ["a", "b"], 2, evidence=evidence(cues), strict=True)
    assert report.status == "PASS_WITH_WARNINGS"


def test_three_short_low_evidence_fail():
    cues = [Cue(i * .5, i * .5 + .45, str(i)) for i in range(3)]
    report = evaluate_sync_quality(cues, [str(i) for i in range(3)], 2,
                                   evidence=evidence(cues, coverage=.4), strict=True)
    assert report.status == "FAIL"


def test_instrumental_gap_with_no_vocal_activity_passes():
    cues = [Cue(0, 2, "a"), Cue(17, 19, "b")]
    report = evaluate_sync_quality(cues, ["a", "b"], 20, instrumental_after={1},
                                   evidence=evidence(cues), strict=True,
                                   vocal_activity_ranges=[])
    assert report.status == "PASS"


def test_plausible_fake_sync_fails_acoustic_gate():
    cues = [Cue(i * 3.0, i * 3.0 + 2.5, f"line {i}") for i in range(40)]
    direct = [i % 4 < 2 for i in range(40)]
    report = evaluate_sync_quality(cues, [c.source for c in cues], 125,
                                   evidence=evidence(cues, coverage=.45, direct=direct,
                                                     interpolated=22), strict=True)
    assert report.status == "FAIL"
    assert "INTERPOLATED_FINAL_CUE" in {issue.code for issue in report.issues}
