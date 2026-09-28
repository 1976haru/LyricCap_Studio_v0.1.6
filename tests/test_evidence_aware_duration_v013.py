from lyriccap.aligners.base import AlignmentEvidence, LineAlignmentEvidence
from lyriccap.models import Cue
from lyriccap.sync_quality import evaluate_sync_quality


def evidence(cues):
    lines = [LineAlignmentEvidence(i, True, 5, 5, 1.0, .95,
                                   cue.start, cue.end, "fixture")
             for i, cue in enumerate(cues)]
    return AlignmentEvidence(5 * len(cues), 5 * len(cues), 1.0,
                             len(cues), 1.0, 0, 0.0, 0, True, True, lines)


def test_strong_evidence_sustained_eleven_second_cue_warns_only():
    cues = [Cue(0, 11, "sustained")]
    report = evaluate_sync_quality(cues, ["sustained"], 20, evidence=evidence(cues),
                                   strict=True, vocal_activity_ranges=[(0, 11)])
    assert report.status == "PASS_WITH_WARNINGS"


def test_strong_evidence_fifteen_second_instrumental_gap_warns_only():
    cues = [Cue(0, 2, "a"), Cue(15, 17, "b")]
    report = evaluate_sync_quality(cues, ["a", "b"], 20, evidence=evidence(cues),
                                   strict=True, vocal_activity_ranges=[])
    assert report.status == "PASS_WITH_WARNINGS"


def test_unexplained_vocals_in_fifteen_second_gap_are_hard():
    cues = [Cue(0, 2, "a"), Cue(15, 17, "b")]
    report = evaluate_sync_quality(cues, ["a", "b"], 20, evidence=evidence(cues),
                                   strict=True, vocal_activity_ranges=[(5, 10)])
    assert report.hard_failed
    assert any(issue.code == "START_GAP" and issue.severity == "HARD"
               for issue in report.issues)


def test_long_cue_with_large_internal_no_vocal_region_is_hard():
    cues = [Cue(0, 11, "not sustained")]
    report = evaluate_sync_quality(cues, ["not sustained"], 20, evidence=evidence(cues),
                                   strict=True, vocal_activity_ranges=[(0, 2), (9, 11)])
    assert report.hard_failed
