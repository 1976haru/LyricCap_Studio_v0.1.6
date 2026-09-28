from pathlib import Path

import pytest

from lyriccap.alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from lyriccap.aligners.base import (AlignmentEvidence, AlignmentResult,
                                    LineAlignmentEvidence, TimeBase)
from lyriccap.models import Cue, LyricSection
from lyriccap.section_timebase import rebase_alignment_result, validate_section_stitch


def local_result(start=5.0, end=8.0, text="line 11"):
    cue = Cue(start, end, text)
    line = LineAlignmentEvidence(0, True, 4, 4, 1.0, .95, start, end, "fixture")
    evidence = AlignmentEvidence(4, 4, 1.0, 1, 1.0, 0, 0.0, 0, True, True, [line])
    return AlignmentResult([cue], "fixture", evidence=evidence,
                           time_base=TimeBase.SECTION_LOCAL,
                           window_start=80.0, window_end=120.0)


def test_single_rebase_updates_cue_and_evidence_exactly():
    result = rebase_alignment_result(local_result(), 80.0, 180.0)
    assert (result.cues[0].start, result.cues[0].end) == (85.0, 88.0)
    assert (result.evidence.line_evidence[0].start,
            result.evidence.line_evidence[0].end) == (85.0, 88.0)
    assert result.time_base == TimeBase.TRACK_ABSOLUTE
    assert result.metadata["rebase_offset"] == 80.0


def test_double_rebase_is_blocked():
    result = rebase_alignment_result(local_result(), 80.0, 180.0)
    with pytest.raises(AssertionError):
        rebase_alignment_result(result, 80.0, 180.0)


def test_missing_rebase_is_detected():
    result = local_result()
    result.metadata["source_line_indices"] = [0]
    report = validate_section_stitch(result, 1, 180.0)
    assert "SECTION_NOT_REBASED" in {issue.code for issue in report.issues}


def test_double_offset_outside_window_is_detected():
    result = local_result(165.0, 168.0)
    result.time_base = TimeBase.TRACK_ABSOLUTE
    result.metadata.update({"source_line_indices": [0], "section_records": [{
        "section_name": "B", "source_start_line": 0, "source_end_line": 0,
        "window_start_abs": 80.0, "window_end_abs": 120.0, "rebase_offset": 80.0,
        "first_absolute_time": 165.0, "last_absolute_time": 168.0,
    }]})
    report = validate_section_stitch(result, 1, 180.0)
    assert "SECTION_DOUBLE_REBASE" in {issue.code for issue in report.issues}


def test_source_order_is_authoritative_not_timestamp_sort():
    result = AlignmentResult([Cue(71, 75, "line 10"), Cue(5, 8, "line 11")], "section",
                             metadata={"source_line_indices": [10, 11]})
    report = validate_section_stitch(result, 12, 180.0)
    assert result.cues[1].source == "line 11"
    assert "SECTION_TIMEBASE_CONFLICT" in {issue.code for issue in report.issues}


def test_neighboring_accepted_sections_bound_recovery_window():
    from lyriccap.aligners.whisperx_exact_section_aligner import WhisperXExactSectionAligner
    aligner = WhisperXExactSectionAligner(None, margin=12.0)
    section = LyricSection("Bridge", "bridge", 10, 11)
    accepted = {9: (Cue(70, 80, "previous"), None),
                12: (Cue(105, 108, "next"), None)}
    low, high = aligner._window(section, 0, 140, 20, accepted)
    assert low >= 79.7
    assert high <= 105.3


@pytest.mark.parametrize("overlap", [14.06, 10.34])
def test_real_overlap_signature_fails_stitch_and_triggers_recovery(overlap):
    previous = Cue(60, 75, "a")
    broken = Cue(75 - overlap, 78 - overlap, "b")
    bad = AlignmentResult([previous, broken], "section", metadata={
        "source_line_indices": [0, 1],
        "section_records": [{"section_name": "B", "source_start_line": 1,
                             "source_end_line": 1, "window_start_abs": 80,
                             "window_end_abs": 120, "rebase_offset": 80,
                             "first_absolute_time": broken.start,
                             "last_absolute_time": broken.end}],
    })
    good_cues = [Cue(71, 75, "a"), Cue(85, 88, "b")]
    good_lines = [LineAlignmentEvidence(i, True, 1, 1, 1.0, .95,
                                        cue.start, cue.end, "recovery")
                  for i, cue in enumerate(good_cues)]
    good_evidence = AlignmentEvidence(2, 2, 1.0, 2, 1.0, 0, 0.0, 0,
                                      True, True, good_lines)
    good = AlignmentResult(good_cues, "recovery", evidence=good_evidence)

    class Scripted:
        def __init__(self, value):
            self.value, self.calls = value, 0
        def align(self, *args, **kwargs):
            self.calls += 1
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    whole, section, recovery = Scripted(RuntimeError("whole failed")), Scripted(bad), Scripted(good)
    outcome = TrackAlignmentTransaction(whole, section, recovery).run(
        audio_path=Path("fixture.wav"), lyric_lines=["a", "b"], language="ja",
        track_duration=140.0)
    assert section.calls == recovery.calls == 1
    assert outcome.state in {TrackAlignmentState.PASS, TrackAlignmentState.PASS_WITH_WARNINGS}
    assert any(issue.code == "SECTION_TIMEBASE_CONFLICT"
               for issue in outcome.attempts[1].report.issues)


def test_section_a_and_b_correct_rebase_has_no_overlap():
    a = rebase_alignment_result(local_result(31, 35, "line 10"), 40, 180)
    b = rebase_alignment_result(local_result(5, 8, "line 11"), 80, 180)
    assert (a.cues[0].start, a.cues[0].end) == (71, 75)
    assert (b.cues[0].start, b.cues[0].end) == (85, 88)
