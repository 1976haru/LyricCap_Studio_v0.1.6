from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .aligners.base import AlignmentResult
from .sync_quality import SyncIssue, SyncQualityReport, evaluate_sync_quality


class TrackAlignmentState(str, Enum):
    PENDING = "PENDING"
    ALIGNING_PRIMARY = "ALIGNING_PRIMARY"
    VALIDATING_PRIMARY = "VALIDATING_PRIMARY"
    RETRYING = "RETRYING"
    ALIGNING_FALLBACK = "ALIGNING_FALLBACK"
    VALIDATING_FALLBACK = "VALIDATING_FALLBACK"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class TrackAlignmentOutcome:
    state: TrackAlignmentState
    result: AlignmentResult | None
    report: SyncQualityReport
    retry_count: int
    history: list[str]


def _exception_report(exc: Exception) -> SyncQualityReport:
    issue = SyncIssue("ALIGNER_EXCEPTION", "HARD", None, None, None, None,
                      f"{type(exc).__name__}: {exc}")
    return SyncQualityReport(False, 0.0, [issue], {})


class TrackAlignmentTransaction:
    def __init__(self, primary, retry_aligner, fallback, progress=None):
        self.primary = primary
        self.retry_aligner = retry_aligner
        self.fallback = fallback
        self.progress = progress or (lambda _message: None)

    def run(self, *, audio_path: Path, lyric_lines: list[str], language: str,
            track_duration: float, instrumental_after: set[int] | None = None) -> TrackAlignmentOutcome:
        history = [TrackAlignmentState.PENDING.value]
        attempts = (
            (TrackAlignmentState.ALIGNING_PRIMARY, TrackAlignmentState.VALIDATING_PRIMARY, self.primary, 0),
            (TrackAlignmentState.RETRYING, TrackAlignmentState.VALIDATING_PRIMARY, self.retry_aligner, 1),
            (TrackAlignmentState.ALIGNING_FALLBACK, TrackAlignmentState.VALIDATING_FALLBACK, self.fallback, 2),
        )
        last_result = None
        last_report = SyncQualityReport(False, 0.0, [], {})
        for aligning, validating, aligner, retry_count in attempts:
            history.append(aligning.value)
            self.progress(aligning.value)
            try:
                # Rejected timestamps are intentionally not passed to the next attempt.
                last_result = aligner.align(audio_path, lyric_lines, language,
                                            bounds=(0.0, track_duration))
                history.append(validating.value)
                last_report = evaluate_sync_quality(
                    last_result.cues, lyric_lines, track_duration,
                    instrumental_after=instrumental_after,
                    confidences=last_result.confidences,
                )
            except Exception as exc:
                last_result = None
                last_report = _exception_report(exc)
            if last_report.passed:
                history.append(TrackAlignmentState.PASS.value)
                return TrackAlignmentOutcome(TrackAlignmentState.PASS, last_result, last_report,
                                             retry_count, history)
        history.append(TrackAlignmentState.FAIL.value)
        return TrackAlignmentOutcome(TrackAlignmentState.FAIL, None, last_report, 2, history)
