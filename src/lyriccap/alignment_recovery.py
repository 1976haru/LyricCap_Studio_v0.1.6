from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .aligners.base import AlignmentResult
from .section_timebase import validate_section_stitch
from .sync_quality import SyncIssue, SyncQualityReport, evaluate_sync_quality


class TrackAlignmentState(str, Enum):
    PENDING = "PENDING"
    ALIGNING_PRIMARY = "ALIGNING_PRIMARY"
    VALIDATING_PRIMARY = "VALIDATING_PRIMARY"
    RETRYING = "RETRYING"
    ALIGNING_FALLBACK = "ALIGNING_FALLBACK"
    VALIDATING_FALLBACK = "VALIDATING_FALLBACK"
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


@dataclass
class AlignmentAttemptRecord:
    attempt_no: int
    engine: str
    mode: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    report: SyncQualityReport | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "attempt_no": self.attempt_no, "engine": self.engine, "mode": self.mode,
            "status": self.status, "error_type": self.error_type,
            "error_message": self.error_message,
            "report": None if self.report is None else self.report.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class TrackAlignmentOutcome:
    state: TrackAlignmentState
    final_result: AlignmentResult | None
    final_report: SyncQualityReport
    attempts: list[AlignmentAttemptRecord]
    history: list[str]

    @property
    def result(self): return self.final_result

    @property
    def report(self): return self.final_report

    @property
    def retry_count(self): return max(0, len(self.attempts) - 1)


def _exception_report(exc: Exception) -> SyncQualityReport:
    issue = SyncIssue("ALIGNER_EXCEPTION", "HARD", None, None, None, None,
                      f"{type(exc).__name__}: {exc}")
    return SyncQualityReport(False, 0.0, [issue], {})


class TrackAlignmentTransaction:
    """Exact-only authority chain. Every failed attempt remains reportable."""

    def __init__(self, primary, section_aligner, recovery_aligner, progress=None,
                 strict: bool = False):
        self.aligners = ((primary, "WHOLE_TRACK"),
                         (section_aligner, "SECTION_WINDOWS"),
                         (recovery_aligner, "FAILED_SECTION_ONLY"))
        self.progress = progress or (lambda _message: None)
        self.strict = bool(strict)

    @staticmethod
    def _call(aligner, audio_path, lyric_lines, language, bounds, sections, previous_result):
        params = inspect.signature(aligner.align).parameters
        kwargs = {"bounds": bounds}
        if "sections" in params:
            kwargs["sections"] = sections
        if "previous_result" in params:
            kwargs["previous_result"] = previous_result
        return aligner.align(audio_path, lyric_lines, language, **kwargs)

    def run(self, *, audio_path: Path, lyric_lines: list[str], language: str,
            track_duration: float, instrumental_after: set[int] | None = None,
            sections=None, start_stage: int = 1) -> TrackAlignmentOutcome:
        history = [TrackAlignmentState.PENDING.value]
        records = []
        previous_result = None
        last_report = SyncQualityReport(False, 0.0, [], {})
        state_names = (TrackAlignmentState.ALIGNING_PRIMARY,
                       TrackAlignmentState.RETRYING,
                       TrackAlignmentState.ALIGNING_FALLBACK)
        for index, ((aligner, mode), aligning) in enumerate(zip(self.aligners, state_names), 1):
            if index < start_stage:
                continue
            history.append(aligning.value)
            self.progress(aligning.value)
            result = None
            error = None
            try:
                result = self._call(aligner, audio_path, lyric_lines, language,
                                    (0.0, track_duration), sections, previous_result)
                is_section_result = bool(result.metadata.get("section_records") or
                                         result.metadata.get("source_line_indices") or
                                         result.time_base.value == "section_local")
                stitch = (validate_section_stitch(result, len(lyric_lines), track_duration)
                          if is_section_result else None)
                if stitch and stitch.hard_failed:
                    last_report = stitch
                else:
                    last_report = evaluate_sync_quality(
                        result.cues, lyric_lines, track_duration,
                        instrumental_after=instrumental_after, confidences=result.confidences,
                        evidence=result.evidence, language=language, strict=self.strict,
                        vocal_activity_ranges=result.metadata.get("vocal_activity_ranges"))
            except Exception as exc:
                error = exc
                last_report = _exception_report(exc)
            engine = getattr(result, "engine", None) or getattr(aligner, "engine", None)
            engine = engine or type(aligner).__name__
            records.append(AlignmentAttemptRecord(
                index, engine, mode, last_report.status,
                None if error is None else type(error).__name__,
                None if error is None else str(error), last_report,
                dict(result.metadata) if result else {}))
            if last_report.passed and result is not None:
                state = (TrackAlignmentState.PASS_WITH_WARNINGS if last_report.has_warnings
                         else TrackAlignmentState.PASS)
                history.append(state.value)
                return TrackAlignmentOutcome(state, result, last_report, records, history)
            if result is not None:
                previous_result = result
        history.append(TrackAlignmentState.FAIL.value)
        return TrackAlignmentOutcome(TrackAlignmentState.FAIL, None, last_report, records, history)
