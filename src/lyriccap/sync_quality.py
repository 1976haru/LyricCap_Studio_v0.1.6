from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Iterable

from .models import Cue
from .aligners.base import AlignmentEvidence


@dataclass(frozen=True)
class SyncIssue:
    code: str
    severity: str
    cue_index: int | None
    start: float | None
    end: float | None
    duration: float | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyncQualityReport:
    passed: bool
    score: float
    issues: list[SyncIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def hard_failed(self) -> bool:
        return any(issue.severity == "HARD" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity != "HARD" for issue in self.issues)

    @property
    def status(self) -> str:
        if self.hard_failed:
            return "FAIL"
        return "PASS_WITH_WARNINGS" if self.has_warnings else "PASS"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "hard_failed": self.hard_failed,
            "has_warnings": self.has_warnings,
            "status": self.status,
            "score": self.score,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
        }


def _issue(code: str, severity: str, index: int | None, cue: Cue | None, message: str) -> SyncIssue:
    return SyncIssue(
        code=code,
        severity=severity,
        cue_index=index,
        start=None if cue is None else cue.start,
        end=None if cue is None else cue.end,
        duration=None if cue is None else cue.end - cue.start,
        message=message,
    )


def evaluate_sync_quality(
    cues: Iterable[Cue],
    expected_lyrics: list[str],
    track_duration: float,
    *,
    instrumental_after: set[int] | None = None,
    confidences: Iterable[float] | None = None,
    evidence: AlignmentEvidence | None = None,
    language: str = "en",
    strict: bool = False,
    vocal_activity_ranges: Iterable[tuple[float, float]] | None = None,
) -> SyncQualityReport:
    """Validate local (track-relative) timestamps before any SRT is written.

    Timing shape is advisory. In strict mode, direct acoustic evidence is the
    authority; warnings do not trigger retries or block playlist output.
    """
    cues = list(cues)
    instrumental_after = set(instrumental_after or ())
    issues: list[SyncIssue] = []
    durations = [cue.end - cue.start for cue in cues]
    gaps = [cues[i].start - cues[i - 1].start for i in range(1, len(cues))]

    if track_duration <= 0:
        issues.append(_issue("TRACK_DURATION_UNKNOWN", "HARD", None, None,
                             "track duration could not be measured"))

    if len(cues) != len(expected_lyrics):
        issues.append(_issue("LYRIC_COUNT_MISMATCH", "HARD", None, None,
                             f"aligned {len(cues)} lyrics; expected {len(expected_lyrics)}"))
    for i, (cue, expected) in enumerate(zip(cues, expected_lyrics)):
        if cue.source != expected:
            issues.append(_issue("LYRIC_ORDER_CHANGED", "HARD", i, cue,
                                 "aligned lyric text/order differs from the source"))
        duration = cue.end - cue.start
        if cue.end <= cue.start:
            issues.append(_issue("NON_POSITIVE_DURATION", "HARD", i, cue, "cue end is not after start"))
        line_ev = evidence.line_evidence[i] if evidence and i < len(evidence.line_evidence) else None
        weak = line_ev is None or not line_ev.direct or line_ev.coverage < 0.5
        if duration < 0.18:
            issues.append(_issue("INSTANT_CUE", "HARD" if weak else "SOFT", i, cue,
                                 "cue is shorter than 0.18 seconds"))
        elif duration < 0.55:
            issues.append(_issue("SHORT_CUE", "SOFT", i, cue, "cue is shorter than 0.55 seconds"))
        if duration > 10.0:
            issues.append(_issue("EXTREME_DURATION", "HARD" if weak else "SOFT", i, cue,
                                 "lyric cue exceeds 10 seconds"))
        elif duration > 7.0:
            issues.append(_issue("LONG_CUE", "SOFT", i, cue, "lyric cue exceeds 7 seconds"))
        if cue.start < 0 or (track_duration > 0 and cue.end > track_duration + 0.01):
            issues.append(_issue("OUTSIDE_TRACK", "HARD", i, cue, "timestamp is outside track duration"))
        if i:
            previous = cues[i - 1]
            if cue.start < previous.start:
                issues.append(_issue("NON_MONOTONIC", "HARD", i, cue, "cue starts before previous cue"))
            overlap = previous.end - cue.start
            if overlap > 0.30:
                issues.append(_issue("OVERLAP", "HARD", i, cue, f"overlap is {overlap:.2f} seconds"))
            start_gap = cue.start - previous.start
            if start_gap > 12.0:
                activity_known = vocal_activity_ranges is not None
                activity = list(vocal_activity_ranges or ())
                gap_has_vocal = any(a < cue.start and b > previous.end for a, b in activity)
                explained = (i in instrumental_after and activity_known and not gap_has_vocal
                             and (evidence is None or not weak))
                if not explained:
                    pathological_neighbor = (previous.end - previous.start > 10.0
                                             or cue.end - cue.start < 0.18)
                    issues.append(_issue("START_GAP", "HARD" if gap_has_vocal or (strict and weak)
                                         or pathological_neighbor else "SOFT", i, cue,
                                         f"consecutive start gap is {start_gap:.2f} seconds"))

    run = 0
    for i, duration in enumerate(durations):
        run = run + 1 if duration < 0.55 else 0
        if run == 3:
            window = evidence.line_evidence[i - 2:i + 1] if evidence else []
            low = not window or any(not item.direct or item.coverage < 0.5 for item in window)
            issues.append(_issue("SHORT_CUE_RUN", "HARD" if low else "SOFT", i, cues[i],
                                 "three or more consecutive cues are shorter than 0.55 seconds"))

    if strict:
        if evidence is None:
            issues.append(_issue("ACOUSTIC_EVIDENCE_MISSING", "HARD", None, None,
                                 "exact acoustic alignment evidence is missing"))
        else:
            threshold = 0.65 if language in {"ja", "ko"} else 0.75
            checks = (
                (evidence.interpolated_line_count > 0, "INTERPOLATED_FINAL_CUE", "FINAL cues contain interpolation"),
                (evidence.direct_line_ratio < 0.95, "LOW_DIRECT_LINE_RATIO",
                 f"direct line ratio is {evidence.direct_line_ratio:.1%}"),
                (not evidence.first_line_direct, "FIRST_LINE_NOT_DIRECT", "first sung line has no direct evidence"),
                (not evidence.last_line_direct, "LAST_LINE_NOT_DIRECT", "last sung line has no direct evidence"),
                (evidence.unit_coverage < threshold, "LOW_SOURCE_UNIT_COVERAGE",
                 f"source unit coverage is {evidence.unit_coverage:.1%}; required {threshold:.0%}"),
            )
            for failed, code, message in checks:
                if failed:
                    issues.append(_issue(code, "HARD", None, None, message))
            missing_run = 0
            for item in evidence.line_evidence:
                missing_run = 0 if item.direct else missing_run + 1
                if missing_run >= 2:
                    issues.append(_issue("MISSING_DIRECT_LINE_RUN", "HARD", item.line_index, None,
                                         "two or more consecutive lines lack direct evidence"))
                    break
            # A strict FINAL cannot serialize a source line without a direct
            # timestamp even when the aggregate ratio happens to be 95%.
            if any(not item.direct for item in evidence.line_evidence):
                issues.append(_issue("SOURCE_LINE_NOT_DIRECT", "HARD", None, None,
                                     "one or more source lines lack direct acoustic evidence"))

    short_count = sum(duration < 0.55 for duration in durations)
    over_7 = sum(duration > 7.0 for duration in durations)
    over_10 = sum(duration > 10.0 for duration in durations)
    if durations and short_count / len(durations) > 0.08:
        issues.append(_issue("SHORT_CUE_RATIO", "SOFT", None, None,
                             f"{short_count}/{len(durations)} cues are shorter than 0.55 seconds"))

    split = max(1, len(durations) // 2)
    first = durations[:split]
    second = durations[split:]
    if first and second and min(median(first), median(second)) > 0:
        ratio = max(median(first), median(second)) / min(median(first), median(second))
        if ratio >= 2.5:
            issues.append(_issue("DURATION_DISTRIBUTION_SHIFT", "SOFT", None, None,
                                 f"first/second-half median duration ratio is {ratio:.2f}"))

    if len(durations) >= 8:
        boundary = max(1, int(len(durations) * 0.75))
        abnormal = lambda value: value < 0.55 or value > 7.0
        early_rate = sum(map(abnormal, durations[:boundary])) / boundary
        late_count = len(durations) - boundary
        late_rate = sum(map(abnormal, durations[boundary:])) / max(1, late_count)
        if late_rate > 0 and late_rate >= max(early_rate * 3.0, 0.12):
            issues.append(_issue("LATE_OUTLIER_SPIKE", "SOFT", None, None,
                                 "last-quarter outlier rate is at least three times the earlier rate"))

    confidence_values = list(confidences or ())
    if confidence_values:
        low_rate = sum(value < 0.35 for value in confidence_values) / len(confidence_values)
        if low_rate > 0.20:
            issues.append(_issue("LOW_CONFIDENCE_RATIO", "SOFT", None, None,
                                 f"low-confidence ratio is {low_rate:.1%}"))

    hard_count = sum(issue.severity == "HARD" for issue in issues)
    soft_count = len(issues) - hard_count
    score = max(0.0, min(100.0, 100.0 - hard_count * 25.0 - soft_count * 4.0))
    stats = {
        "lyric_count_expected": len(expected_lyrics),
        "lyric_count_aligned": len(cues),
        "min_cue_duration": min(durations, default=0.0),
        "median_cue_duration": median(durations) if durations else 0.0,
        "max_cue_duration": max(durations, default=0.0),
        "count_under_0_55": short_count,
        "count_over_7": over_7,
        "count_over_10": over_10,
        "max_consecutive_start_gap": max(gaps, default=0.0),
    }
    hard_failed = any(issue.severity == "HARD" for issue in issues)
    return SyncQualityReport(passed=not hard_failed, score=round(score, 1), issues=issues, stats=stats)
