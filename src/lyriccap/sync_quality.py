from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Iterable

from .models import Cue


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

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
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
) -> SyncQualityReport:
    """Validate local (track-relative) timestamps before any SRT is written.

    Soft findings deliberately make ``passed`` false so the transaction retries
    the track. A lone 0.4--0.5 second rap line is only a soft finding and does
    not become a hard failure unless it is part of a pathological run.
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
        if duration < 0.18:
            issues.append(_issue("INSTANT_CUE", "HARD", i, cue, "cue is shorter than 0.18 seconds"))
        elif duration < 0.55:
            issues.append(_issue("SHORT_CUE", "SOFT", i, cue, "cue is shorter than 0.55 seconds"))
        if duration > 10.0:
            issues.append(_issue("EXTREME_DURATION", "HARD", i, cue, "lyric cue exceeds 10 seconds"))
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
            if start_gap > 12.0 and i not in instrumental_after:
                issues.append(_issue("START_GAP", "HARD", i, cue, f"consecutive start gap is {start_gap:.2f} seconds"))

    run = 0
    for i, duration in enumerate(durations):
        run = run + 1 if duration < 0.55 else 0
        if run == 3:
            issues.append(_issue("SHORT_CUE_RUN", "HARD", i, cues[i],
                                 "three or more consecutive cues are shorter than 0.55 seconds"))

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
    return SyncQualityReport(passed=not issues, score=round(score, 1), issues=issues, stats=stats)
