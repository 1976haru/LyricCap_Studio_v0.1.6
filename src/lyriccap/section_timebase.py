from __future__ import annotations

from .aligners.base import AlignmentResult, TimeBase
from .sync_quality import SyncIssue, SyncQualityReport


def _shift_persisted_timestamps(value, offset: float) -> None:
    """Shift timestamp evidence embedded in metadata, if an aligner persisted it."""
    if isinstance(value, list):
        for item in value:
            _shift_persisted_timestamps(item, offset)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"start", "end"} and isinstance(item, (int, float)):
                value[key] = item + offset
            elif key in {"char_timestamps", "word_timestamps", "segments"}:
                _shift_persisted_timestamps(item, offset)


def rebase_alignment_result(result: AlignmentResult, offset: float,
                            track_duration: float) -> AlignmentResult:
    """Convert one cropped result to track time exactly once."""
    assert result.time_base == TimeBase.SECTION_LOCAL, (
        f"cannot rebase {result.time_base.value}; expected section_local")
    offset = float(offset)
    for cue in result.cues:
        cue.start += offset
        cue.end += offset
    if result.evidence:
        for item in result.evidence.line_evidence:
            if item.start is not None:
                item.start += offset
            if item.end is not None:
                item.end += offset
    for key in ("char_timestamps", "word_timestamps", "segments"):
        if key in result.metadata:
            _shift_persisted_timestamps(result.metadata[key], offset)
    result.metadata["rebased_from"] = TimeBase.SECTION_LOCAL.value
    result.metadata["rebase_offset"] = offset
    result.time_base = TimeBase.TRACK_ABSOLUTE
    result.window_start = offset
    if not result.window_end:
        result.window_end = float(track_duration)
    return result


def _hard(code: str, index: int | None, start: float | None,
          end: float | None, message: str) -> SyncIssue:
    duration = None if start is None or end is None else end - start
    return SyncIssue(code, "HARD", index, start, end, duration, message)


def validate_section_stitch(result: AlignmentResult, expected_line_count: int,
                            track_duration: float, overlap_tolerance: float = 0.30
                            ) -> SyncQualityReport:
    """Validate source-order section output before the generic quality gate."""
    issues: list[SyncIssue] = []
    for failed in result.metadata.get("failed_sections", []):
        detail = str(failed.get("error", ""))
        for code in ("SECTION_WINDOW_CONFLICT", "SECTION_NOT_REBASED",
                     "SECTION_DOUBLE_REBASE", "SECTION_TIMEBASE_CONFLICT",
                     "SECTION_OUTSIDE_WINDOW"):
            if code in detail:
                issues.append(_hard(code, failed.get("first"), None, None, detail))
                break
    if result.time_base != TimeBase.TRACK_ABSOLUTE:
        issues.append(_hard("SECTION_NOT_REBASED", None, None, None,
                            "section result was not converted to track-absolute time"))

    indices = list(result.metadata.get("source_line_indices", []))
    if indices:
        duplicates = sorted({i for i in indices if indices.count(i) > 1})
        if duplicates:
            issues.append(_hard("SECTION_SOURCE_RANGE_DUPLICATE", duplicates[0], None, None,
                                f"source line indices occur more than once: {duplicates}"))
        missing = sorted(set(range(expected_line_count)) - set(indices))
        if missing:
            issues.append(_hard("SECTION_SOURCE_RANGE_MISSING", missing[0], None, None,
                                f"source line indices are missing: {missing}"))
        if indices != sorted(indices):
            issues.append(_hard("SECTION_TIMEBASE_CONFLICT", None, None, None,
                                "section cues are not merged in source line order"))

    for i, cue in enumerate(result.cues):
        line_index = indices[i] if i < len(indices) else i
        if cue.start < 0 or cue.end > track_duration + 0.01:
            issues.append(_hard("SECTION_OUTSIDE_WINDOW", line_index, cue.start, cue.end,
                                "section cue is outside track duration"))
        if i:
            previous = result.cues[i - 1]
            overlap = previous.end - cue.start
            if cue.start < previous.start or overlap > overlap_tolerance:
                issues.append(_hard("SECTION_TIMEBASE_CONFLICT", line_index, cue.start, cue.end,
                                    f"source-order section overlap is {max(0.0, overlap):.2f} seconds"))

    cue_by_index = {indices[i] if i < len(indices) else i: cue
                    for i, cue in enumerate(result.cues)}
    if result.evidence:
        for item in result.evidence.line_evidence:
            cue = cue_by_index.get(item.line_index)
            if cue and item.start is not None and item.end is not None:
                if abs(cue.start - item.start) > 0.02 or abs(cue.end - item.end) > 0.02:
                    issues.append(_hard("SECTION_TIMEBASE_CONFLICT", item.line_index,
                                        cue.start, cue.end,
                                        "cue and acoustic-evidence timestamps disagree"))

    for section in result.metadata.get("section_records", []):
        if section.get("stitch_status") == "RETRIED":
            continue
        first = section.get("first_absolute_time")
        last = section.get("last_absolute_time")
        low = section.get("window_start_abs")
        high = section.get("window_end_abs")
        if first is None or last is None or low is None or high is None:
            continue
        if first < low - 0.01 or last > high + 0.01:
            offset = float(section.get("rebase_offset") or 0.0)
            code = "SECTION_DOUBLE_REBASE" if offset and low <= first - offset <= high else "SECTION_OUTSIDE_WINDOW"
            issues.append(_hard(code, section.get("source_start_line"), first, last,
                                f"{section.get('section_name', 'section')} output {first:.2f}..{last:.2f} "
                                f"is outside {low:.2f}..{high:.2f}"))
            section["stitch_status"] = code
        else:
            section["stitch_status"] = "PASS"

    hard = bool(issues)
    return SyncQualityReport(not hard, 0.0 if hard else 100.0, issues, {
        "lyric_count_expected": expected_line_count,
        "lyric_count_aligned": len(result.cues),
    })
