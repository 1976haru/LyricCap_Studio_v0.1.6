from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .base import AlignmentEvidence, AlignmentResult, LineAlignmentEvidence, TimeBase
from .whisperx_exact_aligner import WhisperXExactAligner
from ..models import LyricSection
from ..section_timebase import rebase_alignment_result, validate_section_stitch


PIPELINE_VERSION = "acoustic_truth_v3_timebase_safe"


def _default_sections(lines: list[str]) -> list[LyricSection]:
    return [LyricSection("Lyrics", "lyrics", 0, len(lines) - 1)] if lines else []


class WhisperXExactSectionAligner:
    """Align structural sections locally, rebase once, then merge by source line."""

    engine = "whisperx-exact-section"
    overlap_tolerance = 0.30

    def __init__(self, exact_aligner: WhisperXExactAligner, margin: float = 6.0):
        self.exact_aligner = exact_aligner
        self.margin = margin

    @staticmethod
    def _accepted(previous_result):
        if not previous_result:
            return {}
        indices = previous_result.metadata.get("source_line_indices", [])
        return {indices[i]: (previous_result.cues[i], previous_result.evidence.line_evidence[indices[i]])
                for i in range(min(len(indices), len(previous_result.cues)))
                if previous_result.evidence and indices[i] < len(previous_result.evidence.line_evidence)
                and previous_result.evidence.line_evidence[indices[i]].direct}

    def _window(self, section, start, end, total_lines, accepted):
        proportional_start = start + (end - start) * section.first_line_index / total_lines
        proportional_end = start + (end - start) * (section.last_line_index + 1) / total_lines
        low = max(start, proportional_start - self.margin)
        high = min(end, proportional_end + self.margin)
        previous = [cue.end for index, (cue, _ev) in accepted.items()
                    if index < section.first_line_index]
        following = [cue.start for index, (cue, _ev) in accepted.items()
                     if index > section.last_line_index]
        if previous:
            low = max(low, max(previous) - self.overlap_tolerance)
        if following:
            high = min(high, min(following) + self.overlap_tolerance)
        if high <= low:
            raise RuntimeError(f"SECTION_WINDOW_CONFLICT: {section.name} {low:.2f}..{high:.2f}")
        return low, high

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds=None, *, sections=None, previous_result=None) -> AlignmentResult:
        start, end = bounds or (0.0, 0.0)
        sections = [s for s in (sections or _default_sections(lyric_lines)) if s.is_sung]
        total_lines = max(1, len(lyric_lines))
        accepted = self._accepted(previous_result)
        failed, passed, section_records = [], [], []

        for section in sections:
            try:
                window = self._window(section, start, end, total_lines, accepted)
                subset = lyric_lines[section.first_line_index:section.last_line_index + 1]
                local = self.exact_aligner.align(
                    audio_path, subset, language, bounds=window, section_local=True)
                returned_base = local.time_base
                first_local = local.cues[0].start if local.cues else None
                last_local = local.cues[-1].end if local.cues else None
                if returned_base != TimeBase.SECTION_LOCAL:
                    raise AssertionError("SECTION_NOT_REBASED: cropped aligner did not return section-local time")
                result = rebase_alignment_result(local, window[0], end)
                if len(result.cues) != len(subset):
                    raise RuntimeError("section did not return exactly one cue per source line")
                for local_index, cue in enumerate(result.cues):
                    source_index = section.first_line_index + local_index
                    ev = result.evidence.line_evidence[local_index] if result.evidence else None
                    accepted[source_index] = (cue, replace(
                        ev, line_index=source_index, engine=self.engine) if ev else
                        LineAlignmentEvidence(source_index, False, 0, 0, 0.0, None,
                                              None, None, self.engine))
                section_records.append({
                    "section_name": section.name,
                    "source_start_line": section.first_line_index,
                    "source_end_line": section.last_line_index,
                    "window_start_abs": window[0], "window_end_abs": window[1],
                    "aligner_return_time_base": returned_base.value,
                    "rebase_offset": window[0],
                    "first_local_time": first_local, "last_local_time": last_local,
                    "first_absolute_time": result.cues[0].start if result.cues else None,
                    "last_absolute_time": result.cues[-1].end if result.cues else None,
                    "stitch_status": "PENDING",
                })
                passed.append(section.name)
            except Exception as exc:
                failed.append({"name": section.name, "first": section.first_line_index,
                               "last": section.last_line_index,
                               "error": f"{type(exc).__name__}: {exc}"})

        indices = sorted(accepted)
        cues = [accepted[i][0] for i in indices]
        evidence_items = [accepted.get(i, (None, LineAlignmentEvidence(
            i, False, 0, 0, 0.0, None, None, None, self.engine)))[1]
            for i in range(len(lyric_lines))]
        section_hash = hashlib.sha256(json.dumps([
            (s.name, s.section_type, s.first_line_index, s.last_line_index,
             s.preceding_break, s.following_break) for s in sections
        ], ensure_ascii=False).encode("utf-8")).hexdigest()
        result = AlignmentResult(cues, self.engine, metadata={
            "failed_sections": failed, "passed_sections": passed,
            "section_definition_hash": section_hash, "pipeline_version": PIPELINE_VERSION,
            "source_line_indices": indices, "section_records": section_records,
        }, evidence=self._evidence(evidence_items), time_base=TimeBase.TRACK_ABSOLUTE,
           window_start=start, window_end=end)
        stitch = validate_section_stitch(result, len(lyric_lines), end, self.overlap_tolerance)
        if stitch.hard_failed:
            for issue in stitch.issues:
                owner = next((s for s in sections if issue.cue_index is not None and
                              s.first_line_index <= issue.cue_index <= s.last_line_index), None)
                failed.append({"name": owner.name if owner else "stitch",
                               "first": owner.first_line_index if owner else issue.cue_index,
                               "last": owner.last_line_index if owner else issue.cue_index,
                               "error": f"{issue.code}: {issue.message}"})
        return result

    @staticmethod
    def _evidence(items):
        source = sum(x.total_units for x in items)
        aligned = sum(x.matched_units for x in items)
        direct = sum(x.direct for x in items)
        return AlignmentEvidence(source, aligned, aligned / max(1, source), direct,
                                 direct / max(1, len(items)), 0, 0.0, 0,
                                 bool(items and items[0].direct),
                                 bool(items and items[-1].direct), items)


class WhisperXExactRecoveryAligner(WhisperXExactSectionAligner):
    """Retry failed source ranges while retaining already accepted sections."""

    engine = "whisperx-exact-recovery"

    def __init__(self, exact_aligner: WhisperXExactAligner):
        super().__init__(exact_aligner, margin=12.0)

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds=None, *, sections=None, previous_result=None) -> AlignmentResult:
        if previous_result is None:
            return super().align(audio_path, lyric_lines, language, bounds, sections=sections)
        failed_ranges = {(x.get("first"), x.get("last")) for x in
                         previous_result.metadata.get("failed_sections", [])
                         if x.get("first") is not None}
        retry_sections = [s for s in (sections or _default_sections(lyric_lines))
                          if (s.first_line_index, s.last_line_index) in failed_ranges]
        recovered = super().align(audio_path, lyric_lines, language, bounds,
                                  sections=retry_sections, previous_result=previous_result)
        recovered.engine = self.engine
        recovered.metadata["retained_passed_sections"] = previous_result.metadata.get(
            "passed_sections", [])
        old_records = []
        for record in previous_result.metadata.get("section_records", []):
            copy = dict(record)
            if (copy.get("source_start_line"), copy.get("source_end_line")) in failed_ranges:
                copy["stitch_status"] = "RETRIED"
            old_records.append(copy)
        recovered.metadata["section_records"] = old_records + recovered.metadata.get(
            "section_records", [])
        return recovered
