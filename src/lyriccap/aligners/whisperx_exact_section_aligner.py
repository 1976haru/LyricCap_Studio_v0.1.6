from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .base import AlignmentEvidence, AlignmentResult, LineAlignmentEvidence
from .whisperx_exact_aligner import WhisperXExactAligner
from ..models import LyricSection


def _default_sections(lines: list[str]) -> list[LyricSection]:
    return [LyricSection("Lyrics", "lyrics", 0, len(lines) - 1)] if lines else []


class WhisperXExactSectionAligner:
    """Align each structural lyric section in a bounded, monotonic window."""

    engine = "whisperx-exact-section"

    def __init__(self, exact_aligner: WhisperXExactAligner, margin: float = 6.0):
        self.exact_aligner = exact_aligner
        self.margin = margin

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds=None, *, sections=None, previous_result=None) -> AlignmentResult:
        start, end = bounds or (0.0, 0.0)
        duration = max(0.0, end - start)
        sections = list(sections or _default_sections(lyric_lines))
        total_lines = max(1, len(lyric_lines))
        cues, failed, passed = [], [], []
        line_evidence = [LineAlignmentEvidence(i, False, 0, 0, 0.0, None, None, None,
                                                self.engine) for i in range(len(lyric_lines))]
        for section in sections:
            if not section.is_sung:
                continue
            proportional_start = start + duration * section.first_line_index / total_lines
            proportional_end = start + duration * (section.last_line_index + 1) / total_lines
            window = (max(start, proportional_start - self.margin),
                      min(end, proportional_end + self.margin))
            subset = lyric_lines[section.first_line_index:section.last_line_index + 1]
            try:
                result = self.exact_aligner.align(audio_path, subset, language, bounds=window)
                if cues and result.cues and result.cues[0].start < cues[-1].start:
                    raise RuntimeError("section result violates monotonic order")
                cues.extend(result.cues)
                if result.evidence:
                    for local in result.evidence.line_evidence:
                        line_evidence[section.first_line_index + local.line_index] = replace(
                            local, line_index=section.first_line_index + local.line_index,
                            engine=self.engine)
                passed.append(section.name)
            except Exception as exc:
                failed.append({"name": section.name, "first": section.first_line_index,
                               "last": section.last_line_index,
                               "error": f"{type(exc).__name__}: {exc}"})
        evidence = self._evidence(line_evidence)
        section_hash = hashlib.sha256(json.dumps([
            (s.name, s.section_type, s.first_line_index, s.last_line_index,
             s.preceding_break, s.following_break) for s in sections
        ], ensure_ascii=False).encode("utf-8")).hexdigest()
        return AlignmentResult(cues, self.engine, metadata={
            "failed_sections": failed, "passed_sections": passed,
            "section_definition_hash": section_hash,
            "pipeline_version": "acoustic_truth_v2_exact_recovery",
        }, evidence=evidence)

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
    """Retry only failed sections, retaining exact evidence already accepted."""

    engine = "whisperx-exact-recovery"

    def __init__(self, exact_aligner: WhisperXExactAligner):
        super().__init__(exact_aligner, margin=12.0)

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds=None, *, sections=None, previous_result=None) -> AlignmentResult:
        if previous_result is None:
            return super().align(audio_path, lyric_lines, language, bounds,
                                 sections=sections)
        failed_ranges = {(x["first"], x["last"]) for x in
                         previous_result.metadata.get("failed_sections", [])}
        retry_sections = [s for s in (sections or _default_sections(lyric_lines))
                          if (s.first_line_index, s.last_line_index) in failed_ranges]
        recovered = super().align(audio_path, lyric_lines, language, bounds,
                                  sections=retry_sections)
        by_index = {}
        for cue, ev in zip(previous_result.cues,
                           [x for x in previous_result.evidence.line_evidence if x.direct]):
            by_index[ev.line_index] = (cue, ev)
        direct_recovered = [x for x in recovered.evidence.line_evidence if x.direct]
        for cue, ev in zip(recovered.cues, direct_recovered):
            by_index[ev.line_index] = (cue, replace(ev, engine=self.engine))
        cues = [by_index[i][0] for i in sorted(by_index)]
        evidence_items = [by_index.get(i, (None, LineAlignmentEvidence(
            i, False, 0, 0, 0.0, None, None, None, self.engine)))[1]
            for i in range(len(lyric_lines))]
        recovered.cues = cues
        recovered.engine = self.engine
        recovered.evidence = self._evidence(evidence_items)
        recovered.metadata["retained_passed_sections"] = previous_result.metadata.get("passed_sections", [])
        return recovered
