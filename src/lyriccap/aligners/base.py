from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..models import Cue


@dataclass
class LineAlignmentEvidence:
    line_index: int
    direct: bool
    matched_units: int
    total_units: int
    coverage: float
    confidence: float | None
    start: float | None
    end: float | None
    engine: str
    interpolated: bool = False


@dataclass
class AlignmentEvidence:
    source_unit_count: int
    aligned_unit_count: int
    unit_coverage: float
    direct_line_count: int
    direct_line_ratio: float
    interpolated_line_count: int
    interpolated_line_ratio: float
    max_interpolated_run: int
    first_line_direct: bool
    last_line_direct: bool
    line_evidence: list[LineAlignmentEvidence] = field(default_factory=list)


@dataclass
class AlignmentResult:
    cues: list[Cue]
    engine: str
    confidences: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    evidence: AlignmentEvidence | None = None


class BaseAligner(Protocol):
    def align(
        self,
        audio_path: Path,
        lyric_lines: list[str],
        language: str,
        bounds: tuple[float, float] | None = None,
    ) -> AlignmentResult: ...
