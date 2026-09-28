from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..models import Cue


@dataclass
class AlignmentResult:
    cues: list[Cue]
    engine: str
    confidences: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseAligner(Protocol):
    def align(
        self,
        audio_path: Path,
        lyric_lines: list[str],
        language: str,
        bounds: tuple[float, float] | None = None,
    ) -> AlignmentResult: ...
