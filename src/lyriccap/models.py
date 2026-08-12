from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Song:
    track_no: int
    title: str
    lyrics: list[str]
    source_language: str = "en"
    audio_path: Optional[Path] = None
    localized_title: str = ""


@dataclass
class Cue:
    start: float
    end: float
    source: str
    source_language: str = "en"
    en: str = ""
    ko: str = ""
    ja: str = ""

    def text(self, lang: str) -> str:
        if lang == self.source_language:
            return self.source
        return getattr(self, lang, "") or self.source

    def set_text(self, lang: str, value: str) -> None:
        if lang == self.source_language:
            self.source = value
        if lang in {"en", "ko", "ja"}:
            setattr(self, lang, value)


@dataclass
class TrackResult:
    song: Song
    cues: list[Cue] = field(default_factory=list)
    duration: float = 0.0
    warning: str = ""
