from __future__ import annotations

import inspect
import hashlib
import math
import unicodedata
from pathlib import Path

from .base import AlignmentEvidence, AlignmentResult, LineAlignmentEvidence
from ..aligner import AlignmentError
from ..audio import audio_duration
from ..models import Cue


class WhisperXExactAligner:
    """CTC-align the supplied source lyrics, never an ASR transcript."""

    DEFAULT_JA_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"

    def __init__(self, model_name: str = "base", device: str = "auto", progress=None,
                 audio_resolver=None, cache_dir: Path | None = None, strict: bool = True):
        self.model_name = model_name
        self.device = device
        self.progress = progress or (lambda _message: None)
        self.audio_resolver = audio_resolver
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.strict = bool(strict)

    @staticmethod
    def _unit(char: str) -> str:
        value = unicodedata.normalize("NFKC", str(char)).casefold()
        return "".join(ch for ch in value if ch.isalnum())

    @classmethod
    def _source_units(cls, lines: list[str]):
        units = []
        counts = [0] * len(lines)
        for line_index, line in enumerate(lines):
            for char in unicodedata.normalize("NFKC", line):
                for normalized in cls._unit(char):
                    units.append((normalized, line_index))
                    counts[line_index] += 1
        return units, counts

    @staticmethod
    def alignment_cache_key(audio_path: Path, vocals_path: Path, lyric_lines: list[str],
                            language: str, align_model: str, strict: bool) -> str:
        """Content-addressed key; a one-character source edit is a cache miss."""
        def fingerprint(path: Path) -> str:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        payload = "\0".join((
            "acoustic_truth_v1", fingerprint(audio_path), fingerprint(vocals_path),
            hashlib.sha256("\n".join(lyric_lines).encode("utf-8")).hexdigest(),
            language, align_model, str(bool(strict)),
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _field(item, name, default=None):
        return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)

    @classmethod
    def _char_items(cls, aligned) -> list:
        if isinstance(aligned, dict) and aligned.get("char_segments"):
            return list(aligned["char_segments"])
        segments = aligned.get("segments", []) if isinstance(aligned, dict) else getattr(aligned, "segments", [])
        return [char for segment in segments or []
                for char in (cls._field(segment, "chars", []) or cls._field(segment, "char_segments", []) or [])]

    @classmethod
    def result_from_aligned(cls, aligned, lyric_lines: list[str], language: str,
                            engine: str = "whisperx-exact-source") -> AlignmentResult:
        source_units, counts = cls._source_units(lyric_lines)
        timed = []
        for item in cls._char_items(aligned):
            char = cls._field(item, "char", cls._field(item, "text", ""))
            start, end = cls._field(item, "start"), cls._field(item, "end")
            score = cls._field(item, "score", cls._field(item, "probability"))
            normalized = cls._unit(char)
            if not normalized:
                continue
            for unit in normalized:
                timed.append((unit, start, end, score))

        # WhisperX receives the exact same text. A monotonic cursor tolerates
        # align-model omissions while never borrowing a timestamp from ASR.
        mapping: dict[int, tuple[float, float, float | None]] = {}
        cursor = 0
        for unit, start, end, score in timed:
            if start is None or end is None:
                continue
            while cursor < len(source_units) and source_units[cursor][0] != unit:
                cursor += 1
            if cursor >= len(source_units):
                break
            try:
                a, b = float(start), float(end)
                confidence = float(score) if score is not None else None
            except (TypeError, ValueError):
                cursor += 1
                continue
            if math.isfinite(a) and math.isfinite(b) and b > a:
                mapping[cursor] = (a, b, confidence)
            cursor += 1

        per_line = [[] for _ in lyric_lines]
        for source_index, stamp in mapping.items():
            per_line[source_units[source_index][1]].append(stamp)

        cues: list[Cue] = []
        line_evidence: list[LineAlignmentEvidence] = []
        for index, (line, stamps) in enumerate(zip(lyric_lines, per_line)):
            total = counts[index]
            matched = len(stamps)
            coverage = matched / total if total else 0.0
            start = min((stamp[0] for stamp in stamps), default=None)
            end = max((stamp[1] for stamp in stamps), default=None)
            scores = [stamp[2] for stamp in stamps if stamp[2] is not None]
            confidence = sum(scores) / len(scores) if scores else None
            direct = bool(stamps and start is not None and end is not None and end > start)
            line_evidence.append(LineAlignmentEvidence(
                index, direct, matched, total, coverage, confidence, start, end, engine, False
            ))
            if direct:
                cues.append(Cue(start, end, line, language))

        direct_count = sum(item.direct for item in line_evidence)
        aligned_count = len(mapping)
        evidence = AlignmentEvidence(
            source_unit_count=len(source_units), aligned_unit_count=aligned_count,
            unit_coverage=aligned_count / max(1, len(source_units)),
            direct_line_count=direct_count,
            direct_line_ratio=direct_count / max(1, len(lyric_lines)),
            interpolated_line_count=0, interpolated_line_ratio=0.0,
            max_interpolated_run=0,
            first_line_direct=bool(line_evidence and line_evidence[0].direct),
            last_line_direct=bool(line_evidence and line_evidence[-1].direct),
            line_evidence=line_evidence,
        )
        if direct_count != len(lyric_lines):
            missing = [str(item.line_index + 1) for item in line_evidence if not item.direct]
            raise AlignmentError("Exact acoustic evidence missing for source lyric line(s): " + ", ".join(missing))
        confidences = [item.confidence for item in line_evidence if item.confidence is not None]
        return AlignmentResult(cues, engine, confidences, {"source_text_used": True}, evidence)

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds: tuple[float, float] | None = None) -> AlignmentResult:
        if not lyric_lines:
            return AlignmentResult([], "whisperx-exact-source")
        try:
            import torch
            import whisperx
        except ImportError as exc:
            raise AlignmentError("WhisperX exact-source alignment is unavailable") from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target = Path(self.audio_resolver(Path(audio_path))) if self.audio_resolver else Path(audio_path)
        duration = bounds[1] if bounds and bounds[1] > 0 else audio_duration(target)
        if duration <= 0:
            raise AlignmentError("Track duration could not be measured for exact alignment")
        self.progress(f"WhisperX exact-source CTC: {target.name}")
        try:
            audio = whisperx.load_audio(str(target))
            kwargs = {"language_code": language, "device": device}
            if language == "ja":
                kwargs["model_name"] = self.DEFAULT_JA_MODEL
            align_model, metadata = whisperx.load_align_model(**kwargs)
            segments = [{"start": 0.0, "end": float(duration), "text": "\n".join(lyric_lines)}]
            align_kwargs = {"return_char_alignments": True}
            if "print_progress" in inspect.signature(whisperx.align).parameters:
                align_kwargs["print_progress"] = False
            aligned = whisperx.align(segments, align_model, metadata, audio, device, **align_kwargs)
        except Exception as exc:
            raise AlignmentError(f"WhisperX exact-source alignment failed: {type(exc).__name__}: {exc}") from exc
        result = self.result_from_aligned(aligned, lyric_lines, language)
        result.metadata.update({
            "alignment_model": self.DEFAULT_JA_MODEL if language == "ja" else "whisperx-default",
            "aligned_audio": str(target), "pipeline_version": "acoustic_truth_v1",
            "alignment_cache_key": self.alignment_cache_key(
                Path(audio_path), target, lyric_lines, language,
                self.DEFAULT_JA_MODEL if language == "ja" else "whisperx-default", self.strict,
            ),
        })
        return result
