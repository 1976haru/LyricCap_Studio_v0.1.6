from __future__ import annotations

import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import Cue
from .vocal_separator import DemucsSeparator, VocalSeparationError


class AlignmentError(RuntimeError):
    """Raised when a real audio/lyric alignment cannot be produced safely."""


@dataclass
class AlignmentStats:
    cue_count: int
    word_count: int
    first_start: float
    last_end: float
    mean_word_probability: float | None = None
    rebuilt_from_words: bool = False
    aligned_audio: str = "original"


class StableTSAligner:
    """Forced-align known lyrics to actual audio using stable-ts.

    v0.1.6 change:
    - music_precise no longer passes denoiser='demucs' into stable-ts.
    - Demucs is run as a separate subprocess to create vocals.wav.
    - stable-ts aligns the known lyric text against that vocals.wav + VAD.

    This avoids cross-library tuple/tensor compatibility errors while still
    using the recommended vocal-isolation + VAD workflow for music.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        sync_mode: str = "music_precise",
        refine: bool = False,
        cache_dir: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.sync_mode = sync_mode
        self.refine = bool(refine)
        self._model = None
        self.last_stats: AlignmentStats | None = None
        self.progress = progress or (lambda _msg: None)
        self.cache_dir = Path(cache_dir or Path.cwd() / ".lyriccap_cache")
        self.separator = DemucsSeparator(self.cache_dir / "vocals", progress=self.progress)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import stable_whisper
        except ImportError as e:
            raise RuntimeError(
                "stable-ts 또는 PyTorch가 설치되지 않았습니다. START_HERE.bat을 다시 실행하세요."
            ) from e
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dynamic quantization can improve CPU inference, but it is not used on
        # CUDA and not all stable-ts versions expose the kwarg consistently.
        try:
            self._model = stable_whisper.load_model(
                self.model_name,
                device=device,
                dq=(device == "cpu"),
            )
        except TypeError:
            self._model = stable_whisper.load_model(self.model_name, device=device)
        return self._model

    def _alignment_options(self) -> dict:
        # Keep this option set deliberately conservative and close to the
        # documented align() interface. In particular, do not pass a denoiser
        # here: precise mode already created a standalone vocal stem.
        opts = dict(
            original_split=True,
            token_step=100,
            nonspeech_skip=5.0,
            failure_threshold=0.45,
            suppress_silence=True,
            vad=True,
            vad_threshold=0.35,
            min_word_dur=0.06,
            min_silence_dur=0.12,
            nonspeech_error=0.10,
            verbose=None,
        )
        if self.sync_mode == "voice":
            opts["only_voice_freq"] = True
        elif self.sync_mode not in {"music_precise", "music_fast"}:
            raise ValueError(f"알 수 없는 싱크 모드: {self.sync_mode}")
        return opts

    def _prepare_audio(self, audio_path: Path) -> tuple[Path, str]:
        if self.sync_mode == "music_precise":
            try:
                return self.separator.separate(audio_path), "demucs-vocals"
            except VocalSeparationError as e:
                raise AlignmentError(str(e)) from e
        return audio_path, "original"

    def align(self, audio_path: Path, lines: list[str], language: str = "en") -> list[Cue]:
        if not lines:
            return []
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AlignmentError(f"음원 파일을 찾을 수 없습니다: {audio_path}")

        model = self._load()
        align_audio, aligned_audio_label = self._prepare_audio(audio_path)
        text = "\n".join(lines)
        opts = self._alignment_options()
        random.seed(0)

        try:
            # Use keyword form for language so argument ordering cannot be
            # misinterpreted across stable-ts minor versions.
            result = model.align(str(align_audio), text, language=language, **opts)
        except Exception as e:
            raise AlignmentError(
                "실제 음원 가사 정렬에 실패했습니다.\n"
                f"정렬 대상: {align_audio}\n"
                f"원본 오류: {type(e).__name__}: {e}"
            ) from e

        if result is None:
            raise AlignmentError("실제 음원에서 가사 타임스탬프를 찾지 못했습니다.")

        if self.refine:
            try:
                # Refine the same audio that was aligned. Do not re-run Demucs
                # inside stable-ts.
                model.refine(str(align_audio), result, precision=0.10, verbose=None)
            except Exception as e:
                raise AlignmentError(f"타임스탬프 정밀 보정에 실패했습니다: {e}") from e

        cues, rebuilt = self._cues_from_result(result, lines, language)
        self._validate(cues, lines)

        word_objs = self._flatten_words(result)
        probs = [self._probability(w) for w in word_objs]
        probs = [p for p in probs if p is not None and math.isfinite(p)]
        self.last_stats = AlignmentStats(
            cue_count=len(cues),
            word_count=len(word_objs),
            first_start=cues[0].start,
            last_end=cues[-1].end,
            mean_word_probability=(sum(probs) / len(probs)) if probs else None,
            rebuilt_from_words=rebuilt,
            aligned_audio=aligned_audio_label,
        )
        return cues

    def _cues_from_result(self, result, lines: list[str], language: str) -> tuple[list[Cue], bool]:
        segments = list(getattr(result, "segments", []) or [])

        if len(segments) == len(lines):
            cues: list[Cue] = []
            for seg, original_line in zip(segments, lines):
                words = [w for w in (getattr(seg, "words", []) or []) if self._word_text(w)]
                if words:
                    start = self._word_start(words[0])
                    end = self._word_end(words[-1])
                else:
                    start = float(getattr(seg, "start", 0.0) or 0.0)
                    end = float(getattr(seg, "end", start) or start)
                cues.append(Cue(start=start, end=end, source=original_line, source_language=language))
            return cues, False

        words = self._flatten_words(result)
        if not words:
            raise AlignmentError(
                "음원에서 실제 단어 타임스탬프를 얻지 못했습니다. "
                "잘못된 자막을 만들지 않기 위해 작업을 중단했습니다."
            )
        cues = self._rebuild_lines_from_aligned_words(words, lines, language)
        return cues, True

    @staticmethod
    def _word_text(word) -> str:
        return str(getattr(word, "word", "") or "").strip()

    @staticmethod
    def _word_start(word) -> float:
        return float(getattr(word, "start", 0.0) or 0.0)

    @staticmethod
    def _word_end(word) -> float:
        return float(getattr(word, "end", 0.0) or 0.0)

    @staticmethod
    def _probability(word) -> float | None:
        value = getattr(word, "probability", None)
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def _flatten_words(self, result) -> list:
        words = []
        for seg in getattr(result, "segments", []) or []:
            for w in getattr(seg, "words", []) or []:
                if self._word_text(w):
                    words.append(w)
        return words

    @staticmethod
    def _norm(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).casefold()
        return "".join(ch for ch in text if ch.isalnum())

    def _rebuild_lines_from_aligned_words(self, words: list, lines: list[str], language: str) -> list[Cue]:
        word_norms = [self._norm(self._word_text(w)) for w in words]
        line_norms = [self._norm(line) for line in lines]

        if any(not n for n in line_norms):
            raise AlignmentError("가사 중 타임코드를 만들 수 없는 빈/기호 전용 줄이 있습니다.")

        out: list[Cue] = []
        cursor = 0

        for line_index, (line, target) in enumerate(zip(lines, line_norms)):
            remaining_lines = len(lines) - line_index
            remaining_words = len(words) - cursor
            if remaining_words < remaining_lines:
                raise AlignmentError(
                    "실제 음원에서 검출된 단어 수가 가사 줄 구조보다 부족합니다. "
                    "정확한 싱크를 보장할 수 없어 중단했습니다."
                )

            start_cursor = cursor
            collected = ""
            target_len = len(target)
            max_cursor = len(words) - (remaining_lines - 1)
            while cursor < max_cursor:
                piece = word_norms[cursor]
                before_diff = abs(target_len - len(collected))
                after = collected + piece
                after_diff = abs(target_len - len(after))
                if collected and len(collected) >= target_len:
                    break
                if collected and len(after) > target_len and before_diff < after_diff:
                    break
                collected = after
                cursor += 1
                if len(collected) >= target_len:
                    break

            if cursor == start_cursor:
                cursor += 1

            line_words = words[start_cursor:cursor]
            if not line_words:
                raise AlignmentError(f"{line_index + 1}번째 가사의 실제 타임스탬프를 만들지 못했습니다.")
            out.append(
                Cue(
                    start=self._word_start(line_words[0]),
                    end=self._word_end(line_words[-1]),
                    source=line,
                    source_language=language,
                )
            )

        if cursor < len(words) and out:
            out[-1].end = max(out[-1].end, self._word_end(words[-1]))
        return out

    @staticmethod
    def _validate(cues: list[Cue], lines: list[str]) -> None:
        if len(cues) != len(lines):
            raise AlignmentError(
                f"가사 {len(lines)}줄 중 {len(cues)}줄만 실제 음원과 정렬되었습니다. "
                "가짜 시간 분배 없이 작업을 중단했습니다."
            )
        if not cues:
            raise AlignmentError("정렬된 자막이 없습니다.")

        prev_start = -1.0
        bad = []
        for i, cue in enumerate(cues, 1):
            if not (math.isfinite(cue.start) and math.isfinite(cue.end)):
                bad.append(i)
                continue
            if cue.start < -0.001 or cue.end <= cue.start + 0.02:
                bad.append(i)
                continue
            if cue.start + 0.02 < prev_start:
                bad.append(i)
                continue
            prev_start = cue.start
        if bad:
            shown = ", ".join(map(str, bad[:10]))
            raise AlignmentError(
                f"실제 타임스탬프 검증에 실패한 가사 줄: {shown}. "
                "잘못된 SRT 생성을 막기 위해 중단했습니다."
            )
