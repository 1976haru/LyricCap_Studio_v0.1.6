from __future__ import annotations

from pathlib import Path
import random

from .base import AlignmentResult
from ..aligner import AlignmentError


class StableTSAdapter:
    """Compatibility adapter for LyricCap's installed primary aligner.

    The existing cached Demucs/Whisper environment remains untouched. Retry is
    a fresh line mapping from the cached vocal analysis and never consumes the
    previously rejected timestamps.
    """

    def __init__(self, aligner, *, retry: bool = False):
        self.aligner = aligner
        self.retry = retry
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import stable_whisper
            import torch
        except ImportError as exc:
            raise AlignmentError("stable-ts primary aligner is not installed") from exc
        device = self.aligner.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = stable_whisper.load_model(
                self.aligner.model_name, device=device, dq=(device == "cpu")
            )
        except TypeError:
            self._model = stable_whisper.load_model(self.aligner.model_name, device=device)
        return self._model

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds: tuple[float, float] | None = None) -> AlignmentResult:
        if not lyric_lines:
            return AlignmentResult([], "stable-ts-exact")
        target_audio, _audio_label = self.aligner._prepare_audio(Path(audio_path))
        model = self._load_model()
        options = {
            "original_split": True,
            "token_step": 80 if self.retry else 100,
            "nonspeech_skip": 3.0 if self.retry else 5.0,
            "failure_threshold": 0.30 if self.retry else 0.45,
            "suppress_silence": True,
            "vad": True,
            "vad_threshold": 0.40 if self.retry else 0.35,
            "min_word_dur": 0.06,
            "min_silence_dur": 0.12,
            "nonspeech_error": 0.10,
            "verbose": None,
        }
        random.seed(0)
        try:
            aligned = model.align(str(target_audio), "\n".join(lyric_lines),
                                  language="ja" if language == "ja" else language,
                                  **options)
        except Exception as exc:
            raise AlignmentError(f"stable-ts exact alignment failed: {type(exc).__name__}: {exc}") from exc
        if aligned is None:
            raise AlignmentError("stable-ts exact alignment returned no result")
        cues, _rebuilt = self.aligner._cues_from_result(aligned, lyric_lines, language)
        return AlignmentResult(
            cues=cues,
            engine="stable-ts-exact-retry" if self.retry else "stable-ts-exact",
            confidences=[],
            metadata={"retry": self.retry},
        )
