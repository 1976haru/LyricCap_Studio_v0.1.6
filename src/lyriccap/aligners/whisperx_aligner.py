from __future__ import annotations

from pathlib import Path

from .base import AlignmentResult
from ..aligner import AlignmentError, StableTSAligner


class WhisperXAligner:
    DEFAULT_JA_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"

    def __init__(self, model_name: str = "base", device: str = "auto", progress=None,
                 audio_resolver=None):
        self.model_name = model_name
        self.device = device
        self.progress = progress or (lambda _message: None)
        self.audio_resolver = audio_resolver

    def align(self, audio_path: Path, lyric_lines: list[str], language: str,
              bounds: tuple[float, float] | None = None) -> AlignmentResult:
        try:
            import torch
            import whisperx
        except ImportError as exc:
            raise AlignmentError(
                "WhisperX fallback is unavailable. Install the optional 'whisperx' dependency."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.progress(f"WhisperX fallback: {Path(audio_path).name}")
        try:
            if self.audio_resolver is not None:
                audio_path = Path(self.audio_resolver(Path(audio_path)))
            audio = whisperx.load_audio(str(audio_path))
            model = whisperx.load_model(self.model_name, device, compute_type=compute_type,
                                        language=language)
            transcript = model.transcribe(audio, batch_size=8, language=language)
            kwargs = {"language_code": language, "device": device}
            if language == "ja":
                kwargs["model_name"] = self.DEFAULT_JA_MODEL
            align_model, metadata = whisperx.load_align_model(**kwargs)
            aligned = whisperx.align(transcript["segments"], align_model, metadata, audio, device,
                                     return_char_alignments=False)
        except Exception as exc:
            raise AlignmentError(f"WhisperX fallback failed: {type(exc).__name__}: {exc}") from exc

        # Reuse the proven exact-source monotonic mapper. It preserves duplicate
        # occurrence order and uses WhisperX only for timing anchors.
        mapper = StableTSAligner(sync_mode="voice")
        cues, _ = mapper._cues_from_result(aligned, lyric_lines, language)
        probabilities = [
            float(word["score"])
            for segment in aligned.get("segments", [])
            for word in segment.get("words", [])
            if word.get("score") is not None
        ]
        return AlignmentResult(cues=cues, engine="whisperx", confidences=probabilities,
                               metadata={"alignment_model": self.DEFAULT_JA_MODEL if language == "ja" else "default"})
