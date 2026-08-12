from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from .models import Cue
from .vocal_separator import DemucsSeparator, VocalSeparationError


class AlignmentError(RuntimeError):
    """Raised when reliable audio/lyric timing cannot be produced."""


@dataclass
class AlignmentStats:
    cue_count: int
    word_count: int
    first_start: float
    last_end: float
    mean_word_probability: float | None = None
    rebuilt_from_words: bool = True
    aligned_audio: str = "original"
    engine: str = "whisper-asr-anchor"
    match_ratio: float | None = None


@dataclass
class _TimedUnit:
    text: str
    norm: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class _SourceUnit:
    text: str
    norm: str
    line_index: int


class StableTSAligner:
    """Lyric synchronizer using Demucs + OpenAI Whisper word timestamps.

    v0.1.7 deliberately does NOT call stable-ts model.align().  On some
    Windows/PyTorch/stable-ts combinations that forced-alignment path can fail
    inside the library with errors such as::

        TypeError: tuple indices must be integers or slices, not tuple

    Instead we:
      1) isolate vocals with Demucs (precise music mode),
      2) transcribe the vocal stem with the official OpenAI Whisper
         word_timestamps=True pipeline,
      3) monotonically align the user's known lyric tokens to the ASR tokens,
      4) keep the user's exact lyric text and use only the ASR timestamps.

    This is intentionally a different engine, not another wrapper around the
    failing stable-ts forced-align call.
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
            import whisper
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Whisper 또는 PyTorch가 설치되지 않았습니다. START_HERE.bat을 다시 실행하세요."
            ) from e

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress(f"Whisper 모델 로드: {self.model_name} / {device}")
        self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def _prepare_audio(self, audio_path: Path) -> tuple[Path, str]:
        if self.sync_mode == "music_precise":
            try:
                return self.separator.separate(audio_path), "demucs-vocals"
            except VocalSeparationError as e:
                raise AlignmentError(str(e)) from e
        if self.sync_mode in {"music_fast", "voice"}:
            return audio_path, "original"
        raise ValueError(f"알 수 없는 싱크 모드: {self.sync_mode}")

    def align(self, audio_path: Path, lines: list[str], language: str = "en") -> list[Cue]:
        if not lines:
            return []
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AlignmentError(f"음원 파일을 찾을 수 없습니다: {audio_path}")

        model = self._load()
        target_audio, aligned_audio_label = self._prepare_audio(audio_path)
        self.progress(f"보컬 음성인식 + 단어 타임스탬프 분석: {audio_path.name}")

        # A small prompt can help Whisper recognize title/lyric vocabulary, but
        # it is deliberately capped so it does not become a fake transcript.
        prompt = " ".join(lines)
        prompt = prompt[:1200] if len(prompt) > 1200 else prompt

        kwargs = dict(
            language=language if language in {"en", "ko", "ja"} else None,
            word_timestamps=True,
            verbose=None,
            condition_on_previous_text=False,
            temperature=0.0,
            initial_prompt=prompt or None,
            hallucination_silence_threshold=2.0,
        )
        # FP16 is not supported on CPU.  Whisper accepts this in decode_options.
        try:
            import torch
            kwargs["fp16"] = bool(torch.cuda.is_available() and str(getattr(model, "device", "")).startswith("cuda"))
        except Exception:
            kwargs["fp16"] = False

        try:
            result = model.transcribe(str(target_audio), **kwargs)
        except TypeError:
            # Older compatible Whisper builds may not expose
            # hallucination_silence_threshold. Retry with the documented core
            # word timestamp arguments instead of failing the whole job.
            kwargs.pop("hallucination_silence_threshold", None)
            try:
                result = model.transcribe(str(target_audio), **kwargs)
            except Exception as e:
                raise AlignmentError(
                    "Whisper 음원 분석에 실패했습니다.\n"
                    f"분석 대상: {target_audio}\n"
                    f"원본 오류: {type(e).__name__}: {e}"
                ) from e
        except Exception as e:
            raise AlignmentError(
                "Whisper 음원 분석에 실패했습니다.\n"
                f"분석 대상: {target_audio}\n"
                f"원본 오류: {type(e).__name__}: {e}"
            ) from e

        asr_units = self._extract_asr_units(result, language)
        if not asr_units:
            raise AlignmentError(
                "음원에서 단어 타임스탬프를 하나도 얻지 못했습니다. "
                "보컬 분리 결과가 비어 있거나 Whisper가 보컬을 인식하지 못했습니다."
            )

        source_units, line_unit_counts = self._make_source_units(lines, language)
        if not source_units:
            raise AlignmentError("가사에서 정렬할 수 있는 문자/단어를 찾지 못했습니다.")

        mapping, matched = self._global_align(source_units, asr_units)
        match_ratio = matched / max(1, len(source_units))
        self.progress(f"가사-보컬 매칭률: {match_ratio*100:.1f}%")

        # A very low ratio means line timing would mostly be guesswork.  Stop
        # instead of generating another misleading SRT.
        min_ratio = 0.16 if language == "ja" else 0.20
        if match_ratio < min_ratio:
            raise AlignmentError(
                f"가사와 실제 보컬의 매칭률이 너무 낮습니다 ({match_ratio*100:.1f}%).\n"
                "잘못된 자막을 만들지 않기 위해 중단했습니다. "
                "음원과 JSON의 곡이 같은지 확인하거나 small/medium 모델을 사용해 보세요."
            )

        cues = self._build_line_cues(
            lines=lines,
            language=language,
            source_units=source_units,
            line_unit_counts=line_unit_counts,
            asr_units=asr_units,
            mapping=mapping,
        )
        self._validate(cues, lines)

        probs = [u.probability for u in asr_units if u.probability is not None and math.isfinite(u.probability)]
        self.last_stats = AlignmentStats(
            cue_count=len(cues),
            word_count=len(asr_units),
            first_start=cues[0].start,
            last_end=cues[-1].end,
            mean_word_probability=(sum(probs) / len(probs)) if probs else None,
            rebuilt_from_words=True,
            aligned_audio=aligned_audio_label,
            engine="openai-whisper-asr-anchor",
            match_ratio=match_ratio,
        )
        return cues

    @staticmethod
    def _norm_token(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text)).casefold()
        return "".join(ch for ch in text if ch.isalnum())

    def _tokenize_text(self, text: str, language: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text)
        if language == "ja":
            # Japanese Whisper word chunks are inconsistent. Character units
            # make the monotonic matcher much more stable across chunks.
            return [ch for ch in text if ch.isalnum()]
        # Keep apostrophes inside English words; punctuation is normalized away.
        raw = re.findall(r"[\w]+(?:['’][\w]+)?", text, flags=re.UNICODE)
        return [t for t in raw if self._norm_token(t)]

    def _make_source_units(self, lines: list[str], language: str) -> tuple[list[_SourceUnit], list[int]]:
        units: list[_SourceUnit] = []
        counts: list[int] = []
        for li, line in enumerate(lines):
            toks = self._tokenize_text(line, language)
            counts.append(len(toks))
            for tok in toks:
                n = self._norm_token(tok)
                if n:
                    units.append(_SourceUnit(tok, n, li))
        return units, counts

    def _extract_asr_units(self, result, language: str) -> list[_TimedUnit]:
        segments = result.get("segments", []) if isinstance(result, dict) else getattr(result, "segments", []) or []
        out: list[_TimedUnit] = []
        for seg in segments:
            words = seg.get("words", []) if isinstance(seg, dict) else getattr(seg, "words", []) or []
            if not words:
                text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
                start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
                end = seg.get("end", start) if isinstance(seg, dict) else getattr(seg, "end", start)
                words = [{"word": text, "start": start, "end": end, "probability": None}]

            for word in words:
                text = word.get("word", "") if isinstance(word, dict) else getattr(word, "word", "")
                start = float(word.get("start", 0.0) if isinstance(word, dict) else getattr(word, "start", 0.0) or 0.0)
                end = float(word.get("end", start) if isinstance(word, dict) else getattr(word, "end", start) or start)
                p = word.get("probability", None) if isinstance(word, dict) else getattr(word, "probability", None)
                try:
                    p = float(p) if p is not None else None
                except Exception:
                    p = None

                toks = self._tokenize_text(str(text), language)
                if not toks:
                    continue
                dur = max(0.02, end - start)
                for i, tok in enumerate(toks):
                    n = self._norm_token(tok)
                    if not n:
                        continue
                    a = start + dur * (i / len(toks))
                    b = start + dur * ((i + 1) / len(toks))
                    out.append(_TimedUnit(tok, n, a, max(a + 0.01, b), p))
        return out

    @staticmethod
    def _token_score(a: str, b: str) -> float:
        if a == b:
            return 3.0
        if not a or not b:
            return -1.6
        # Single-character fuzzy matches are not useful.
        if len(a) == 1 or len(b) == 1:
            return -1.6
        ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
        if ratio >= 0.85:
            return 2.1
        if ratio >= 0.67:
            return 1.0
        if ratio >= 0.52 and min(len(a), len(b)) >= 4:
            return 0.25
        return -1.6

    def _global_align(self, source: list[_SourceUnit], asr: list[_TimedUnit]) -> tuple[dict[int, int], int]:
        """Needleman-Wunsch style monotonic token alignment.

        Uses only two score rows and a bytearray backtrace so a few hundred
        lyric/ASR units stay lightweight on Windows.
        """
        n, m = len(source), len(asr)
        gap = -1.0
        cols = m + 1
        bt = bytearray((n + 1) * cols)  # 0 diag, 1 up, 2 left

        prev = [j * gap for j in range(cols)]
        for j in range(1, cols):
            bt[j] = 2

        for i in range(1, n + 1):
            cur = [0.0] * cols
            cur[0] = i * gap
            bt[i * cols] = 1
            s = source[i - 1].norm
            for j in range(1, cols):
                ms = self._token_score(s, asr[j - 1].norm)
                diag = prev[j - 1] + ms
                up = prev[j] + gap
                left = cur[j - 1] + gap
                if diag >= up and diag >= left:
                    cur[j] = diag
                    bt[i * cols + j] = 0
                elif up >= left:
                    cur[j] = up
                    bt[i * cols + j] = 1
                else:
                    cur[j] = left
                    bt[i * cols + j] = 2
            prev = cur

        mapping: dict[int, int] = {}
        matched = 0
        i, j = n, m
        while i > 0 or j > 0:
            direction = bt[i * cols + j]
            if i > 0 and j > 0 and direction == 0:
                score = self._token_score(source[i - 1].norm, asr[j - 1].norm)
                # Keep only genuinely useful diagonal matches; weak diagonals
                # may be chosen to bridge gaps but should not anchor timing.
                if score >= 0.25:
                    mapping[i - 1] = j - 1
                    matched += 1
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or direction == 1):
                i -= 1
            elif j > 0:
                j -= 1
            else:
                break
        return mapping, matched

    def _build_line_cues(
        self,
        *,
        lines: list[str],
        language: str,
        source_units: list[_SourceUnit],
        line_unit_counts: list[int],
        asr_units: list[_TimedUnit],
        mapping: dict[int, int],
    ) -> list[Cue]:
        per_line: list[list[tuple[int, int]]] = [[] for _ in lines]
        for src_idx, asr_idx in mapping.items():
            per_line[source_units[src_idx].line_index].append((src_idx, asr_idx))

        raw: list[tuple[float, float] | None] = [None] * len(lines)
        confidence: list[float] = [0.0] * len(lines)
        for li, pairs in enumerate(per_line):
            total = max(1, line_unit_counts[li])
            confidence[li] = len(pairs) / total
            if pairs and (len(pairs) >= 2 or confidence[li] >= 0.34 or total == 1):
                ids = sorted(a for _, a in pairs)
                raw[li] = (asr_units[ids[0]].start, asr_units[ids[-1]].end)

        anchor_ids = [i for i, r in enumerate(raw) if r is not None]
        if not anchor_ids:
            raise AlignmentError(
                "Whisper는 보컬을 인식했지만 가사 줄과 연결되는 확실한 지점을 찾지 못했습니다."
            )

        # Estimate a conservative seconds-per-token from anchored lines.
        rates = []
        for i in anchor_ids:
            a, b = raw[i]  # type: ignore[misc]
            rates.append((b - a) / max(1, line_unit_counts[i]))
        rates.sort()
        sec_per_unit = rates[len(rates)//2] if rates else 0.45
        sec_per_unit = min(1.2, max(0.22, sec_per_unit))

        # Fill missing line blocks only between/around real anchors.  This is
        # local interpolation, not whole-song even timing.
        def fill_block(lo: int, hi: int, left_time: float, right_time: float):
            if lo > hi:
                return
            weights = [max(1, line_unit_counts[k]) for k in range(lo, hi + 1)]
            available = max(0.3 * len(weights), right_time - left_time)
            total_w = sum(weights)
            cur = left_time
            for k, w in zip(range(lo, hi + 1), weights):
                dur = available * (w / total_w)
                dur = max(0.35, dur)
                raw[k] = (cur, min(right_time, cur + dur))
                cur = raw[k][1]  # type: ignore[index]

        first = anchor_ids[0]
        if first > 0:
            first_start = raw[first][0]  # type: ignore[index]
            est = sum(max(1, line_unit_counts[k]) for k in range(first)) * sec_per_unit
            left = max(0.0, first_start - est)
            fill_block(0, first - 1, left, max(left + 0.35 * first, first_start - 0.05))

        for a_idx, b_idx in zip(anchor_ids, anchor_ids[1:]):
            if b_idx - a_idx <= 1:
                continue
            left = raw[a_idx][1] + 0.03  # type: ignore[index]
            right = raw[b_idx][0] - 0.03  # type: ignore[index]
            if right <= left:
                right = left + 0.45 * (b_idx - a_idx - 1)
            fill_block(a_idx + 1, b_idx - 1, left, right)

        last = anchor_ids[-1]
        if last < len(lines) - 1:
            cur = raw[last][1] + 0.03  # type: ignore[index]
            for k in range(last + 1, len(lines)):
                dur = max(0.5, max(1, line_unit_counts[k]) * sec_per_unit)
                raw[k] = (cur, cur + dur)
                cur += dur + 0.03

        cues: list[Cue] = []
        prev_end = 0.0
        for i, line in enumerate(lines):
            if raw[i] is None:
                raise AlignmentError(f"{i+1}번째 가사 줄의 시간을 만들지 못했습니다.")
            start, end = raw[i]
            start = max(0.0, start)
            end = max(start + 0.25, end)
            # Keep SRT lines monotonic. A tiny overlap from Whisper is resolved
            # at the midpoint instead of allowing CapCut subtitle pile-ups.
            if cues and start < prev_end:
                midpoint = max(cues[-1].start + 0.15, (start + prev_end) / 2.0)
                cues[-1].end = max(cues[-1].start + 0.15, midpoint - 0.01)
                start = midpoint + 0.01
                end = max(start + 0.25, end)
            cue = Cue(start=start, end=end, source=line, source_language=language)
            cues.append(cue)
            prev_end = cue.end
        return cues

    def _cues_from_result(self, result, lines: list[str], language: str):
        """Compatibility helper for older LyricCap tests/tools.

        Converts an already timestamped ASR result into exact source-line cues
        through the same monotonic matcher used by v0.1.7.
        """
        asr_units = self._extract_asr_units(result, language)
        if not asr_units:
            raise AlignmentError(
                "음원에서 실제 단어 타임스탬프를 얻지 못했습니다. "
                "잘못된 자막을 만들지 않기 위해 작업을 중단했습니다."
            )
        source_units, counts = self._make_source_units(lines, language)
        mapping, matched = self._global_align(source_units, asr_units)
        if matched == 0:
            raise AlignmentError("가사와 인식 결과를 연결할 수 없습니다.")
        cues = self._build_line_cues(
            lines=lines, language=language, source_units=source_units,
            line_unit_counts=counts, asr_units=asr_units, mapping=mapping,
        )
        segs = result.get("segments", []) if isinstance(result, dict) else getattr(result, "segments", []) or []
        rebuilt = len(segs) != len(lines)
        return cues, rebuilt

    @staticmethod
    def _validate(cues: list[Cue], lines: list[str]) -> None:
        if len(cues) != len(lines):
            raise AlignmentError(f"가사 줄 수({len(lines)})와 자막 줄 수({len(cues)})가 다릅니다.")
        last = -1.0
        for i, cue in enumerate(cues, 1):
            if not (math.isfinite(cue.start) and math.isfinite(cue.end)):
                raise AlignmentError(f"{i}번째 자막에 유효하지 않은 시간값이 있습니다.")
            if cue.start < 0 or cue.end <= cue.start:
                raise AlignmentError(f"{i}번째 자막 시간 범위가 잘못되었습니다: {cue.start}~{cue.end}")
            if cue.start < last - 0.02:
                raise AlignmentError(f"{i}번째 자막 시간이 앞 자막보다 뒤섞였습니다.")
            last = cue.start
