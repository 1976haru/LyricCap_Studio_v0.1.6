from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable
from .aligner import StableTSAligner
from .audio import audio_duration
from .models import TrackResult
from .srt import write_srt, write_combined
from .translator import build_translator

PROFILES = ["en", "en_ko", "en_ja", "ko", "ja"]


def safe_name(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', '_', text).strip().strip('.')
    return text[:120] or "track"


def required_languages(profiles: list[str]) -> set[str]:
    langs: set[str] = set()
    for p in profiles:
        if p == "en": langs.add("en")
        elif p == "ko": langs.add("ko")
        elif p == "ja": langs.add("ja")
        elif p == "en_ko": langs.update({"en", "ko"})
        elif p == "en_ja": langs.update({"en", "ja"})
    return langs


def process_songs(
    songs,
    output_dir: Path,
    profiles: list[str],
    model_name: str = "base",
    gap_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
    translation_engine: str = "gemini",
    translation_api_key: str = "",
    translation_model: str = "gemini-2.5-flash",
    sync_mode: str = "music_precise",
    refine_timestamps: bool = False,
):
    progress = progress or (lambda _: None)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligner = StableTSAligner(
        model_name=model_name,
        sync_mode=sync_mode,
        refine=refine_timestamps,
        cache_dir=output_dir / ".cache",
        progress=progress,
    )
    translator = None
    results: list[TrackResult] = []
    sync_report: list[dict] = []
    langs_needed = required_languages(profiles)

    needs_translation = any(
        lang != song.source_language
        for song in songs
        for lang in langs_needed
    )
    if needs_translation:
        translator = build_translator(
            translation_engine,
            output_dir / ".cache" / "translations.json",
            api_key=translation_api_key,
            model_name=translation_model,
        )
        if translation_engine == "argos":
            # Fail early and install offline packages before expensive alignment.
            route_pairs = sorted({
                (song.source_language, lang)
                for song in songs
                for lang in langs_needed
                if lang != song.source_language
            })
            for src, dst in route_pairs:
                label = {"en": "영어", "ko": "한국어", "ja": "일본어"}.get(dst, dst)
                progress(f"오프라인 번역 모델 확인: {src} → {label}")
                translator.ensure_route(src, dst)

    for idx, song in enumerate(songs, 1):
        if not song.audio_path:
            raise RuntimeError(f"{song.track_no:02d} {song.title}: 연결된 음원 파일이 없습니다.")
        sync_label = {
            "music_precise": "보컬 분리+VAD 정밀싱크",
            "music_fast": "VAD 빠른싱크",
            "voice": "음성 싱크",
        }.get(sync_mode, sync_mode)
        progress(f"[{idx}/{len(songs)}] 음원 분석/가사 정렬 ({sync_label}): {song.title}")
        cues = aligner.align(song.audio_path, song.lyrics, song.source_language)
        if aligner.last_stats:
            st = aligner.last_stats
            extra = " / 라인 재구성" if st.rebuilt_from_words else ""
            progress(
                f"[{idx}/{len(songs)}] 싱크 완료: {song.title} "
                f"({st.cue_count}줄, {st.first_start:.2f}s~{st.last_end:.2f}s{extra})"
            )
            sync_report.append({
                "trackNo": song.track_no,
                "title": song.title,
                "audio": str(song.audio_path),
                "syncMode": sync_mode,
                "model": model_name,
                "cueCount": st.cue_count,
                "alignedWordCount": st.word_count,
                "firstVocalStart": round(st.first_start, 3),
                "lastVocalEnd": round(st.last_end, 3),
                "meanWordProbability": (round(st.mean_word_probability, 4) if st.mean_word_probability is not None else None),
                "rebuiltFromWordTimestamps": st.rebuilt_from_words,
                "alignedAudio": st.aligned_audio,
                "syncEngine": getattr(st, "engine", "unknown"),
                "lyricMatchRatio": (round(st.match_ratio, 4) if getattr(st, "match_ratio", None) is not None else None),
            })

        for c in cues:
            c.set_text(song.source_language, c.source)

        source_texts = [c.source for c in cues]
        for lang in sorted(langs_needed):
            if lang == song.source_language:
                continue
            if translator is None:
                raise RuntimeError("번역기가 초기화되지 않았습니다.")
            label = {"en": "영어", "ko": "한국어", "ja": "일본어"}[lang]
            mode_label = "AI 자연번역" if translation_engine == "gemini" else "오프라인 번역"
            progress(f"[{idx}/{len(songs)}] {label} {mode_label}: {song.title}")
            translated = translator.translate_many(
                source_texts,
                song.source_language,
                lang,
                title=song.title,
            )
            for c, t in zip(cues, translated):
                c.set_text(lang, t)

        duration = audio_duration(song.audio_path)
        if duration <= 0 and cues:
            duration = max(c.end for c in cues)

        track_dir = output_dir / "tracks"
        stem = f"{song.track_no:02d}_{safe_name(song.title)}"
        for profile in profiles:
            write_srt(track_dir / f"{stem}_{profile}.srt", cues, profile)
        results.append(TrackResult(song=song, cues=cues, duration=duration))


    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "sync_report.json").write_text(
        json.dumps(sync_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    progress("통합 SRT 생성 중...")
    offset = 0.0
    tracks_with_offsets = []
    for r in results:
        tracks_with_offsets.append((r.cues, offset))
        offset += r.duration + gap_seconds

    combined_dir = output_dir / "combined"
    for profile in profiles:
        write_combined(combined_dir / f"PLAYLIST_{profile}.srt", tracks_with_offsets, profile)

    progress("완료")
    return results
