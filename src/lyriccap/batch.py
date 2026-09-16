from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable
from .aligner import StableTSAligner
from .audio import audio_duration
from .languages import language_name
from .models import TrackResult
from .srt import write_srt, write_combined, write_titles_srt
from .translator import build_translator

PROFILES = ["en", "en_ko", "en_ja", "ko", "ja"]


class JobCancelled(RuntimeError):
    """사용자가 중단 버튼을 눌렀을 때. 오류가 아니라 정상적인 종료입니다."""


def _hhmmss(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


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
    should_cancel: Callable[[], bool] | None = None,
    title_seconds: float = 5.0,
    fallback_enabled: bool = True,
):
    progress = progress or (lambda _: None)
    should_cancel = should_cancel or (lambda: False)

    def check_cancel():
        if should_cancel():
            raise JobCancelled(
                "사용자가 작업을 중단했습니다. "
                "지금까지 분석/번역한 결과는 캐시에 남아 있어, 다시 실행하면 이어서 진행합니다."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    aligner = StableTSAligner(
        model_name=model_name,
        sync_mode=sync_mode,
        refine=refine_timestamps,
        cache_dir=output_dir / ".cache",
        progress=progress,
        fallback_enabled=fallback_enabled,
    )
    translator = None
    results: list[TrackResult] = []
    sync_report: list[dict] = []
    duration_warnings: list[dict] = []
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
            progress=progress,
            check_cancel=check_cancel,
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
                label = language_name(dst)
                progress(f"오프라인 번역 모델 확인: {src} → {label}")
                translator.ensure_route(src, dst)

    for idx, song in enumerate(songs, 1):
        check_cancel()
        if not song.audio_path:
            raise RuntimeError(f"{song.track_no:02d} {song.title}: 연결된 음원 파일이 없습니다.")
        sync_label = {
            "music_precise": "보컬 분리+VAD 정밀싱크",
            "music_fast": "VAD 빠른싱크",
            "voice": "음성 싱크",
        }.get(sync_mode, sync_mode)
        progress(f"[{idx}/{len(songs)}] 음원 분석/가사 정렬 ({sync_label}): {song.title}")
        cues = aligner.align(song.audio_path, song.lyrics, song.source_language)
        check_cancel()
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
        target_langs = [l for l in sorted(langs_needed) if l != song.source_language]
        if target_langs and translator is None:
            raise RuntimeError("번역기가 초기화되지 않았습니다.")

        if target_langs and hasattr(translator, "translate_multi"):
            # Gemini 경로. 모든 언어를 API 호출 한 번으로 처리해 무료 등급의
            # 분당 요청 한도에 걸리는 일을 줄입니다. 곡당 3회 -> 1회.
            names = " + ".join(language_name(l) for l in target_langs)
            progress(f"[{idx}/{len(songs)}] {names} AI 자연번역: {song.title}")
            bundle = translator.translate_multi(
                source_texts,
                song.source_language,
                target_langs,
                title=song.title,
            )
            for lang in target_langs:
                for c, t in zip(cues, bundle.get(lang, [])):
                    c.set_text(lang, t)
        else:
            # Argos 등 한 언어씩 처리하는 엔진.
            for lang in target_langs:
                label = language_name(lang)
                progress(f"[{idx}/{len(songs)}] {label} 오프라인 번역: {song.title}")
                translated = translator.translate_many(
                    source_texts,
                    song.source_language,
                    lang,
                    title=song.title,
                )
                for c, t in zip(cues, translated):
                    c.set_text(lang, t)

        duration = audio_duration(song.audio_path)
        duration_source = "audio-file"
        if duration <= 0:
            # 기존 버전은 여기서 조용히 "마지막 자막 끝시간"을 곡 길이로 썼습니다.
            # 실제 곡에는 마지막 가사 뒤에 간주/아웃트로가 남아 있으므로 곡마다
            # 수십 초씩 짧아지고, 그 오차가 통합 SRT에서 누적되어 뒤 곡의 자막이
            # 전부 앞으로 밀립니다. 이제는 경고를 남겨 눈에 보이게 합니다.
            duration = max((c.end for c in cues), default=0.0)
            duration_source = "last-cue-end(부정확)"
            duration_warnings.append({
                "trackNo": song.track_no,
                "title": song.title,
                "audio": str(song.audio_path),
                "fallbackDuration": round(duration, 3),
            })
            progress(
                f"⚠ [{idx}/{len(songs)}] 경고: '{song.title}'의 음원 길이를 읽지 못했습니다. "
                f"마지막 자막 끝({duration:.1f}초)으로 대체합니다. "
                f"이 곡 이후의 통합 SRT 자막이 앞으로 밀릴 수 있습니다."
            )

        if sync_report and sync_report[-1].get("trackNo") == song.track_no:
            sync_report[-1]["audioDuration"] = round(duration, 3)
            sync_report[-1]["durationSource"] = duration_source

        track_dir = output_dir / "tracks"
        stem = f"{song.track_no:02d}_{safe_name(song.title)}"
        for profile in profiles:
            write_srt(track_dir / f"{stem}_{profile}.srt", cues, profile)
        results.append(TrackResult(song=song, cues=cues, duration=duration))


    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if translator is not None:
        used_model = getattr(translator, "model_name", "")
        if used_model and used_model != translation_model:
            progress(
                f"참고: 요청한 모델 '{translation_model}' 대신 '{used_model}'로 번역했습니다. "
                f"이 이름을 'AI 모델' 칸에 넣으면 다음부터 바로 사용합니다."
            )

    (diagnostics_dir / "sync_report.json").write_text(
        json.dumps(sync_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    progress("통합 SRT 생성 중...")
    offset = 0.0
    tracks_with_offsets = []
    title_tracks: list[tuple] = []
    offset_report: list[dict] = []
    for r in results:
        tracks_with_offsets.append((r.cues, offset))
        title_tracks.append((r.song, offset, r.duration))
        offset_report.append({
            "trackNo": r.song.track_no,
            "title": r.song.title,
            "audio": str(r.song.audio_path),
            "startsAt": _hhmmss(offset),
            "startsAtSeconds": round(offset, 3),
            "durationSeconds": round(r.duration, 3),
            "lastCueEndSeconds": round(max((c.end for c in r.cues), default=0.0), 3),
            "trailingInstrumentalSeconds": round(
                max(0.0, r.duration - max((c.end for c in r.cues), default=0.0)), 3
            ),
        })
        offset += r.duration + gap_seconds

    (diagnostics_dir / "playlist_offsets.json").write_text(
        json.dumps(
            {
                "gapSeconds": gap_seconds,
                "totalSeconds": round(offset, 3),
                "totalFormatted": _hhmmss(offset),
                "durationWarnings": duration_warnings,
                "tracks": offset_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    combined_dir = output_dir / "combined"
    for profile in profiles:
        write_combined(combined_dir / f"PLAYLIST_{profile}.srt", tracks_with_offsets, profile)

    if title_seconds > 0:
        progress("곡 제목 자막 생성 중...")
        write_titles_srt(combined_dir / "PLAYLIST_titles.srt", title_tracks, title_seconds)

    if duration_warnings:
        names = ", ".join(w["title"] for w in duration_warnings[:5])
        progress(
            f"완료 (경고 {len(duration_warnings)}곡: {names} — 음원 길이를 읽지 못해 "
            f"통합 SRT 싱크가 밀릴 수 있습니다)"
        )
    else:
        progress("완료")
    return results
