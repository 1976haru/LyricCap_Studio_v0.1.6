from __future__ import annotations

import json
import re
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .aligner import StableTSAligner
from .aligners import (WhisperXExactAligner, WhisperXExactSectionAligner,
                       WhisperXExactRecoveryAligner)
from .alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from .audio import audio_duration
from .languages import SOURCE_LANGUAGES, language_name
from .models import Cue, TrackResult
from .srt import write_combined, write_srt, write_titles_srt
from .translator import build_translator

APP_VERSION = "v0.2.1"
PIPELINE_VERSION = "acoustic_truth_v3_timebase_safe"

TIMEBASE_FAILURE_CODES = {
    "SECTION_TIMEBASE_CONFLICT", "SECTION_NOT_REBASED", "SECTION_DOUBLE_REBASE",
    "SECTION_OUTSIDE_WINDOW", "SECTION_WINDOW_CONFLICT",
}


class JobCancelled(RuntimeError):
    pass


def _hhmmss(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text).strip().strip(".")[:120] or "track"


def _popup_reason(primary: dict, fallback: str) -> str:
    code = primary.get("code")
    message = primary.get("message", "")
    overlap = re.search(r"([0-9]+(?:\.[0-9]+)?) seconds", message)
    amount = f"({overlap.group(1)}초 중첩)" if overlap else ""
    labels = {
        "SECTION_TIMEBASE_CONFLICT": "섹션 시간축 충돌",
        "SECTION_NOT_REBASED": "섹션 절대시간 변환 누락",
        "SECTION_DOUBLE_REBASE": "섹션 시간 오프셋 중복 적용",
        "SECTION_OUTSIDE_WINDOW": "섹션 검색 구간 이탈",
        "SECTION_WINDOW_CONFLICT": "섹션 검색 구간 충돌",
    }
    return labels.get(code, message or fallback) + amount


def required_languages(profiles: list[str]) -> set[str]:
    """Return the language codes named by profiles, rejecting malformed input."""
    required: set[str] = set()
    allowed = set(SOURCE_LANGUAGES)
    for profile in profiles:
        languages = profile.split("_")
        if not languages or any(not lang or lang not in allowed for lang in languages):
            raise ValueError(f"Unsupported subtitle profile: {profile}")
        if len(languages) > 2 or len(set(languages)) != len(languages):
            raise ValueError(f"Invalid subtitle profile: {profile}")
        required.update(languages)
    return required


def profiles_for_source(source_language: str) -> list[str]:
    """Profiles shown in the UI for one resolved source language."""
    if source_language not in SOURCE_LANGUAGES:
        raise ValueError(f"Unsupported source language: {source_language}")
    preferred_targets = {
        "en": ["ko", "ja"],
        "fr": ["ko", "ja"],
        "ja": ["ko", "en"],
        "ko": ["ja", "en"],
    }.get(source_language, ["ko", "ja", "en"])
    targets = [lang for lang in preferred_targets if lang != source_language]
    profiles = [source_language, *(f"{source_language}_{lang}" for lang in targets)]
    profiles.extend(lang for lang in ("ko", "ja", "en") if lang != source_language)
    return profiles


def default_profiles_for_source(source_language: str) -> list[str]:
    if source_language == "en":
        return ["en_ko", "en_ja"]
    if source_language == "fr":
        return ["fr", "fr_ko", "fr_ja"]
    return [source_language]


def _read_srt(path: Path) -> list[Cue]:
    def seconds(value):
        h, m, rest = value.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and " --> " in lines[1]:
            start, end = lines[1].split(" --> ", 1)
            cues.append(Cue(seconds(start), seconds(end), "\n".join(lines[2:])))
    return cues


def _diagnostic_entry(song, duration, outcome) -> dict:
    evidence = outcome.result.evidence if outcome.result else None
    all_issues = [issue for attempt in outcome.attempts
                  for issue in (attempt.report.issues if attempt.report else [])]
    primary = next((issue for issue in all_issues if issue.code in TIMEBASE_FAILURE_CODES),
                   next((issue for issue in outcome.report.issues if issue.severity == "HARD"), None))
    section_records = [record for attempt in outcome.attempts
                       for record in attempt.metadata.get("section_records", [])]
    return {
        "track_no": song.track_no, "title": song.title,
        "duration": round(duration, 3),
        "lyric_count_expected": len(song.lyrics),
        "lyric_count_aligned": outcome.report.stats.get("lyric_count_aligned", 0),
        "engine": outcome.result.engine if outcome.result else "none",
        "retry_count": outcome.retry_count, "state": outcome.state.value,
        "history": outcome.history, "quality_score": outcome.report.score,
        "attempts": [attempt.to_dict() for attempt in outcome.attempts],
        **outcome.report.stats,
        "issues": [issue.to_dict() for issue in outcome.report.issues],
        "status": outcome.report.status,
        "primary_failure": None if primary is None else primary.to_dict(),
        "sections": section_records,
        "direct_line_ratio": None if evidence is None else evidence.direct_line_ratio,
        "unit_coverage": None if evidence is None else evidence.unit_coverage,
        "failed_line_indices": sorted({issue.cue_index for attempt in outcome.attempts
                                        for issue in (attempt.report.issues if attempt.report else [])
                                        if issue.cue_index is not None}),
        "failed_section_names": sorted({section["name"] for attempt in outcome.attempts
                                         for section in attempt.metadata.get("failed_sections", [])}),
        "whole_track_exact": next((a.status for a in outcome.attempts if a.mode == "WHOLE_TRACK"), "NOT_RUN"),
        "section_exact": next((a.status for a in outcome.attempts if a.mode == "SECTION_WINDOWS"), "NOT_RUN"),
        "recovery": next((a.status for a in outcome.attempts if a.mode == "FAILED_SECTION_ONLY"), "NOT_RUN"),
        "aligned_audio": outcome.result.metadata.get("aligned_audio") if outcome.result else None,
        "source_vs_asr_similarity": outcome.result.metadata.get("source_vs_asr_similarity") if outcome.result else None,
        "source_coverage_by_asr": outcome.result.metadata.get("source_coverage_by_asr") if outcome.result else None,
        "probable_missing_source_lines": outcome.result.metadata.get("probable_missing_source_lines", []) if outcome.result else [],
        "probable_extra_vocal_ranges": outcome.result.metadata.get("probable_extra_vocal_ranges", []) if outcome.result else [],
        "probable_repeat_ranges": outcome.result.metadata.get("probable_repeat_ranges", []) if outcome.result else [],
        "vocal_activity_ranges": outcome.result.metadata.get("vocal_activity_ranges", []) if outcome.result else [],
        "exact_evidence": None if evidence is None else {
            "source_unit_count": evidence.source_unit_count,
            "aligned_unit_count": evidence.aligned_unit_count,
            "unit_coverage": evidence.unit_coverage,
            "direct_line_count": evidence.direct_line_count,
            "direct_line_ratio": evidence.direct_line_ratio,
            "interpolated_line_count": evidence.interpolated_line_count,
            "interpolated_line_ratio": evidence.interpolated_line_ratio,
            "max_interpolated_run": evidence.max_interpolated_run,
        },
    }


def _write_diagnostics(output_dir: Path, report: list[dict]) -> Path:
    directory = output_dir / "_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sync_report.json").write_text(
        json.dumps({"tracks": report}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines: list[str] = []
    for item in report:
        primary = item.get("primary_failure")
        primary_text = "" if not primary else f" {primary['code']}: {primary['message']}"
        lines.append(f"{item['track_no']:02d} {item['state']} {item['engine']} "
                     f"score={item['quality_score']:.0f} {item['title']}{primary_text}")
        for section in item.get("sections", []):
            lines.append(
                f"  {section['section_name']} lines={section['source_start_line']}..{section['source_end_line']} "
                f"local={section['first_local_time']}..{section['last_local_time']} "
                f"window={section['window_start_abs']}..{section['window_end_abs']} "
                f"rebased={section['first_absolute_time']}..{section['last_absolute_time']} "
                f"offset={section['rebase_offset']} stitch={section['stitch_status']}")
        for attempt in item["attempts"]:
            detail = attempt["error_message"]
            if not detail and attempt["report"] and attempt["report"]["issues"]:
                detail = "; ".join(x["message"] for x in attempt["report"]["issues"][:3])
            lines.append(f"  attempt {attempt['attempt_no']} {attempt['engine']} "
                         f"{attempt['status']}: {detail or 'exact evidence accepted'}")
    (directory / "sync_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _quarantine_stale_playlists(output_dir: Path, diagnostics: Path) -> None:
    """Prevent a failed run from leaving an older SRT under the success name."""
    combined = output_dir / "combined"
    if not combined.exists():
        return
    quarantine = diagnostics / "previous_success_outputs"
    for source in combined.glob("PLAYLIST_*.srt"):
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / source.name
        suffix = 1
        while target.exists():
            target = quarantine / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        source.replace(target)


def process_songs(
    songs, output_dir: Path, profiles: list[str], model_name: str = "base",
    gap_seconds: float = 0.0, progress: Callable[[str], None] | None = None,
    translation_engine: str = "gemini", translation_api_key: str = "",
    translation_model: str = "gemini-2.5-flash", sync_mode: str = "music_precise",
    refine_timestamps: bool = False, should_cancel: Callable[[], bool] | None = None,
    title_seconds: float = 5.0, fallback_enabled: bool = True,
    primary_aligner=None, retry_aligner=None, fallback_aligner=None,
    lyrics_file: Path | None = None, strict: bool = True,
    retry_failed_only: bool = False,
):
    """Align each audio independently and commit output only after validation."""
    progress = progress or (lambda _message: None)
    should_cancel = should_cancel or (lambda: False)
    run_started_at = datetime.now(timezone.utc).isoformat()

    def check_cancel():
        if should_cancel():
            raise JobCancelled("The user cancelled the job; cached analysis remains reusable.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_songs = list(songs)
    previous_report = []
    if retry_failed_only:
        manifest_path = output_dir / "_diagnostics" / "run_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("No previous run manifest is available for failed-track retry")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed_numbers = set(previous.get("failed_track_numbers", []))
        if not failed_numbers:
            raise RuntimeError("The previous run has no failed tracks")
        previous_report = list(previous.get("tracks", []))
        songs = [song for song in all_songs if song.track_no in failed_numbers]
        # Only versioned/derived timing artifacts are invalidated. Vocals, ASR
        # diagnostics and translations live in sibling caches and remain intact.
        derived = output_dir / ".cache" / "alignment" / "derived"
        if derived.exists():
            for song in songs:
                for path in derived.glob(f"{song.track_no:02d}_*"):
                    if path.is_file():
                        path.unlink()
    legacy = StableTSAligner(model_name=model_name, sync_mode=sync_mode,
                             refine=refine_timestamps, cache_dir=output_dir / ".cache",
                             progress=progress, fallback_enabled=fallback_enabled)
    primary = primary_aligner or WhisperXExactAligner(
        model_name=model_name, progress=progress, cache_dir=output_dir / ".cache" / "alignment",
        audio_resolver=lambda path: legacy._prepare_audio(path)[0],
    )
    retry = retry_aligner or WhisperXExactSectionAligner(primary)
    fallback = fallback_aligner or WhisperXExactRecoveryAligner(primary)

    langs_needed = required_languages(profiles)
    needs_translation = any(lang != song.source_language for song in songs for lang in langs_needed)
    translator = None
    if needs_translation:
        translator = build_translator(translation_engine, output_dir / ".cache" / "translations.json",
                                      api_key=translation_api_key, model_name=translation_model,
                                      progress=progress, check_cancel=check_cancel)
        if translation_engine == "argos":
            pairs = {(song.source_language, lang) for song in songs for lang in langs_needed
                     if lang != song.source_language}
            for source, target in sorted(pairs):
                progress(f"Checking offline translation model: {source} -> {language_name(target)}")
                translator.ensure_route(source, target)

    # Phase B: translate/cache every source line before timing begins. Translation
    # is text-only and remains reusable even when exact alignment later fails.
    translations_by_track: dict[int, dict[str, list[str]]] = {}
    translation_failures_by_track: dict[int, list[str]] = {}
    for song in songs:
        translations_by_track[song.track_no] = {song.source_language: list(song.lyrics)}
        translation_failures_by_track[song.track_no] = []
        target_languages = sorted(langs_needed - {song.source_language})
        if target_languages and translation_engine == "gemini" and hasattr(translator, "translate_multi"):
            check_cancel()
            try:
                translated_by_language = translator.translate_multi(
                    song.lyrics, song.source_language, target_languages, title=song.title,
                )
                for lang in target_languages:
                    translated = translated_by_language.get(lang, [])
                    if len(translated) != len(song.lyrics):
                        raise RuntimeError(
                            f"{lang} translated line count {len(translated)} "
                            f"!= source lyric count {len(song.lyrics)}"
                        )
                    translations_by_track[song.track_no][lang] = translated
            except Exception as exc:
                translation_failures_by_track[song.track_no].extend(target_languages)
                progress(f"{song.track_no:02d} MULTILINGUAL TRANSLATION FAIL - {exc}")
        else:
            for lang in target_languages:
                check_cancel()
                try:
                    translated = translator.translate_many(
                        song.lyrics, song.source_language, lang, title=song.title,
                    )
                    if len(translated) != len(song.lyrics):
                        raise RuntimeError(
                            f"translated line count {len(translated)} != source lyric count {len(song.lyrics)}"
                        )
                    translations_by_track[song.track_no][lang] = translated
                except Exception as exc:
                    translation_failures_by_track[song.track_no].append(lang)
                    progress(f"{song.track_no:02d} {lang.upper()} TRANSLATION FAIL - {exc}")

    results: list[TrackResult] = []
    failures: list[TrackResult] = []
    report: list[dict] = []
    for index, song in enumerate(songs, 1):
        check_cancel()
        if not song.audio_path:
            raise RuntimeError(f"{song.track_no:02d} {song.title}: no audio file is connected")
        duration = audio_duration(song.audio_path)
        transaction = TrackAlignmentTransaction(
            primary, retry, fallback,
            progress=lambda state, no=song.track_no: progress(f"{no:02d} {state}"),
            strict=strict,
        )
        progress(f"[{index}/{len(songs)}] per-track alignment: {song.title}")
        outcome = transaction.run(audio_path=Path(song.audio_path), lyric_lines=song.lyrics,
                                  language=song.source_language, track_duration=duration,
                                  instrumental_after=song.instrumental_after,
                                  sections=song.sections,
                                  start_stage=2 if retry_failed_only else 1)
        report.append(_diagnostic_entry(song, duration, outcome))
        entry = report[-1]
        primary_failure = entry.get("primary_failure")
        detail = primary_failure["message"] if primary_failure else ", ".join(
            issue.message for issue in outcome.report.issues[:1])
        progress(f"{song.track_no:02d} {outcome.state.value} score {outcome.report.score:.0f}" +
                 (f" - {detail}" if detail else ""))
        if outcome.state not in {TrackAlignmentState.PASS, TrackAlignmentState.PASS_WITH_WARNINGS} or outcome.result is None:
            pending = output_dir / "translations_pending_sync"
            for lang, translated in translations_by_track[song.track_no].items():
                if lang != song.source_language:
                    pending.mkdir(parents=True, exist_ok=True)
                    (pending / f"{song.track_no:02d}_{safe_name(song.title)}_{lang}.txt").write_text(
                        "\n".join(translated) + "\n", encoding="utf-8")
            failures.append(TrackResult(song=song, duration=duration, warning=detail,
                                        state="FAIL", retry_count=outcome.retry_count,
                                        quality_score=outcome.report.score))
            continue

        cues = outcome.result.cues
        for cue in cues:
            cue.set_text(song.source_language, cue.source)
        translation_failures = translation_failures_by_track[song.track_no]
        for lang, translated in translations_by_track[song.track_no].items():
            for cue, text in zip(cues, translated):
                cue.set_text(lang, text)

        stem = f"{song.track_no:02d}_{safe_name(song.title)}"
        source_profile = song.source_language
        write_srt(output_dir / "tracks" / f"{stem}_source.srt", cues, source_profile)
        for profile in profiles:
            profile_langs = required_languages([profile]) - {song.source_language}
            if profile_langs.intersection(translation_failures):
                continue
            write_srt(output_dir / "tracks" / f"{stem}_{profile}.srt", cues, profile)
        results.append(TrackResult(song=song, cues=cues, duration=duration,
                                   state=outcome.report.status,
                                   engine=outcome.result.engine, retry_count=outcome.retry_count,
                                   quality_score=outcome.report.score))

    if retry_failed_only:
        retried = {item["track_no"]: item for item in report}
        report = [retried.get(item.get("track_no"), item) for item in previous_report]
    diagnostics = _write_diagnostics(output_dir, report)
    if failures:
        _quarantine_stale_playlists(output_dir, diagnostics)
        (output_dir / "combined").mkdir(parents=True, exist_ok=True)
        (output_dir / "combined" / "SYNC_FAILED_DO_NOT_USE.txt").write_text(
            "This run failed sync validation. See ../_diagnostics/sync_report.txt.\n",
            encoding="utf-8",
        )
        translation_cached = sum(
            all(lang in translations_by_track[song.track_no] for lang in langs_needed)
            for song in songs)
        (diagnostics / "run_manifest.json").write_text(json.dumps({
            "app_version": APP_VERSION,
            "sync_pipeline": PIPELINE_VERSION,
            "run_started_at": run_started_at,
            "state": "WAITING_FOR_EXACT_RECOVERY",
            "translation_cached": translation_cached,
            "translation_total": len(songs),
            "failed_track_numbers": [item.song.track_no for item in failures],
            "tracks": report,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        progress(f"Sync PASS: {len(results)} | Sync FAIL: {len(failures)} | "
                 f"Translation cached: {translation_cached}/{len(songs)} | "
                 f"Combined SRT: Waiting for {len(failures)} tracks")
        failed_by_number = {item["track_no"]: item for item in report if item["state"] == "FAIL"}
        descriptions = []
        for failed in failures:
            item = failed_by_number.get(failed.song.track_no, {})
            primary = item.get("primary_failure") or {}
            reason = _popup_reason(primary, failed.warning or "정렬 복구 실패")
            descriptions.append(f"{failed.song.track_no:02d} {failed.song.title} — {reason}")
        raise RuntimeError(f"{len(all_songs)}곡 중 {len(all_songs) - len(failures)}곡 정상 / "
                           f"{len(failures)}곡 복구 필요\n" + "\n".join(descriptions))

    if retry_failed_only:
        for profile in profiles + ["source"]:
            shifted, retry_offset = [], 0.0
            for song in all_songs:
                stem = f"{song.track_no:02d}_{safe_name(song.title)}"
                path = output_dir / "tracks" / f"{stem}_{profile}.srt"
                if not path.exists():
                    shifted = []
                    break
                shifted.append((_read_srt(path), retry_offset))
                retry_offset += audio_duration(song.audio_path) + gap_seconds
            if shifted:
                write_combined(output_dir / "combined" / f"PLAYLIST_{profile}.srt", shifted, "en")

    offset = 0.0
    shifted_tracks, title_tracks, offset_report = [], [], []
    for result in results:
        if result.duration <= 0:
            raise RuntimeError(f"{result.song.track_no:02d} {result.song.title}: duration read failure")
        shifted_tracks.append((result.cues, offset))
        title_tracks.append((result.song, offset, result.duration))
        offset_report.append({"track_no": result.song.track_no, "title": result.song.title,
                              "starts_at": _hhmmss(offset),
                              "starts_at_seconds": round(offset, 3),
                              "duration_seconds": round(result.duration, 3),
                              "first_vocal": round(result.cues[0].start, 3),
                              "last_vocal": round(result.cues[-1].end, 3),
                              "trailing_instrumental": round(max(0.0, result.duration - result.cues[-1].end), 3)})
        offset += result.duration + gap_seconds
    (diagnostics / "playlist_offsets.json").write_text(
        json.dumps({"gap_seconds": gap_seconds, "total_seconds": round(offset, 3),
                    "tracks": offset_report}, ensure_ascii=False, indent=2), encoding="utf-8")
    for profile in ([] if retry_failed_only else profiles):
        # Only emit a translated profile when every track has that language.
        profile_langs = required_languages([profile])
        if all(all(lang == item.song.source_language or all(getattr(cue, lang, "") for cue in item.cues)
                   for lang in profile_langs) for item in results):
            write_combined(output_dir / "combined" / f"PLAYLIST_{profile}.srt", shifted_tracks, profile)
    # A translation failure must never hide an acoustically valid source SRT.
    source_languages = {item.song.source_language for item in results}
    if not retry_failed_only and len(source_languages) == 1:
        source_language = next(iter(source_languages))
        source_profile = source_language
        write_combined(output_dir / "combined" / "PLAYLIST_source.srt", shifted_tracks, source_profile)
    failure_marker = output_dir / "combined" / "SYNC_FAILED_DO_NOT_USE.txt"
    if failure_marker.exists():
        failure_marker.unlink()
    if title_seconds > 0 and not retry_failed_only:
        write_titles_srt(output_dir / "combined" / "PLAYLIST_titles.srt", title_tracks, title_seconds)
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), capture_output=True,
            text=True, check=False, timeout=3,
        ).stdout.strip() or "unknown"
    except Exception:
        git_commit = "unknown"
    outputs = {path.name: sha256(path) for path in (output_dir / "combined").glob("PLAYLIST_*.srt")}
    manifest = {
        "app_version": APP_VERSION, "git_commit": git_commit,
        "sync_pipeline": PIPELINE_VERSION,
        "run_started_at": run_started_at,
        "lyrics_file": None if lyrics_file is None else {"path": str(lyrics_file), "sha256": sha256(Path(lyrics_file))},
        "sync_mode": sync_mode, "model": model_name,
        "primary_engine": "WhisperX Exact Source", "fallback_engine": "WhisperX Exact Recovery",
        "demucs_used": any(item.get("aligned_audio") and Path(item["aligned_audio"]) != Path(song.audio_path)
                           for item, song in zip(report, songs)),
        "audio": [{"path": str(item.song.audio_path), "sha256": sha256(Path(item.song.audio_path))}
                  for item in results],
        "tracks": report, "output_srt_sha256": outputs,
    }
    (diagnostics / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "combined" / "PLAYLIST_SYNC_INFO.txt").write_text(
        f"LyricCap Studio {APP_VERSION}\n"
        f"git={git_commit}\nsync_pipeline={PIPELINE_VERSION}\n"
        f"Engine: WhisperX Exact Source / {'Demucs' if sync_mode == 'music_precise' else 'Original audio'}\n",
        encoding="utf-8",
    )
    warnings = sum(item["status"] == "PASS_WITH_WARNINGS" for item in report)
    progress(f"Playlist SRT created: {len(results) - warnings} PASS, {warnings} PASS WITH WARNINGS.")
    return results
