from __future__ import annotations

import json
import re
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .aligner import StableTSAligner
from .aligners import StableTSAdapter, WhisperXExactAligner
from .alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from .audio import audio_duration
from .languages import language_name
from .models import TrackResult
from .srt import write_combined, write_srt, write_titles_srt
from .translator import build_translator

PROFILES = ["en", "en_ko", "en_ja", "ko", "ja"]
APP_VERSION = "v0.1.11"
PIPELINE_VERSION = "acoustic_truth_v1"


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


def required_languages(profiles: list[str]) -> set[str]:
    mapping = {"en": {"en"}, "ko": {"ko"}, "ja": {"ja"},
               "en_ko": {"en", "ko"}, "en_ja": {"en", "ja"}}
    return set().union(*(mapping.get(profile, set()) for profile in profiles))


def _diagnostic_entry(song, duration, outcome) -> dict:
    evidence = outcome.result.evidence if outcome.result else None
    return {
        "track_no": song.track_no, "title": song.title,
        "duration": round(duration, 3),
        "lyric_count_expected": len(song.lyrics),
        "lyric_count_aligned": outcome.report.stats.get("lyric_count_aligned", 0),
        "engine": outcome.result.engine if outcome.result else "none",
        "retry_count": outcome.retry_count, "state": outcome.state.value,
        "history": outcome.history, "quality_score": outcome.report.score,
        **outcome.report.stats,
        "issues": [issue.to_dict() for issue in outcome.report.issues],
        "status": outcome.report.status,
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
        lines.append(f"{item['track_no']:02d} {item['state']} score {item['quality_score']:.0f} "
                     f"engine={item['engine']} retry={item['retry_count']} title={item['title']}")
        lines.extend(f"  [{issue['severity']}] {issue['code']}: {issue['message']}"
                     for issue in item["issues"])
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
    legacy = StableTSAligner(model_name=model_name, sync_mode=sync_mode,
                             refine=refine_timestamps, cache_dir=output_dir / ".cache",
                             progress=progress, fallback_enabled=fallback_enabled)
    primary = primary_aligner or WhisperXExactAligner(
        model_name=model_name, progress=progress, cache_dir=output_dir / ".cache" / "alignment",
        audio_resolver=lambda path: legacy._prepare_audio(path)[0],
    )
    retry = retry_aligner or WhisperXExactAligner(
        model_name=model_name, progress=progress, cache_dir=output_dir / ".cache" / "alignment",
        audio_resolver=lambda path: legacy._prepare_audio(path)[0], strict=strict,
    )
    fallback = fallback_aligner or StableTSAdapter(legacy)

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
                                  instrumental_after=song.instrumental_after)
        report.append(_diagnostic_entry(song, duration, outcome))
        detail = ", ".join(issue.message for issue in outcome.report.issues[:3])
        progress(f"{song.track_no:02d} {outcome.state.value} score {outcome.report.score:.0f}" +
                 (f" - {detail}" if detail else ""))
        if outcome.state not in {TrackAlignmentState.PASS, TrackAlignmentState.PASS_WITH_WARNINGS} or outcome.result is None:
            failures.append(TrackResult(song=song, duration=duration, warning=detail,
                                        state="FAIL", retry_count=outcome.retry_count,
                                        quality_score=outcome.report.score))
            continue

        cues = outcome.result.cues
        for cue in cues:
            cue.set_text(song.source_language, cue.source)
        targets = [lang for lang in sorted(langs_needed) if lang != song.source_language]
        source_texts = [cue.source for cue in cues]
        translation_failures = []
        if targets:
            for lang in targets:
                try:
                    # Translation consumes the master cue text exactly once;
                    # it never invokes an aligner or quality gate.
                    translated = translator.translate_many(source_texts, song.source_language, lang,
                                                           title=song.title)
                    if len(translated) != len(cues):
                        raise RuntimeError(
                            f"translated line count {len(translated)} != source cue count {len(cues)}"
                        )
                    for cue, text in zip(cues, translated):
                        cue.set_text(lang, text)
                except Exception as exc:
                    translation_failures.append(lang)
                    progress(f"{song.track_no:02d} {lang.upper()} TRANSLATION FAIL - {exc}")

        stem = f"{song.track_no:02d}_{safe_name(song.title)}"
        source_profile = song.source_language if song.source_language in {"en", "ko", "ja"} else "en"
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

    diagnostics = _write_diagnostics(output_dir, report)
    if failures:
        _quarantine_stale_playlists(output_dir, diagnostics)
        (output_dir / "combined").mkdir(parents=True, exist_ok=True)
        (output_dir / "combined" / "SYNC_FAILED_DO_NOT_USE.txt").write_text(
            "This run failed sync validation. See ../_diagnostics/sync_report.txt.\n",
            encoding="utf-8",
        )
        descriptions = "; ".join(
            f"{failed.song.track_no}번 곡 {failed.song.title}: {failed.warning or 'alignment failed'}"
            for failed in failures
        )
        raise RuntimeError(f"Sync validation failed; normal playlist SRT was not created. "
                           f"{descriptions}. See {diagnostics / 'sync_report.txt'}")

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
    for profile in profiles:
        # Only emit a translated profile when every track has that language.
        profile_langs = required_languages([profile])
        if all(all(lang == item.song.source_language or all(getattr(cue, lang, "") for cue in item.cues)
                   for lang in profile_langs) for item in results):
            write_combined(output_dir / "combined" / f"PLAYLIST_{profile}.srt", shifted_tracks, profile)
    # A translation failure must never hide an acoustically valid source SRT.
    source_languages = {item.song.source_language for item in results}
    if len(source_languages) == 1:
        source_language = next(iter(source_languages))
        source_profile = source_language if source_language in {"en", "ko", "ja"} else "en"
        write_combined(output_dir / "combined" / "PLAYLIST_source.srt", shifted_tracks, source_profile)
    failure_marker = output_dir / "combined" / "SYNC_FAILED_DO_NOT_USE.txt"
    if failure_marker.exists():
        failure_marker.unlink()
    if title_seconds > 0:
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
        "primary_engine": "WhisperX Exact Source", "fallback_engine": "stable-ts exact-direct",
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
