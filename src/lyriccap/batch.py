from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .aligner import StableTSAligner
from .aligners import StableTSAdapter, WhisperXAligner
from .alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from .audio import audio_duration
from .languages import language_name
from .models import TrackResult
from .srt import write_combined, write_srt, write_titles_srt
from .translator import build_translator

PROFILES = ["en", "en_ko", "en_ja", "ko", "ja"]


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
):
    """Align each audio independently and commit output only after validation."""
    progress = progress or (lambda _message: None)
    should_cancel = should_cancel or (lambda: False)

    def check_cancel():
        if should_cancel():
            raise JobCancelled("The user cancelled the job; cached analysis remains reusable.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = StableTSAligner(model_name=model_name, sync_mode=sync_mode,
                             refine=refine_timestamps, cache_dir=output_dir / ".cache",
                             progress=progress, fallback_enabled=fallback_enabled)
    primary = primary_aligner or StableTSAdapter(legacy)
    retry = retry_aligner or StableTSAdapter(legacy, retry=True)
    fallback = fallback_aligner or WhisperXAligner(
        model_name=model_name, progress=progress,
        audio_resolver=lambda path: legacy._prepare_audio(path)[0],
    )

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
        )
        progress(f"[{index}/{len(songs)}] per-track alignment: {song.title}")
        outcome = transaction.run(audio_path=Path(song.audio_path), lyric_lines=song.lyrics,
                                  language=song.source_language, track_duration=duration,
                                  instrumental_after=song.instrumental_after)
        report.append(_diagnostic_entry(song, duration, outcome))
        detail = ", ".join(issue.message for issue in outcome.report.issues[:3])
        progress(f"{song.track_no:02d} {outcome.state.value} score {outcome.report.score:.0f}" +
                 (f" - {detail}" if detail else ""))
        if outcome.state != TrackAlignmentState.PASS or outcome.result is None:
            failures.append(TrackResult(song=song, duration=duration, warning=detail,
                                        state="FAIL", retry_count=outcome.retry_count,
                                        quality_score=outcome.report.score))
            continue

        cues = outcome.result.cues
        for cue in cues:
            cue.set_text(song.source_language, cue.source)
        targets = [lang for lang in sorted(langs_needed) if lang != song.source_language]
        source_texts = [cue.source for cue in cues]
        if targets and hasattr(translator, "translate_multi"):
            bundle = translator.translate_multi(source_texts, song.source_language, targets,
                                                title=song.title)
            for lang in targets:
                for cue, text in zip(cues, bundle.get(lang, [])):
                    cue.set_text(lang, text)
        elif targets:
            for lang in targets:
                translated = translator.translate_many(source_texts, song.source_language, lang,
                                                       title=song.title)
                for cue, text in zip(cues, translated):
                    cue.set_text(lang, text)

        stem = f"{song.track_no:02d}_{safe_name(song.title)}"
        for profile in profiles:
            write_srt(output_dir / "tracks" / f"{stem}_{profile}.srt", cues, profile)
        results.append(TrackResult(song=song, cues=cues, duration=duration, state="PASS",
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
        shifted_tracks.append((result.cues, offset))
        title_tracks.append((result.song, offset, result.duration))
        offset_report.append({"track_no": result.song.track_no, "title": result.song.title,
                              "starts_at": _hhmmss(offset),
                              "starts_at_seconds": round(offset, 3),
                              "duration_seconds": round(result.duration, 3)})
        offset += result.duration + gap_seconds
    (diagnostics / "playlist_offsets.json").write_text(
        json.dumps({"gap_seconds": gap_seconds, "total_seconds": round(offset, 3),
                    "tracks": offset_report}, ensure_ascii=False, indent=2), encoding="utf-8")
    for profile in profiles:
        write_combined(output_dir / "combined" / f"PLAYLIST_{profile}.srt", shifted_tracks, profile)
    failure_marker = output_dir / "combined" / "SYNC_FAILED_DO_NOT_USE.txt"
    if failure_marker.exists():
        failure_marker.unlink()
    if title_seconds > 0:
        write_titles_srt(output_dir / "combined" / "PLAYLIST_titles.srt", title_tracks, title_seconds)
    progress("All tracks passed sync validation; playlist SRT created.")
    return results
