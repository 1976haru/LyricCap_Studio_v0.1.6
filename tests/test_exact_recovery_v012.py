from pathlib import Path
import wave
import pytest

from lyriccap.alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from lyriccap.aligners.base import AlignmentResult
from lyriccap.batch import process_songs
from lyriccap.models import Cue, Song


class ScriptedAligner:
    def __init__(self, engine, outcome):
        self.engine, self.outcome, self.calls = engine, outcome, 0

    def align(self, audio_path, lyric_lines, language, bounds=None):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return AlignmentResult([Cue(c.start, c.end, c.source, language)
                                for c in self.outcome], self.engine)


class SequenceAligner(ScriptedAligner):
    def align(self, audio_path, lyric_lines, language, bounds=None):
        self.calls += 1
        outcome = self.outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return AlignmentResult([Cue(c.start, c.end, c.source, language)
                                for c in outcome], self.engine)


GOOD = [Cue(0.2, 1.0, "a"), Cue(1.2, 2.0, "b")]


def run_chain(whole, section, recovery):
    return TrackAlignmentTransaction(whole, section, recovery).run(
        audio_path=Path("song.wav"), lyric_lines=["a", "b"], language="en",
        track_duration=3.0)


def test_whole_failure_section_success_and_stable_ts_is_not_called():
    whole = ScriptedAligner("whisperx-exact-whole", RuntimeError("direct lines 18-20 missing"))
    section = ScriptedAligner("whisperx-exact-section", GOOD)
    stable = ScriptedAligner("stable-ts", TypeError("tuple indices must be integers or slices, not tuple"))
    outcome = run_chain(whole, section, stable)
    assert outcome.state == TrackAlignmentState.PASS
    assert [a.status for a in outcome.attempts] == ["FAIL", "PASS"]
    assert stable.calls == 0


def test_failed_section_recovery_succeeds():
    outcome = run_chain(
        ScriptedAligner("whole", RuntimeError("whole failed")),
        ScriptedAligner("section", RuntimeError("section failed")),
        ScriptedAligner("recovery", GOOD))
    assert outcome.state == TrackAlignmentState.PASS
    assert [a.status for a in outcome.attempts] == ["FAIL", "FAIL", "PASS"]


def test_all_exact_errors_are_preserved():
    outcome = run_chain(*[ScriptedAligner(name, RuntimeError(f"{name} reason"))
                          for name in ("whole", "section", "recovery")])
    assert outcome.state == TrackAlignmentState.FAIL
    assert [a.error_message for a in outcome.attempts] == [
        "whole reason", "section reason", "recovery reason"]


class FifteenTrackWhole(ScriptedAligner):
    def align(self, audio_path, lyric_lines, language, bounds=None):
        self.calls += 1
        if Path(audio_path).stem in {"02", "05", "12"}:
            raise RuntimeError("whole exact failed")
        return AlignmentResult([Cue(c.start, c.end, c.source, language) for c in GOOD], self.engine)


def test_mocked_fifteen_tracks_three_section_recoveries(tmp_path):
    songs = []
    for number in range(1, 16):
        audio = tmp_path / f"{number:02d}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000)
            handle.writeframes(b"\0\0" * 24000)
        songs.append(Song(number, f"Track {number}", ["a", "b"], "en", audio_path=audio))
    whole = FifteenTrackWhole("whisperx-exact-whole", GOOD)
    section = ScriptedAligner("whisperx-exact-section", GOOD)
    recovery = ScriptedAligner("whisperx-exact-recovery", GOOD)
    results = process_songs(songs, tmp_path / "output", ["en"], strict=False,
                            primary_aligner=whole, retry_aligner=section,
                            fallback_aligner=recovery, title_seconds=0)
    assert len(results) == 15
    assert whole.calls == 15
    assert section.calls == 3
    assert recovery.calls == 0
    assert (tmp_path / "output/combined/PLAYLIST_en.srt").exists()


def test_translation_cached_while_sync_waits_and_reused_on_retry(tmp_path, monkeypatch):
    audio = tmp_path / "01.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000)
        handle.writeframes(b"\0\0" * 24000)
    song = Song(1, "Waiting", ["a", "b"], "en", audio_path=audio)

    class CachedTranslator:
        calls = 0
        value = None
        def translate_many(self, texts, source, target, title=""):
            if self.value is None:
                type(self).calls += 1
                self.value = ["あ", "い"]
            return list(self.value)

    translator = CachedTranslator()
    monkeypatch.setattr("lyriccap.batch.build_translator", lambda *a, **k: translator)
    whole = ScriptedAligner("whole", RuntimeError("whole failed"))
    section = SequenceAligner("section", [RuntimeError("section failed"), GOOD])
    recovery = ScriptedAligner("recovery", RuntimeError("recovery failed"))
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        process_songs([song], output, ["ja"], strict=False, primary_aligner=whole,
                      retry_aligner=section, fallback_aligner=recovery, title_seconds=0)
    assert (output / "translations_pending_sync/01_Waiting_ja.txt").exists()
    assert not (output / "tracks/01_Waiting_ja.srt").exists()

    process_songs([song], output, ["ja"], strict=False, primary_aligner=whole,
                  retry_aligner=section, fallback_aligner=recovery, title_seconds=0,
                  retry_failed_only=True)
    assert CachedTranslator.calls == 1
    source_times = [line for line in (output / "tracks/01_Waiting_source.srt").read_text(
        encoding="utf-8").splitlines() if " --> " in line]
    target_times = [line for line in (output / "tracks/01_Waiting_ja.srt").read_text(
        encoding="utf-8").splitlines() if " --> " in line]
    assert source_times == target_times
