from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyriccap.alignment_recovery import TrackAlignmentState, TrackAlignmentTransaction
from lyriccap.aligners.base import AlignmentResult
from lyriccap.models import Cue
from lyriccap.models import Song
from lyriccap.batch import process_songs


class FakeAligner:
    def __init__(self, cues, engine):
        self.cues = cues
        self.engine = engine
        self.calls = 0

    def align(self, audio_path, lyric_lines, language, bounds=None):
        self.calls += 1
        return AlignmentResult([Cue(c.start, c.end, c.source) for c in self.cues], self.engine)


def test_only_failed_track_retries_and_fallback_must_pass():
    good = [Cue(0, 1, "a"), Cue(1.2, 2.2, "b")]
    bad = [Cue(0, 21.78, "a"), Cue(30.32, 31, "b")]
    primary_good = FakeAligner(good, "primary")
    retry_good = FakeAligner(good, "retry")
    fallback_good = FakeAligner(good, "fallback")
    first = TrackAlignmentTransaction(primary_good, retry_good, fallback_good).run(
        audio_path=Path("01.wav"), lyric_lines=["a", "b"], language="ja", track_duration=40)
    assert first.state == TrackAlignmentState.PASS
    assert retry_good.calls == fallback_good.calls == 0

    primary_bad = FakeAligner(bad, "primary")
    retry_bad = FakeAligner(bad, "retry")
    fallback = FakeAligner(good, "whisperx")
    second = TrackAlignmentTransaction(primary_bad, retry_bad, fallback).run(
        audio_path=Path("02.wav"), lyric_lines=["a", "b"], language="ja", track_duration=40)
    assert second.state == TrackAlignmentState.PASS
    assert second.result.engine == "whisperx"
    assert primary_bad.calls == retry_bad.calls == fallback.calls == 1
    assert [(c.start, c.end) for c in first.result.cues] == [(0, 1), (1.2, 2.2)]


def test_final_result_is_absent_when_fallback_also_fails():
    bad = [Cue(0, 42.72, "a"), Cue(43, 43.175, "b")]
    outcome = TrackAlignmentTransaction(
        FakeAligner(bad, "primary"), FakeAligner(bad, "retry"), FakeAligner(bad, "fallback")
    ).run(audio_path=Path("x.wav"), lyric_lines=["a", "b"], language="ja", track_duration=60)
    assert outcome.state == TrackAlignmentState.FAIL
    assert outcome.result is None


def test_fifteen_track_dry_run_aligns_each_track_once(tmp_path):
    import wave

    songs = []
    for number in range(1, 16):
        audio = tmp_path / f"{number:02d}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\0\0" * 24000)
        songs.append(Song(number, f"Track {number}", ["a", "b"], "ja", audio_path=audio))
    primary = FakeAligner([Cue(0, 1, "a"), Cue(1.2, 2.2, "b")], "primary")
    retry = FakeAligner([], "retry")
    fallback = FakeAligner([], "fallback")
    results = process_songs(songs, tmp_path / "output", ["ja"],
                            primary_aligner=primary, retry_aligner=retry,
                            fallback_aligner=fallback, title_seconds=0)
    assert len(results) == 15
    assert primary.calls == 15
    assert retry.calls == fallback.calls == 0
    assert (tmp_path / "output" / "combined" / "PLAYLIST_ja.srt").exists()
