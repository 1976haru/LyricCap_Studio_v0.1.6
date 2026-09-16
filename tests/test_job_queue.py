from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyriccap.batch import JobCancelled
from lyriccap.job_queue import (
    CANCELLED, COMPLETED, FAILED, WAITING,
    DuplicateJobError, QueueJob, QueueLimitError, QueueManager,
)


def make_job(tmp_path: Path, number: int, **changes) -> QueueJob:
    lyrics = tmp_path / f"EP{number:03d}" / "lyrics.json"
    audio = tmp_path / f"audio{number}" / "song.wav"
    lyrics.parent.mkdir(parents=True, exist_ok=True)
    audio.parent.mkdir(parents=True, exist_ok=True)
    lyrics.write_text("{}", encoding="utf-8")
    audio.write_bytes(b"RIFF")
    values = dict(
        lyrics_path=lyrics,
        audio_folder=None,
        audio_files=[audio],
        output_folder=lyrics.parent,
        source_language="en",
        model_name="base",
        sync_mode="music_precise",
        fallback_enabled=True,
        profiles=["en"],
        translation_engine="gemini",
        translation_model="gemini-2.5-flash",
        gap_seconds=0.0,
        title_seconds=5.0,
        refine_timestamps=False,
    )
    values.update(changes)
    return QueueJob(**values)


def test_queue_accepts_five_and_rejects_sixth(tmp_path):
    queue = QueueManager()
    for number in range(1, 6):
        queue.add(make_job(tmp_path, number))
    assert len(queue.jobs) == 5
    with pytest.raises(QueueLimitError, match="최대 5개"):
        queue.add(make_job(tmp_path, 6))


def test_queue_runs_fifo_and_continues_after_failure(tmp_path):
    queue = QueueManager(make_job(tmp_path, n) for n in range(1, 4))
    calls = []

    def processor(job, _progress):
        calls.append(job.lyrics_path.parent.name)
        if "002" in job.lyrics_path.parent.name:
            raise RuntimeError("mock failure")

    report = queue.run(processor)
    assert calls == ["EP001", "EP002", "EP003"]
    assert [job.status for job in queue.jobs] == [COMPLETED, FAILED, COMPLETED]
    assert report["completed"] == 2 and report["failed"] == 1


def test_completed_job_is_skipped_on_rerun(tmp_path):
    queue = QueueManager([make_job(tmp_path, 1), make_job(tmp_path, 2)])
    queue.jobs[0].status = COMPLETED
    calls = []
    queue.run(lambda job, _progress: calls.append(job.lyrics_path.parent.name))
    assert calls == ["EP002"]


def test_cancel_stops_before_next_job(tmp_path):
    queue = QueueManager([make_job(tmp_path, 1), make_job(tmp_path, 2)])
    calls = []

    def processor(job, _progress):
        calls.append(job.lyrics_path.parent.name)
        raise JobCancelled("stop")

    queue.run(processor)
    assert calls == ["EP001"]
    assert queue.jobs[0].status == CANCELLED
    assert queue.jobs[1].status == WAITING


def test_move_up_and_down(tmp_path):
    queue = QueueManager(make_job(tmp_path, n) for n in range(1, 4))
    assert queue.move(2, -1) == 1
    assert [j.lyrics_path.parent.name for j in queue.jobs] == ["EP001", "EP003", "EP002"]
    assert queue.move(0, 1) == 1
    assert [j.lyrics_path.parent.name for j in queue.jobs] == ["EP003", "EP001", "EP002"]


def test_duplicate_job_is_blocked(tmp_path):
    job = make_job(tmp_path, 1)
    queue = QueueManager([job])
    with pytest.raises(DuplicateJobError, match="이미"):
        queue.add(make_job(tmp_path, 1))


def test_folder_and_explicit_file_audio_inputs_are_distinct(tmp_path):
    explicit = make_job(tmp_path, 1)
    folder = explicit.audio_files[0].parent
    folder_job = make_job(tmp_path, 1, audio_folder=folder, audio_files=[])
    queue = QueueManager([explicit])
    queue.add(folder_job)
    assert queue.jobs[0].audio_files and queue.jobs[0].audio_folder is None
    assert queue.jobs[1].audio_folder == folder and not queue.jobs[1].audio_files


def test_each_job_has_independent_output_folder(tmp_path):
    first = make_job(tmp_path, 1)
    second = make_job(tmp_path, 2)
    assert first.output_folder != second.output_folder
    assert first.output_folder / ".cache" != second.output_folder / ".cache"


def test_queue_preserves_fallback_setting_for_processor(tmp_path):
    queue = QueueManager([make_job(tmp_path, 1, fallback_enabled=True)])
    seen = []
    queue.run(lambda job, _progress: seen.append(job.fallback_enabled))
    assert seen == [True]


def test_failed_job_writes_aggregate_and_job_diagnostics(tmp_path):
    job = make_job(tmp_path, 1)
    report_path = tmp_path / "queue_report.json"

    def fail(_job, _progress):
        raise RuntimeError("expected failure")

    report = QueueManager([job]).run(fail, report_path=report_path)
    diagnostic = json.loads((job.output_folder / "diagnostics" / "queue_error.json").read_text(encoding="utf-8"))
    assert report["failed"] == 1
    assert "expected failure" in report["jobs"][0]["error"]
    assert "expected failure" in diagnostic["traceback"]


def test_api_key_is_not_in_job_or_report(tmp_path):
    job = make_job(tmp_path, 1)
    assert "api_key" not in job.__dict__
    report_path = tmp_path / "queue_report.json"
    QueueManager([job]).run(lambda _job, _progress: None, report_path=report_path)
    raw = report_path.read_text(encoding="utf-8")
    assert "secret" not in raw
    report = json.loads(raw)
    assert all("api" not in key.lower() for item in report["jobs"] for key in item)
    assert report["jobs"][0]["status"] == "completed"


def test_auto_output_folder_is_preserved_in_queue_job(tmp_path):
    sys.path.insert(0, str(ROOT))
    from app import default_output_folder_for_lyrics

    lyrics = tmp_path / "EP002" / "lyrics.json"
    lyrics.parent.mkdir()
    lyrics.write_text("{}", encoding="utf-8")
    job = make_job(tmp_path, 2, lyrics_path=lyrics, output_folder=default_output_folder_for_lyrics(lyrics))
    assert job.output_folder == lyrics.parent
