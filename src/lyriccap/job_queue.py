from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .batch import JobCancelled

WAITING = "대기"
RUNNING = "실행중"
COMPLETED = "완료"
FAILED = "실패"
CANCELLED = "중단됨"
RETRYABLE_STATUSES = {WAITING, FAILED, CANCELLED}


@dataclass
class QueueJob:
    lyrics_path: Path
    output_folder: Path
    source_language: str
    model_name: str
    sync_mode: str
    fallback_enabled: bool
    profiles: list[str]
    translation_engine: str
    translation_model: str
    gap_seconds: float
    title_seconds: float
    refine_timestamps: bool
    audio_folder: Path | None = None
    audio_files: list[Path] = field(default_factory=list)
    status: str = WAITING
    error: str = ""

    def __post_init__(self) -> None:
        self.lyrics_path = Path(self.lyrics_path)
        self.output_folder = Path(self.output_folder)
        self.audio_folder = Path(self.audio_folder) if self.audio_folder else None
        self.audio_files = [Path(path) for path in self.audio_files]
        self.profiles = list(self.profiles)

    @property
    def audio_count(self) -> int:
        return len(self.audio_files)

    def identity(self) -> tuple[str, tuple[str, ...], str]:
        audio = (
            (str(self.audio_folder.resolve()),)
            if self.audio_folder
            else tuple(sorted(str(path.resolve()) for path in self.audio_files))
        )
        return (
            str(self.lyrics_path.resolve()),
            audio,
            str(self.output_folder.resolve()),
        )

    def report_dict(self, queue_no: int) -> dict:
        data = {
            "queueNo": queue_no,
            "lyrics": str(self.lyrics_path),
            "output": str(self.output_folder),
            "status": {
                COMPLETED: "completed",
                FAILED: "failed",
                CANCELLED: "cancelled",
                RUNNING: "running",
                WAITING: "waiting",
            }.get(self.status, self.status),
        }
        if self.error:
            data["error"] = self.error
        return data


class QueueLimitError(ValueError):
    pass


class DuplicateJobError(ValueError):
    pass


class QueueManager:
    """Maximum-five FIFO job list. Processing is deliberately sequential."""

    MAX_JOBS = 5

    def __init__(self, jobs: Iterable[QueueJob] = ()) -> None:
        self.jobs: list[QueueJob] = []
        self._run_lock = threading.Lock()
        for job in jobs:
            self.add(job)

    def add(self, job: QueueJob) -> None:
        if len(self.jobs) >= self.MAX_JOBS:
            raise QueueLimitError("대기열은 최대 5개까지 등록할 수 있습니다.")
        if any(existing.identity() == job.identity() for existing in self.jobs):
            raise DuplicateJobError("같은 작업이 이미 대기열에 있습니다.")
        self.jobs.append(job)

    def remove(self, index: int) -> QueueJob:
        if self.jobs[index].status == RUNNING:
            raise ValueError("실행 중인 작업은 삭제할 수 없습니다.")
        return self.jobs.pop(index)

    def move(self, index: int, delta: int) -> int:
        if self.jobs[index].status == RUNNING:
            raise ValueError("실행 중인 작업은 순서를 변경할 수 없습니다.")
        target = index + delta
        if target < 0 or target >= len(self.jobs):
            return index
        if self.jobs[target].status == RUNNING:
            raise ValueError("실행 중인 작업은 순서를 변경할 수 없습니다.")
        self.jobs[index], self.jobs[target] = self.jobs[target], self.jobs[index]
        return target

    def clear(self) -> None:
        if any(job.status == RUNNING for job in self.jobs):
            raise ValueError("실행 중에는 대기열을 비울 수 없습니다.")
        self.jobs.clear()

    def reset(self, index: int) -> None:
        if self.jobs[index].status == RUNNING:
            raise ValueError("실행 중인 작업은 초기화할 수 없습니다.")
        self.jobs[index].status = WAITING
        self.jobs[index].error = ""

    def run(
        self,
        processor: Callable[[QueueJob, Callable[[str], None]], object],
        *,
        should_cancel: Callable[[], bool] | None = None,
        progress: Callable[[int, int, QueueJob, str], None] | None = None,
        report_path: Path | None = None,
    ) -> dict:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("대기열은 이미 실행 중입니다.")
        try:
            return self._run_unlocked(
                processor,
                should_cancel=should_cancel,
                progress=progress,
                report_path=report_path,
            )
        finally:
            self._run_lock.release()

    def _run_unlocked(
        self,
        processor: Callable[[QueueJob, Callable[[str], None]], object],
        *,
        should_cancel: Callable[[], bool] | None = None,
        progress: Callable[[int, int, QueueJob, str], None] | None = None,
        report_path: Path | None = None,
    ) -> dict:
        """Run retryable jobs one by one in list order; never concurrently."""
        should_cancel = should_cancel or (lambda: False)
        progress = progress or (lambda _no, _total, _job, _message: None)
        runnable = [(i, job) for i, job in enumerate(self.jobs) if job.status in RETRYABLE_STATUSES]
        total = len(runnable)
        started_at = _timestamp()
        for position, (_index, job) in enumerate(runnable, 1):
            if should_cancel():
                break
            job.status = RUNNING
            job.error = ""
            progress(position, total, job, "작업 시작")
            try:
                processor(job, lambda message, p=position, j=job: progress(p, total, j, message))
            except JobCancelled as exc:
                job.status = CANCELLED
                job.error = str(exc)
                progress(position, total, job, "중단됨")
                break
            except Exception as exc:
                job.status = FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                self._write_job_error(job, traceback.format_exc())
                progress(position, total, job, f"실패: {exc}")
                continue
            job.status = COMPLETED
            progress(position, total, job, "완료")

        report = self.make_report(started_at=started_at)
        if report_path:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _write_job_error(job: QueueJob, details: str) -> None:
        try:
            diagnostics = job.output_folder / "diagnostics"
            diagnostics.mkdir(parents=True, exist_ok=True)
            (diagnostics / "queue_error.json").write_text(
                json.dumps(
                    {
                        "timestamp": _timestamp(), "status": "failed",
                        "error": job.error, "traceback": details,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def make_report(self, *, started_at: str | None = None) -> dict:
        return {
            "timestamp": _timestamp(),
            "startedAt": started_at,
            "totalJobs": len(self.jobs),
            "completed": sum(job.status == COMPLETED for job in self.jobs),
            "failed": sum(job.status == FAILED for job in self.jobs),
            "cancelled": sum(job.status == CANCELLED for job in self.jobs),
            "jobs": [job.report_dict(i) for i, job in enumerate(self.jobs, 1)],
        }


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
