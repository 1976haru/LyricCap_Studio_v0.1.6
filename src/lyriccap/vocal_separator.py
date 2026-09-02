from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


class VocalSeparationError(RuntimeError):
    pass


class DemucsSeparator:
    """Separate vocals by calling Demucs as a subprocess.

    stable-ts can call Demucs internally, but keeping source separation in a
    separate process avoids API/type compatibility problems between stable-ts,
    Demucs, torchaudio and PyTorch. The resulting vocals.wav is then aligned by
    stable-ts as ordinary audio.
    """

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "htdemucs",
        progress: Callable[[str], None] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.progress = progress or (lambda _msg: None)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fingerprint(audio_path: Path) -> str:
        st = audio_path.stat()
        raw = f"{audio_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "surrogatepass")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _track_cache(self, audio_path: Path) -> Path:
        return self.cache_dir / self._fingerprint(audio_path)

    def separate(self, audio_path: Path) -> Path:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise VocalSeparationError(f"음원 파일을 찾을 수 없습니다: {audio_path}")

        cache = self._track_cache(audio_path)
        vocals = cache / "vocals.wav"
        meta = cache / "source.json"
        if vocals.exists() and vocals.stat().st_size > 44:
            self.progress(f"보컬 캐시 사용: {audio_path.name}")
            return vocals

        cache.mkdir(parents=True, exist_ok=True)
        demucs_out = cache / "demucs_out"
        demucs_out.mkdir(parents=True, exist_ok=True)

        # Demucs CLI writes <out>/<model>/<track-stem>/vocals.wav.
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            self.model_name,
            "-o",
            str(demucs_out),
            str(audio_path),
        ]
        self.progress(f"보컬 분리(Demucs): {audio_path.name}")

        env = os.environ.copy()
        # PyTorch 2.6+ changed torch.load's default. Demucs' official model
        # packages predate that change. This compatibility variable only
        # applies when the caller did not explicitly set weights_only.
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        # Demucs prints the track name as it works. On Windows the child's
        # stdout defaults to cp949, so a non-ASCII file name (La derniere
        # nacelle) kills it with UnicodeEncodeError before separation ends.
        # Our own encoding="utf-8" below only decodes what we receive, so the
        # child has to be told to encode as UTF-8 in the first place.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except Exception as e:
            raise VocalSeparationError(f"Demucs 실행 자체에 실패했습니다: {e}") from e

        log_path = cache / "demucs.log"
        log_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-25:])
            raise VocalSeparationError(
                "보컬 분리(Demucs)에 실패했습니다.\n"
                f"로그: {log_path}\n"
                f"마지막 출력:\n{tail}"
            )

        candidates = list(demucs_out.glob(f"{self.model_name}/**/vocals.wav"))
        if not candidates:
            candidates = list(demucs_out.glob("**/vocals.wav"))
        if not candidates:
            raise VocalSeparationError(
                "Demucs는 종료됐지만 vocals.wav를 찾지 못했습니다. "
                f"로그를 확인하세요: {log_path}"
            )

        produced = max(candidates, key=lambda p: p.stat().st_size)
        if produced.stat().st_size <= 44:
            raise VocalSeparationError("분리된 보컬 파일이 비어 있습니다.")

        # Move to a stable cache path so later Demucs folder-name changes do
        # not affect LyricCap.
        produced.replace(vocals)
        meta.write_text(
            json.dumps(
                {
                    "source": str(audio_path),
                    "model": self.model_name,
                    "demucsLog": str(log_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return vocals
