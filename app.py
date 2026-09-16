from __future__ import annotations
import json
import os
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lyriccap.parsers import parse_lyrics_file, SKIPPED_TRACKS
from lyriccap.utils import discover_audio
from lyriccap.matcher import match_audio
from lyriccap.batch import process_songs, PROFILES, required_languages, JobCancelled
from lyriccap.languages import SOURCE_LANGUAGES, language_name
from lyriccap.job_queue import (
    QueueJob, QueueManager, QueueLimitError, DuplicateJobError,
    WAITING, RUNNING, COMPLETED, FAILED, CANCELLED,
)

SYNC_MODE_LABELS = {
    "정밀 음악 싱크 (보컬 분리 + VAD, 권장)": "music_precise",
    "빠른 음악 싱크 (VAD, 보컬분리 없음)": "music_fast",
    "음성/내레이션 싱크": "voice",
}

PROFILE_LABELS = {
    "en": "영어",
    "en_ko": "영어 + 한국어",
    "en_ja": "영어 + 일본어",
    "ko": "한국어",
    "ja": "일본어",
}
MODEL_CHOICES = ["tiny", "base", "small", "medium", "tiny.en", "base.en", "small.en", "medium.en"]
# "가사 원문" 드롭다운. 코드만 보여주면 fr/es가 뭔지 알기 어려워 이름을 붙입니다.
SOURCE_LANG_CHOICES = ["auto"] + [f"{c} ({language_name(c)})" for c in SOURCE_LANGUAGES]
TRANSLATION_ENGINE_LABELS = {
    "AI 자연번역 (Gemini, 권장)": "gemini",
    "오프라인 빠른번역 (Argos, 직역 가능)": "argos",
}
SETTINGS_PATH = ROOT / "translation_settings.json"


def default_output_folder_for_lyrics(lyrics_path: Path) -> Path:
    return Path(lyrics_path).parent


def load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LyricCap Studio v0.1.9 - Five-job Queue + Safe Whisper Fallback")
        self.geometry("1100x960")
        self.minsize(940, 700)
        self.songs = []
        self.lyrics_path = tk.StringVar()
        self.audio_folder = tk.StringVar()
        self.selected_audio_files = []
        self.output_folder = tk.StringVar(value=str(ROOT / "output"))
        self.source_lang = tk.StringVar(value="auto")
        self.model_name = tk.StringVar(value="base")
        self.gap_seconds = tk.DoubleVar(value=0.0)
        self.title_seconds = tk.DoubleVar(value=5.0)
        self.sync_mode_label = tk.StringVar(value="정밀 음악 싱크 (보컬 분리 + VAD, 권장)")
        self.refine_timestamps = tk.BooleanVar(value=False)
        self.fallback_enabled = tk.BooleanVar(value=True)
        self.profile_vars = {p: tk.BooleanVar(value=True) for p in PROFILES}
        self.queue = QueueManager()
        self._queue_running = False
        self._single_running = False

        settings = load_settings()
        engine_code = str(settings.get("translation_engine", "gemini"))
        engine_label = next(
            (label for label, code in TRANSLATION_ENGINE_LABELS.items() if code == engine_code),
            "AI 자연번역 (Gemini, 권장)",
        )
        self.translation_engine_label = tk.StringVar(value=engine_label)
        self.translation_model = tk.StringVar(value=str(settings.get("gemini_model", "gemini-2.5-flash")))
        self.gemini_api_key = tk.StringVar(value=str(settings.get("gemini_api_key", "")))
        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=10)

        ttk.Label(top, text="1. 가사 JSON/TXT").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.lyrics_path).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(top, text="파일 선택", command=self.choose_lyrics).grid(row=0, column=2, **pad)

        ttk.Label(top, text="2. WAV/MP3 폴더").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.audio_folder).grid(row=1, column=1, sticky="ew", **pad)
        audio_buttons = ttk.Frame(top)
        audio_buttons.grid(row=1, column=2, **pad)
        ttk.Button(audio_buttons, text="파일 선택", command=self.choose_audio_files).pack(side="left", padx=(0,4))
        ttk.Button(audio_buttons, text="폴더 선택", command=self.choose_audio).pack(side="left")

        ttk.Label(top, text="3. 출력 폴더").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.output_folder).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(top, text="폴더 선택", command=self.choose_output).grid(row=2, column=2, **pad)
        top.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(self, text="자막/싱크 옵션")
        opts.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(opts, text="가사 원문").grid(row=0, column=0, padx=8, pady=8)
        ttk.Combobox(opts, textvariable=self.source_lang, state="readonly", width=16,
                     values=SOURCE_LANG_CHOICES).grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(opts, text="정렬 모델").grid(row=0, column=2, padx=8, pady=8)
        ttk.Combobox(opts, textvariable=self.model_name, state="readonly", width=12,
                     values=MODEL_CHOICES).grid(row=0, column=3, padx=8, pady=8)
        ttk.Label(opts, text="싱크 방식").grid(row=0, column=4, padx=8, pady=8)
        ttk.Combobox(
            opts,
            textvariable=self.sync_mode_label,
            state="readonly",
            width=34,
            values=list(SYNC_MODE_LABELS.keys()),
        ).grid(row=0, column=5, padx=8, pady=8)
        ttk.Button(opts, text="가사/음원 매칭 확인", command=self.load_and_match).grid(row=0, column=6, padx=12, pady=8)

        ttk.Label(opts, text="곡 사이 간격(초)").grid(row=1, column=0, padx=8, pady=(0,8))
        ttk.Spinbox(opts, from_=0, to=30, increment=0.1, textvariable=self.gap_seconds, width=8).grid(row=1, column=1, padx=8, pady=(0,8))
        ttk.Checkbutton(
            opts,
            text="타임스탬프 정밀 보정(느림)",
            variable=self.refine_timestamps,
        ).grid(row=1, column=2, columnspan=2, padx=8, pady=(0,8), sticky="w")
        ttk.Label(
            opts,
            text="※ v0.1.7: stable-ts 강제정렬을 사용하지 않습니다. Demucs 보컬 분리 → OpenAI Whisper 단어 타임스탬프 → JSON/TXT 원문 가사와 순차 매칭하여 실제 보컬 시간을 잡습니다.",
        ).grid(row=1, column=4, columnspan=3, padx=8, pady=(0,8), sticky="w")

        ttk.Label(opts, text="곡 제목 자막 표시(초)").grid(row=2, column=0, padx=8, pady=(0,8))
        ttk.Spinbox(opts, from_=0, to=30, increment=0.5, textvariable=self.title_seconds, width=8).grid(row=2, column=1, padx=8, pady=(0,8))
        ttk.Label(
            opts,
            text="0이면 곡 제목 SRT를 만들지 않습니다. combined/PLAYLIST_en.srt=가사(아래쪽 트랙), "
                 "combined/PLAYLIST_titles.srt=곡 제목(위쪽 트랙)으로 CapCut에서 두 트랙으로 올리세요.",
        ).grid(row=2, column=2, columnspan=5, padx=8, pady=(0,8), sticky="w")

        trans = ttk.LabelFrame(self, text="번역 - JSON/TXT에 번역문이 없어도 자동 생성")
        trans.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(trans, text="번역 방식").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Combobox(
            trans,
            textvariable=self.translation_engine_label,
            state="readonly",
            width=31,
            values=list(TRANSLATION_ENGINE_LABELS.keys()),
        ).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(trans, text="Gemini API 키").grid(row=0, column=2, padx=8, pady=8, sticky="e")
        self.api_entry = ttk.Entry(trans, textvariable=self.gemini_api_key, show="*", width=30)
        self.api_entry.grid(row=0, column=3, padx=8, pady=8, sticky="ew")
        ttk.Button(trans, text="키 저장", command=self.save_translation_settings).grid(row=0, column=4, padx=4, pady=8)

        ttk.Label(trans, text="AI 모델").grid(row=1, column=0, padx=8, pady=(0,8), sticky="e")
        ttk.Entry(trans, textvariable=self.translation_model, width=31).grid(row=1, column=1, padx=8, pady=(0,8), sticky="w")
        ttk.Label(
            trans,
            text="AI 자연번역은 곡 전체 문맥을 보고 번역하여 직역을 줄이고 후렴/호칭/감정을 일관되게 유지합니다.",
        ).grid(row=1, column=2, columnspan=3, padx=8, pady=(0,8), sticky="w")
        trans.columnconfigure(3, weight=1)

        prof = ttk.LabelFrame(self, text="생성할 자막")
        prof.pack(fill="x", padx=12, pady=(0, 8))
        for i, p in enumerate(PROFILES):
            ttk.Checkbutton(prof, text=PROFILE_LABELS[p], variable=self.profile_vars[p]).grid(row=0, column=i, padx=12, pady=8, sticky="w")
        ttk.Checkbutton(
            prof, text="Demucs 실패 시 원본 음원 Whisper fallback",
            variable=self.fallback_enabled,
        ).grid(row=0, column=len(PROFILES), padx=12, pady=8, sticky="w")

        table_frame = ttk.LabelFrame(self, text="곡 매칭")
        table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        cols = ("no", "title", "audio", "lines")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=7)
        self.tree.heading("no", text="#")
        self.tree.heading("title", text="곡 제목")
        self.tree.heading("audio", text="연결된 음원")
        self.tree.heading("lines", text="가사 줄")
        self.tree.column("no", width=50, anchor="center")
        self.tree.column("title", width=260)
        self.tree.column("audio", width=560)
        self.tree.column("lines", width=80, anchor="center")
        y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y.set)
        self.tree.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")

        queue_frame = ttk.LabelFrame(self, text="작업 대기열 (최대 5개, FIFO 순차 실행)")
        queue_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        qcols = ("no", "lyrics", "audio", "output", "status")
        self.queue_tree = ttk.Treeview(queue_frame, columns=qcols, show="headings", height=5)
        for col, title, width in (
            ("no", "#", 38), ("lyrics", "가사 파일", 210), ("audio", "음원", 180),
            ("output", "출력 폴더", 420), ("status", "상태", 75),
        ):
            self.queue_tree.heading(col, text=title)
            self.queue_tree.column(col, width=width, anchor="center" if col in {"no", "status"} else "w")
        self.queue_tree.pack(fill="both", expand=True, padx=6, pady=6)
        qbuttons = ttk.Frame(queue_frame)
        qbuttons.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(qbuttons, text="현재 작업 대기열에 추가", command=self.add_current_to_queue).pack(side="left")
        ttk.Button(qbuttons, text="선택 삭제", command=self.remove_queue_job).pack(side="left", padx=3)
        ttk.Button(qbuttons, text="↑ 위로", command=lambda: self.move_queue_job(-1)).pack(side="left", padx=3)
        ttk.Button(qbuttons, text="↓ 아래로", command=lambda: self.move_queue_job(1)).pack(side="left", padx=3)
        ttk.Button(qbuttons, text="선택 작업 다시 실행", command=self.reset_queue_job).pack(side="left", padx=3)
        ttk.Button(qbuttons, text="대기열 비우기", command=self.clear_queue).pack(side="right")
        self.queue_run_btn = ttk.Button(qbuttons, text="대기열 전체 실행", command=self.start_queue)
        self.queue_run_btn.pack(side="right", padx=3)
        self.queue_summary = tk.StringVar(value="전체 진행: - / -    현재 작업: -")
        ttk.Label(queue_frame, textvariable=self.queue_summary).pack(anchor="w", padx=7)
        self.queue_progress = ttk.Progressbar(queue_frame, mode="determinate")
        self.queue_progress.pack(fill="x", padx=7, pady=(2, 6))

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=12, pady=(0, 12))
        self.progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 6))
        self.status = tk.StringVar(value="JSON/TXT와 음원을 선택하세요.")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")
        self.run_btn = ttk.Button(bottom, text="▶ 자막 전체 생성", command=self.start_process)
        self.run_btn.pack(side="right")
        self.stop_btn = ttk.Button(bottom, text="■ 중단", command=self.stop_process, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 6))
        # 작업 스레드에 중단 의사를 전달하는 신호입니다.
        self.cancel_event = threading.Event()

    def translation_engine(self) -> str:
        return TRANSLATION_ENGINE_LABELS.get(self.translation_engine_label.get(), "gemini")

    def sync_mode(self) -> str:
        return SYNC_MODE_LABELS.get(self.sync_mode_label.get(), "music_precise")

    def save_translation_settings(self, silent: bool = False):
        data = {
            "translation_engine": self.translation_engine(),
            "gemini_api_key": self.gemini_api_key.get().strip(),
            "gemini_model": self.translation_model.get().strip() or "gemini-2.5-flash",
        }
        try:
            save_settings(data)
            if not silent:
                messagebox.showinfo(
                    "저장 완료",
                    "번역 설정을 이 프로그램 폴더에 저장했습니다.\n다음 실행부터 자동으로 불러옵니다.",
                )
        except Exception as e:
            if not silent:
                messagebox.showerror("저장 오류", str(e))

    def choose_lyrics(self):
        p = filedialog.askopenfilename(filetypes=[("Lyrics", "*.json *.txt"), ("All files", "*.*")])
        if p:
            self.lyrics_path.set(p)
            # 가사 파일이 있는 폴더를 출력 폴더의 기본값으로 자동 지정합니다.
            self.output_folder.set(str(default_output_folder_for_lyrics(Path(p))))
            self.status.set(f"출력 폴더 자동 설정: {Path(p).parent}")
            self.load_and_match()

    def choose_audio_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"), ("All files", "*.*")]
        )
        if paths:
            self.selected_audio_files = [Path(x) for x in paths]
            self.audio_folder.set(f"선택한 음원 {len(paths)}개")
            self.load_and_match()

    def choose_audio(self):
        p = filedialog.askdirectory()
        if p:
            self.selected_audio_files = []
            self.audio_folder.set(p)
            self.load_and_match()

    def choose_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_folder.set(p)

    def load_and_match(self):
        lyrics = Path(self.lyrics_path.get()) if self.lyrics_path.get() else None
        folder_text = self.audio_folder.get()
        folder = Path(folder_text) if folder_text and not self.selected_audio_files else None
        if not lyrics or not lyrics.exists():
            return
        try:
            songs = parse_lyrics_file(lyrics, self._source_lang_code())
            audios = list(self.selected_audio_files) if self.selected_audio_files else (discover_audio(folder) if folder else [])
            self.songs = match_audio(songs, audios)
            self.tree.delete(*self.tree.get_children())
            for s in self.songs:
                self.tree.insert("", "end", values=(s.track_no, s.title, s.audio_path.name if s.audio_path else "(없음)", len(s.lyrics)))
            matched = sum(bool(s.audio_path) for s in self.songs)
            skipped = list(SKIPPED_TRACKS)
            if skipped:
                self.status.set(
                    f"가사 {len(self.songs)}곡 / 음원 매칭 {matched}곡 "
                    f"/ 가사 없어 제외 {len(skipped)}곡"
                )
                messagebox.showwarning(
                    "가사가 비어 있는 곡",
                    "아래 곡은 JSON에 가사 내용이 없어 목록에서 제외했습니다.\n\n"
                    + "\n".join(skipped)
                    + "\n\n원본 가사 파일을 확인해 주세요.",
                )
            else:
                self.status.set(f"가사 {len(self.songs)}곡 / 음원 매칭 {matched}곡")
        except Exception as e:
            messagebox.showerror("불러오기 오류", str(e))

    def _source_lang_code(self) -> str:
        """드롭다운의 "fr (프랑스어)" 표기에서 언어 코드만 꺼냅니다."""
        return self.source_lang.get().split(" ", 1)[0].strip() or "auto"

    def _needs_translation(self, profiles: list[str]) -> bool:
        langs = required_languages(profiles)
        return any(lang != song.source_language for song in self.songs for lang in langs)

    def _build_queue_job(self) -> tuple[QueueJob, list]:
        lyrics = Path(self.lyrics_path.get().strip()) if self.lyrics_path.get().strip() else None
        if not lyrics or not lyrics.is_file():
            raise ValueError("가사 파일이 존재하지 않습니다.")
        songs = parse_lyrics_file(lyrics, self._source_lang_code())
        if not songs:
            raise ValueError("가사 파일에서 처리할 곡을 찾지 못했습니다.")
        folder = None
        if self.selected_audio_files:
            audios = list(self.selected_audio_files)
        else:
            folder_text = self.audio_folder.get().strip()
            folder = Path(folder_text) if folder_text else None
            if not folder or not folder.is_dir():
                raise ValueError("음원 폴더가 존재하지 않습니다.")
            audios = discover_audio(folder)
        if not audios or any(not path.is_file() for path in audios):
            raise ValueError("선택한 음원 파일이 존재하지 않습니다.")
        matched = match_audio(songs, audios)
        missing = [song.title for song in matched if not song.audio_path]
        if missing:
            raise ValueError("가사와 매칭되지 않은 음원이 있습니다:\n" + "\n".join(missing[:10]))
        output_text = self.output_folder.get().strip()
        if not output_text:
            raise ValueError("출력 폴더를 선택하세요.")
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        if not output.is_dir():
            raise ValueError("출력 경로가 올바른 폴더가 아닙니다.")
        profiles = [p for p, variable in self.profile_vars.items() if variable.get()]
        if not profiles:
            raise ValueError("생성할 자막 형식을 최소 1개 선택하세요.")
        engine = self.translation_engine()
        langs = required_languages(profiles)
        needs_translation = any(lang != song.source_language for song in matched for lang in langs)
        if needs_translation and engine == "gemini" and not self.gemini_api_key.get().strip():
            raise ValueError("번역에 필요한 Gemini API Key를 애플리케이션 설정에 입력하세요.")
        return QueueJob(
            lyrics_path=lyrics, audio_folder=folder,
            audio_files=[] if folder else audios, output_folder=output,
            source_language=self._source_lang_code(), model_name=self.model_name.get(),
            sync_mode=self.sync_mode(), fallback_enabled=bool(self.fallback_enabled.get()),
            profiles=profiles, translation_engine=engine,
            translation_model=self.translation_model.get().strip() or "gemini-2.5-flash",
            gap_seconds=float(self.gap_seconds.get()), title_seconds=float(self.title_seconds.get()),
            refine_timestamps=bool(self.refine_timestamps.get()),
        ), matched

    def add_current_to_queue(self):
        if self._queue_running:
            messagebox.showwarning("대기열 실행 중", "실행 중에는 작업을 추가할 수 없습니다.")
            return
        try:
            job, _songs = self._build_queue_job()
            self.queue.add(job)
        except (ValueError, OSError) as exc:
            messagebox.showwarning("대기열 등록 실패", str(exc))
            return
        self._refresh_queue_tree()
        self.status.set(f"대기열에 추가했습니다: {job.lyrics_path.name}")

    def _selected_queue_index(self):
        selection = self.queue_tree.selection()
        return int(selection[0]) if selection else None

    def _refresh_queue_tree(self, select_index=None):
        self.queue_tree.delete(*self.queue_tree.get_children())
        for index, job in enumerate(self.queue.jobs):
            if job.audio_folder:
                try:
                    audio_label = f"{job.audio_folder.name} ({len(discover_audio(job.audio_folder))}곡)"
                except Exception:
                    audio_label = job.audio_folder.name
            else:
                audio_label = f"선택 파일 {len(job.audio_files)}곡"
            self.queue_tree.insert("", "end", iid=str(index), values=(
                index + 1, job.lyrics_path.name, audio_label, str(job.output_folder), job.status,
            ))
        if select_index is not None and 0 <= select_index < len(self.queue.jobs):
            self.queue_tree.selection_set(str(select_index))

    def remove_queue_job(self):
        if self._queue_running:
            messagebox.showwarning("대기열", "실행 중에는 대기열을 편집할 수 없습니다.")
            return
        index = self._selected_queue_index()
        if index is None:
            return
        try:
            self.queue.remove(index)
        except ValueError as exc:
            messagebox.showwarning("대기열", str(exc))
        self._refresh_queue_tree()

    def move_queue_job(self, delta):
        if self._queue_running:
            messagebox.showwarning("대기열", "실행 중에는 대기열을 편집할 수 없습니다.")
            return
        index = self._selected_queue_index()
        if index is None:
            return
        try:
            target = self.queue.move(index, delta)
        except ValueError as exc:
            messagebox.showwarning("대기열", str(exc))
            return
        self._refresh_queue_tree(target)

    def reset_queue_job(self):
        if self._queue_running:
            messagebox.showwarning("대기열", "실행 중에는 대기열을 편집할 수 없습니다.")
            return
        index = self._selected_queue_index()
        if index is None:
            return
        try:
            self.queue.reset(index)
        except ValueError as exc:
            messagebox.showwarning("대기열", str(exc))
        self._refresh_queue_tree(index)

    def clear_queue(self):
        if self._queue_running:
            messagebox.showwarning("대기열", "실행 중에는 대기열을 편집할 수 없습니다.")
            return
        try:
            self.queue.clear()
        except ValueError as exc:
            messagebox.showwarning("대기열", str(exc))
        self._refresh_queue_tree()

    def start_queue(self):
        if self._queue_running or self._single_running:
            if self._single_running:
                messagebox.showwarning("작업 실행 중", "현재 단일 작업이 끝난 뒤 대기열을 실행하세요.")
            return
        if not any(job.status in {WAITING, FAILED, CANCELLED} for job in self.queue.jobs):
            messagebox.showinfo("대기열", "실행할 대기/실패/중단 작업이 없습니다.")
            return
        self.save_translation_settings(silent=True)
        self._queue_api_key = self.gemini_api_key.get().strip()
        self.cancel_event.clear()
        self._queue_running = True
        self.run_btn.config(state="disabled")
        self.queue_run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start(12)
        threading.Thread(target=self._queue_worker, daemon=True).start()

    def _queue_worker(self):
        try:
            report = self.queue.run(
                self._process_queue_job,
                should_cancel=self.cancel_event.is_set,
                progress=lambda no, total, job, msg: self.after(
                    0, self._queue_progress_update, no, total, job, msg
                ),
                report_path=ROOT / "queue_report.json",
            )
            self.after(0, self._queue_finished, report)
        except Exception as exc:
            details = traceback.format_exc()
            self.after(0, self._queue_failed, str(exc), details)

    def _process_queue_job(self, job, progress):
        songs = parse_lyrics_file(job.lyrics_path, job.source_language)
        audios = discover_audio(job.audio_folder) if job.audio_folder else list(job.audio_files)
        songs = match_audio(songs, audios)
        if any(not song.audio_path for song in songs):
            raise RuntimeError("실행 시점에 가사/음원 매칭이 변경되었습니다.")
        return process_songs(
            songs, job.output_folder, job.profiles, model_name=job.model_name,
            gap_seconds=job.gap_seconds, progress=progress,
            translation_engine=job.translation_engine,
            translation_api_key=self._queue_api_key,
            translation_model=job.translation_model, sync_mode=job.sync_mode,
            refine_timestamps=job.refine_timestamps,
            should_cancel=self.cancel_event.is_set, title_seconds=job.title_seconds,
            fallback_enabled=job.fallback_enabled,
        )

    def _queue_progress_update(self, no, total, job, message):
        self._refresh_queue_tree()
        self.queue_progress.configure(maximum=max(1, total), value=no - (0 if job.status in {COMPLETED, FAILED} else 1))
        self.queue_summary.set(f"전체 진행: 작업 {no} / {total}    현재 작업: {job.lyrics_path.stem}")
        self.status.set(message)

    def _queue_finished(self, report):
        self._queue_running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.queue_run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._refresh_queue_tree()
        self.queue_progress.configure(value=report["completed"] + report["failed"])
        summary = f"완료 {report['completed']} / 실패 {report['failed']} / 중단 {report['cancelled']}"
        self.queue_summary.set("전체 진행: " + summary)
        self.status.set(summary)
        messagebox.showinfo("대기열 실행 결과", summary + f"\n\n{ROOT / 'queue_report.json'}")

    def _queue_failed(self, message, details):
        self._queue_running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.queue_run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        log = ROOT / "error.log"
        log.write_text(details, encoding="utf-8")
        messagebox.showerror("대기열 오류", f"{message}\n\n자세한 내용: {log}")

    def start_process(self):
        if self._queue_running or self._single_running:
            return
        self.load_and_match()
        if not self.songs:
            messagebox.showwarning("확인", "가사 파일을 먼저 선택하세요.")
            return
        missing = [s.title for s in self.songs if not s.audio_path]
        if missing:
            messagebox.showwarning("음원 부족", "연결되지 않은 곡이 있습니다:\n" + "\n".join(missing[:10]))
            return
        profiles = [p for p, v in self.profile_vars.items() if v.get()]
        if not profiles:
            messagebox.showwarning("확인", "생성할 자막 형식을 하나 이상 선택하세요.")
            return

        engine = self.translation_engine()
        if self._needs_translation(profiles) and engine == "gemini" and not self.gemini_api_key.get().strip():
            messagebox.showwarning(
                "Gemini API 키 필요",
                "한국어/일본어를 자연스럽게 자동 번역하려면 Gemini API 키가 필요합니다.\n\n"
                "위의 'Gemini API 키' 칸에 키를 입력한 뒤 다시 실행하세요.\n"
                "인터넷 없이 사용하려면 번역 방식을 '오프라인 빠른번역(Argos)'으로 바꿀 수 있지만 직역이 생길 수 있습니다.",
            )
            return

        self.save_translation_settings(silent=True)
        self.cancel_event.clear()
        self._single_running = True
        self.run_btn.config(state="disabled")
        self.queue_run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start(12)
        self.status.set("작업 시작... 정밀 음악 싱크는 보컬 분리/음원 분석 때문에 시간이 걸릴 수 있습니다.")
        threading.Thread(target=self._worker, args=(profiles,), daemon=True).start()

    def stop_process(self):
        """진행 중인 작업을 멈춥니다.

        지금까지의 보컬 분리, 음성인식, 번역 결과는 캐시에 남으므로
        다시 실행하면 끝난 곡은 건너뛰고 이어서 진행합니다.
        """
        if not self.cancel_event.is_set():
            self.cancel_event.set()
            self.stop_btn.config(state="disabled")
            self.status.set("중단 요청됨... 진행 중인 단계를 정리하는 중입니다.")

    def _worker(self, profiles):
        try:
            results = process_songs(
                self.songs,
                Path(self.output_folder.get()),
                profiles,
                model_name=self.model_name.get(),
                gap_seconds=float(self.gap_seconds.get()),
                progress=lambda msg: self.after(0, self.status.set, msg),
                translation_engine=self.translation_engine(),
                translation_api_key=self.gemini_api_key.get().strip(),
                translation_model=self.translation_model.get().strip() or "gemini-2.5-flash",
                sync_mode=self.sync_mode(),
                refine_timestamps=bool(self.refine_timestamps.get()),
                should_cancel=self.cancel_event.is_set,
                title_seconds=float(self.title_seconds.get()),
                fallback_enabled=bool(self.fallback_enabled.get()),
            )
            out = Path(self.output_folder.get())
            self.after(0, self._done, len(results), out)
        except JobCancelled as e:
            self.after(0, self._cancelled, str(e))
        except Exception as e:
            details = traceback.format_exc()
            self.after(0, self._failed, str(e), details)

    def _cancelled(self, message):
        self._single_running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.queue_run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set("중단됨 - 다시 실행하면 이어서 진행합니다.")
        messagebox.showinfo("중단됨", message)

    def _done(self, count, out):
        self._single_running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.queue_run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set(f"완료: {count}곡")
        if messagebox.askyesno("완료", f"{count}곡 자막을 생성했습니다.\n\n{out}\n\n출력 폴더를 열까요?"):
            try:
                os.startfile(out)  # Windows
            except Exception:
                pass

    def _failed(self, message, details):
        self._single_running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.queue_run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set("오류 발생")
        log = ROOT / "error.log"
        log.write_text(details, encoding="utf-8")
        messagebox.showerror("작업 오류", f"{message}\n\n자세한 내용: {log}")


if __name__ == "__main__":
    App().mainloop()
