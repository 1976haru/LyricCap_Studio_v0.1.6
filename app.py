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
TRANSLATION_ENGINE_LABELS = {
    "AI 자연번역 (Gemini, 권장)": "gemini",
    "오프라인 빠른번역 (Argos, 직역 가능)": "argos",
}
SETTINGS_PATH = ROOT / "translation_settings.json"


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
        self.title("LyricCap Studio v0.1.8 - Demucs + Whisper ASR Anchor Sync + CapCut SRT")
        self.geometry("1100x820")
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
        self.profile_vars = {p: tk.BooleanVar(value=True) for p in PROFILES}

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
        ttk.Combobox(opts, textvariable=self.source_lang, state="readonly", width=10,
                     values=["auto", "en", "ko", "ja"]).grid(row=0, column=1, padx=8, pady=8)
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

        table_frame = ttk.LabelFrame(self, text="곡 매칭")
        table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        cols = ("no", "title", "audio", "lines")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=13)
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
            songs = parse_lyrics_file(lyrics, self.source_lang.get())
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

    def _needs_translation(self, profiles: list[str]) -> bool:
        langs = required_languages(profiles)
        return any(lang != song.source_language for song in self.songs for lang in langs)

    def start_process(self):
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
        self.run_btn.config(state="disabled")
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
            )
            out = Path(self.output_folder.get())
            self.after(0, self._done, len(results), out)
        except JobCancelled as e:
            self.after(0, self._cancelled, str(e))
        except Exception as e:
            details = traceback.format_exc()
            self.after(0, self._failed, str(e), details)

    def _cancelled(self, message):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set("중단됨 - 다시 실행하면 이어서 진행합니다.")
        messagebox.showinfo("중단됨", message)

    def _done(self, count, out):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set(f"완료: {count}곡")
        if messagebox.askyesno("완료", f"{count}곡 자막을 생성했습니다.\n\n{out}\n\n출력 폴더를 열까요?"):
            try:
                os.startfile(out)  # Windows
            except Exception:
                pass

    def _failed(self, message, details):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set("오류 발생")
        log = ROOT / "error.log"
        log.write_text(details, encoding="utf-8")
        messagebox.showerror("작업 오류", f"{message}\n\n자세한 내용: {log}")


if __name__ == "__main__":
    App().mainloop()
