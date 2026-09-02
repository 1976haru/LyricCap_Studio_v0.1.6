from __future__ import annotations
from pathlib import Path
from .models import Cue, Song


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def profile_text(cue: Cue, profile: str) -> str:
    if profile == "en":
        return cue.text("en")
    if profile == "en_ko":
        return f"{cue.text('en')}\n{cue.text('ko')}"
    if profile == "en_ja":
        return f"{cue.text('en')}\n{cue.text('ja')}"
    if profile == "ko":
        return cue.text("ko")
    if profile == "ja":
        return cue.text("ja")
    raise ValueError(profile)


def render(cues: list[Cue], profile: str, offset: float = 0.0) -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        text = profile_text(cue, profile)
        blocks.append(f"{i}\n{srt_time(cue.start + offset)} --> {srt_time(cue.end + offset)}\n{text.strip()}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, cues: list[Cue], profile: str, offset: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cues, profile, offset), encoding="utf-8-sig")


def write_combined(path: Path, tracks: list[tuple[list[Cue], float]], profile: str):
    all_cues: list[Cue] = []
    for cues, offset in tracks:
        for c in cues:
            shifted = Cue(c.start + offset, c.end + offset, c.source, c.source_language, c.en, c.ko, c.ja)
            all_cues.append(shifted)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(all_cues, profile, 0.0), encoding="utf-8-sig")


def title_card_text(song: Song) -> str:
    label = f"{song.track_no:02d}. {song.title}"
    localized = song.localized_title.strip()
    if localized and localized != song.title:
        label += f" ({localized})"
    return label


def write_titles_srt(
    path: Path,
    tracks: list[tuple[Song, float, float]],
    display_seconds: float = 5.0,
):
    """곡이 시작할 때 곡 제목을 표시하는 별도 SRT를 만듭니다.

    tracks: [(Song, offset, duration), ...]. offset/duration은 batch.py가
    통합 SRT를 만들 때 쓰는 값을 그대로 재사용해야 가사 자막과 어긋나지 않습니다.
    """
    blocks = []
    for i, (song, offset, duration) in enumerate(tracks, 1):
        end = offset + min(display_seconds, duration)
        text = title_card_text(song)
        blocks.append(f"{i}\n{srt_time(offset)} --> {srt_time(end)}\n{text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8-sig")
