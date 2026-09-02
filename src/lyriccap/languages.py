"""지원 언어를 한 곳에서 정의합니다.

en/ko/ja 세 언어가 parsers, aligner, batch, app, translator에 각각 하드코딩되어
있었습니다. 그래서 불어 가사를 넣으면

  1) parsers가 meta.lyricLanguage="french"를 모르고 "en"으로 되돌리고,
  2) aligner가 Whisper에 language="en"을 넘겨 불어 노래를 영어로 받아쓰게 하고,
  3) 그 받아쓰기와 불어 가사를 맞추다 매칭률이 바닥(15.9%)으로 떨어져 중단

되는 흐름이 만들어졌습니다. 언어를 늘릴 때 이 파일만 고치면 되도록 모았습니다.
"""

from __future__ import annotations

LANG_NAMES_KO: dict[str, str] = {
    "en": "영어",
    "ko": "한국어",
    "ja": "일본어",
    "fr": "프랑스어",
    "es": "스페인어",
    "de": "독일어",
    "it": "이탈리아어",
}

# 가사 원문으로 고를 수 있는 언어. Whisper가 모두 학습한 언어 코드입니다.
SOURCE_LANGUAGES: list[str] = ["en", "ko", "ja", "fr", "es", "de", "it"]

# 번역해서 자막으로 뽑을 수 있는 언어. models.Cue의 en/ko/ja 필드와 짝입니다.
# 원문 언어와는 별개입니다. (불어 원문 -> 한국어 자막이 성립합니다.)
TARGET_LANGUAGES: list[str] = ["en", "ko", "ja"]

# Whisper에 language= 로 넘겨도 되는 코드.
WHISPER_LANGUAGES: frozenset[str] = frozenset(SOURCE_LANGUAGES)

# 글자 단위로 쪼개 정렬하는 언어. 단어 경계가 없어 Whisper의 단어 묶음이
# 들쭉날쭉한 언어들입니다.
CHARACTER_LANGUAGES: frozenset[str] = frozenset({"ja", "zh"})

# meta.lyricLanguage처럼 사람이 적은 표기를 코드로 바꿉니다.
LANG_ALIASES: dict[str, str] = {
    "english": "en", "영어": "en",
    "korean": "ko", "한국어": "ko",
    "japanese": "ja", "일본어": "ja",
    "french": "fr", "francais": "fr", "français": "fr", "프랑스어": "fr", "불어": "fr",
    "spanish": "es", "espanol": "es", "español": "es", "castellano": "es", "스페인어": "es",
    "german": "de", "deutsch": "de", "독일어": "de",
    "italian": "it", "italiano": "it", "이탈리아어": "it",
}
# 코드를 그대로 적어 넣은 경우도 받습니다.
LANG_ALIASES.update({code: code for code in SOURCE_LANGUAGES})


def normalize_language(value: str | None, default: str = "en") -> str:
    """사람이 쓴 언어 표기를 코드로 바꿉니다. 모르는 값이면 default."""
    if not value:
        return default
    return LANG_ALIASES.get(str(value).strip().lower(), default)


def language_name(code: str) -> str:
    """진행 메시지와 번역 프롬프트에 쓰는 한국어 언어 이름."""
    return LANG_NAMES_KO.get(code, code)


def whisper_language(code: str | None) -> str | None:
    """Whisper에 넘길 언어 코드. 모르는 언어면 None(자동 감지)."""
    return code if code in WHISPER_LANGUAGES else None
