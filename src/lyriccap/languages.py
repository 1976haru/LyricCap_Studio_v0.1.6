"""지원 언어와 언어 감지를 한 곳에서 정의합니다.

JSON의 meta.lyricLanguage에는 단순한 ``japanese`` 대신
``Japanese-dominant with sparse English code-switch``처럼 제작 지침이 함께
들어갈 수 있습니다. 정확히 일치하는 별칭만 허용하면 이런 값이 영어 기본값으로
되돌아가 Whisper가 일본어 음원을 영어로 받아쓰게 됩니다.

이 모듈은
1) 정확한 별칭,
2) 설명형 문자열 안의 첫 번째 언어 이름,
3) 실제 가사 문자권
순서로 원문 언어를 판정합니다.
"""

from __future__ import annotations

import re
import unicodedata

LANG_NAMES_KO: dict[str, str] = {
    "en": "영어",
    "ko": "한국어",
    "ja": "일본어",
    "fr": "프랑스어",
    "es": "스페인어",
    "de": "독일어",
    "it": "이탈리아어",
}

SOURCE_LANGUAGES: list[str] = ["en", "ko", "ja", "fr", "es", "de", "it"]
TARGET_LANGUAGES: list[str] = ["en", "ko", "ja"]
WHISPER_LANGUAGES: frozenset[str] = frozenset(SOURCE_LANGUAGES)
CHARACTER_LANGUAGES: frozenset[str] = frozenset({"ja", "zh"})

LANG_ALIASES: dict[str, str] = {
    "english": "en", "영어": "en",
    "korean": "ko", "한국어": "ko",
    "japanese": "ja", "일본어": "ja",
    "french": "fr", "francais": "fr", "français": "fr", "프랑스어": "fr", "불어": "fr",
    "spanish": "es", "espanol": "es", "español": "es", "castellano": "es", "스페인어": "es",
    "german": "de", "deutsch": "de", "독일어": "de",
    "italian": "it", "italiano": "it", "이탈리아어": "it",
}
LANG_ALIASES.update({code: code for code in SOURCE_LANGUAGES})

# 언어 이름 앞뒤를 이루는 문자. ``ja``가 ``japanese`` 안에서 잘못 잡히거나
# ``en``이 ``french`` 안에서 잡히지 않도록 코드/별칭 경계를 엄격히 봅니다.
_LANGUAGE_WORD_CHARS = r"0-9A-Za-zÀ-ÖØ-öø-ÿ가-힣ぁ-んァ-ヶ一-龯"


def _normalize_label(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def normalize_language(value: str | None, default: str = "en") -> str:
    """사람이 쓴 언어 표기를 코드로 바꿉니다.

    정확한 별칭뿐 아니라 ``Japanese-dominant ...`` 같은 설명형 표기도
    문자열에서 가장 먼저 명시된 언어를 원문 언어로 해석합니다.
    모르는 값이면 ``default``를 반환합니다. 호출자가 미판정을 구분하려면
    ``default=""``를 넘길 수 있습니다.
    """
    if not value:
        return default

    raw = _normalize_label(value)
    exact = LANG_ALIASES.get(raw)
    if exact:
        return exact

    found: list[tuple[int, int, str]] = []
    for alias, code in LANG_ALIASES.items():
        pattern = rf"(?<![{_LANGUAGE_WORD_CHARS}]){re.escape(alias)}(?![{_LANGUAGE_WORD_CHARS}])"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            # 같은 위치라면 'ja'보다 'japanese'처럼 긴 별칭을 우선합니다.
            found.append((match.start(), -len(alias), code))
    if found:
        found.sort()
        return found[0][2]
    return default


def detect_language_from_text(text: str | None, default: str = "en") -> str:
    """가사 문자권으로 일본어/한국어를 안전하게 보조 판정합니다.

    라틴 문자권 언어끼리(en/fr/es/de/it)는 문자만으로 신뢰성 있게 구별할 수
    없으므로 메타데이터가 없을 때 기본값을 유지합니다. 반면 일본어와 한국어는
    히라가나·가타카나·한글을 통해 안정적으로 구별할 수 있습니다.
    """
    if not text:
        return default
    sample = unicodedata.normalize("NFKC", str(text))
    kana = len(re.findall(r"[ぁ-んァ-ヶー]", sample))
    hangul = len(re.findall(r"[가-힣]", sample))
    latin = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", sample))
    kanji = len(re.findall(r"[一-龯]", sample))

    # 일본어 가사는 한자를 많이 쓰므로, 가나가 실제로 존재할 때만 한자에
    # 보조 가중치를 줍니다. sparse English code-switch는 판정을 뒤집지 않습니다.
    ja_score = kana * 3 + (kanji if kana >= 2 else 0)
    ko_score = hangul * 3

    if kana >= 4 and ja_score > ko_score and ja_score >= max(8, latin * 0.35):
        return "ja"
    if hangul >= 4 and ko_score > ja_score and ko_score >= max(8, latin * 0.35):
        return "ko"
    return default


def resolve_language(value: str | None, lyric_text: str | None = None, default: str = "en") -> str:
    """메타 언어값을 우선하고, 미판정이면 실제 가사 문자권으로 보완합니다."""
    explicit = normalize_language(value, default="")
    if explicit:
        return explicit
    return detect_language_from_text(lyric_text, default=default)


def language_name(code: str) -> str:
    return LANG_NAMES_KO.get(code, code)


def whisper_language(code: str | None) -> str | None:
    return code if code in WHISPER_LANGUAGES else None
