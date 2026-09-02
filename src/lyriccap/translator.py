from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 언어 이름 표는 languages.py 한 곳에 있습니다. 프롬프트 코드가 이 이름을
# 널리 참조하고 있어 모듈 이름으로 그대로 다시 내보냅니다.
from .languages import LANG_NAMES_KO


class _RateLimited(RuntimeError):
    """재시도하면 풀릴 수 있는 일시적 오류.

    kind="quota"  : HTTP 429. 요청을 너무 자주 보낸 경우.
    kind="server" : HTTP 5xx. 구글 서버가 혼잡하거나 불안정한 경우.
                    우리 설정 문제가 아니므로 더 오래, 더 여러 번 기다립니다.
    """

    def __init__(self, detail: str = "", kind: str = "quota"):
        super().__init__(f"retryable: {kind}")
        self.detail = detail or ""
        self.kind = kind


class _ModelUnavailable(RuntimeError):
    """설정된 모델이 404인 경우. 키 문제가 아니라 모델 이름 문제입니다."""


class TranslationCache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            self.data = {}

    def get(self, key: str) -> object | None:
        return self.data.get(key)

    def set(self, key: str, value: object) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def batch_key(
        engine: str,
        model: str,
        src: str,
        dst: str,
        title: str,
        texts: list[str],
        variant: str = "",
    ) -> str:
        """캐시 키.

        variant는 프롬프트 버전이나 곡 분석 요약처럼 '같은 가사라도 결과가
        달라지는 조건'을 구분하기 위한 값입니다. 프롬프트를 개선했는데 예전
        번역이 캐시에서 그대로 나오는 일을 막습니다.
        """
        raw = json.dumps(
            {
                "engine": engine,
                "model": model,
                "src": src,
                "dst": dst,
                "title": title,
                "texts": texts,
                "variant": variant,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return f"batch:{digest}"


class ArgosTranslator:
    """Offline translation with Argos Translate.

    This mode is intentionally kept as a fast/offline fallback. It may be more
    literal than the AI natural-translation mode.
    """

    engine_name = "argos"
    model_name = "offline"

    def __init__(self, cache_path: Path):
        self.cache = TranslationCache(cache_path)

    @staticmethod
    def _imports():
        try:
            import argostranslate.package
            import argostranslate.translate
            return argostranslate.package, argostranslate.translate
        except ImportError as e:
            raise RuntimeError("Argos Translate가 설치되지 않았습니다.") from e

    @staticmethod
    def _language_by_code(translate, code: str):
        try:
            getter = getattr(translate, "get_language_from_code", None)
            if getter is not None:
                lang = getter(code)
                if lang is not None:
                    return lang
        except Exception:
            pass

        try:
            return next(
                (lang for lang in translate.get_installed_languages()
                 if getattr(lang, "code", None) == code),
                None,
            )
        except Exception:
            return None

    def _pair_installed(self, src: str, dst: str) -> bool:
        if src == dst:
            return True

        _, translate = self._imports()

        # Argos translations_to contains translation objects, not Language
        # objects. Use the public translation APIs instead of reading .code
        # from those objects (fixes IdentityTranslation .code crash).
        try:
            getter = getattr(translate, "get_translation_from_codes", None)
            if getter is not None:
                tr = getter(src, dst)
                if tr is not None:
                    to_lang = getattr(tr, "to_lang", None)
                    return getattr(to_lang, "code", dst) == dst
        except Exception:
            pass

        src_lang = self._language_by_code(translate, src)
        dst_lang = self._language_by_code(translate, dst)
        if src_lang is None or dst_lang is None:
            return False
        try:
            tr = src_lang.get_translation(dst_lang)
            return tr is not None
        except Exception:
            return False

    def _install_direct(self, src: str, dst: str) -> bool:
        package, _ = self._imports()

        helper = getattr(package, "install_package_for_language_pair", None)
        if helper is not None:
            try:
                package.update_package_index()
                if bool(helper(src, dst)):
                    return True
            except Exception:
                pass

        package.update_package_index()
        packages = package.get_available_packages()
        match = next(
            (
                p for p in packages
                if getattr(p, "from_code", None) == src
                and getattr(p, "to_code", None) == dst
            ),
            None,
        )
        if match is None:
            return False
        package.install_from_path(match.download())
        return self._pair_installed(src, dst)

    def ensure_route(self, src: str, dst: str) -> None:
        if src == dst or self._pair_installed(src, dst):
            return

        if self._install_direct(src, dst):
            return

        if src != "en" and dst != "en":
            if not self._pair_installed(src, "en"):
                if not self._install_direct(src, "en"):
                    raise RuntimeError(f"번역 패키지 {src}→en을 찾을 수 없습니다.")
            if not self._pair_installed("en", dst):
                if not self._install_direct("en", dst):
                    raise RuntimeError(f"번역 패키지 en→{dst}를 찾을 수 없습니다.")
            if self._pair_installed(src, dst):
                return
            if self._pair_installed(src, "en") and self._pair_installed("en", dst):
                return

        raise RuntimeError(f"Argos 번역 패키지 {src}→{dst}를 찾을 수 없습니다.")

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if src == dst:
            return text
        _, translate = self._imports()
        try:
            return str(translate.translate(text, src, dst))
        except Exception as direct_error:
            if src != "en" and dst != "en":
                try:
                    mid = str(translate.translate(text, src, "en"))
                    return str(translate.translate(mid, "en", dst))
                except Exception:
                    pass
            raise RuntimeError(
                f"번역 실행 실패 ({src}→{dst}): {direct_error}"
            ) from direct_error

    def translate_many(
        self,
        texts: list[str],
        src: str,
        dst: str,
        *,
        title: str = "",
    ) -> list[str]:
        if src == dst:
            return list(texts)

        self.ensure_route(src, dst)
        key = self.cache.batch_key(self.engine_name, self.model_name, src, dst, title, texts)
        cached = self.cache.get(key)
        if isinstance(cached, list) and len(cached) == len(texts):
            return [str(x) for x in cached]

        out = [self._translate_one(text, src, dst) for text in texts]
        self.cache.set(key, out)
        self.cache.save()
        return out


class GeminiNaturalTranslator:
    """Context-aware, non-literal lyric subtitle translation via Gemini API.

    The full song (or cue list) is translated together so the model can keep
    pronouns, imagery, emotional tone, and recurring hook phrases consistent.
    The output is validated to keep exactly one translated subtitle line per
    source line, so existing SRT timing remains intact.
    """

    engine_name = "gemini"

    def __init__(
        self,
        cache_path: Path,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        timeout: int = 120,
        progress=None,
        check_cancel=None,
    ):
        self.cache = TranslationCache(cache_path)
        self.progress = progress or (lambda _msg: None)
        # 호출하면 중단 시 예외를 던지는 함수. 긴 대기 중에도 반응하기 위함입니다.
        self.check_cancel = check_cancel or (lambda: None)
        self._last_request_at = 0.0
        self._interval = float(self.MIN_REQUEST_INTERVAL)
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model_name or "gemini-2.5-flash").strip()
        self.timeout = int(timeout)
        self._tried_models = None
        self._supports_thinking_config = True
        if not self.api_key:
            raise RuntimeError(
                "자연번역을 사용하려면 Gemini API 키가 필요합니다.\n"
                "프로그램의 'Gemini API 키' 칸에 키를 한 번 입력하고 저장하세요."
            )

    # 프롬프트를 바꿀 때마다 올립니다. 캐시가 예전 번역을 돌려주는 것을 막습니다.
    PROMPT_VERSION = "v2-context-length"

    # 자막 한 줄 권장/최대 글자수. 화면에서 한눈에 읽히는 길이 기준입니다.
    LINE_BUDGET = {
        "ko": (20, 30),
        "ja": (18, 28),
        "en": (42, 60),
    }

    @classmethod
    def _analysis_prompt(cls, texts: list[str], src: str, title: str) -> str:
        src_name = LANG_NAMES_KO.get(src, src)
        payload = json.dumps(texts, ensure_ascii=False, indent=2)
        return f"""당신은 노래 가사를 분석하는 번역 감수자입니다.
아래 {src_name} 가사 전체를 읽고, 번역가가 참고할 짧은 메모를 작성하세요.

곡 제목: {title or '(제목 없음)'}

판단할 항목:
- speaker: 화자가 누구인지, 누구에게 말하는지 (한 문장)
- situation: 곡 전체의 상황이나 장면 (한 문장)
- mood: 정서와 분위기 (쉼표로 구분한 단어 3~5개)
- register: 번역할 때 쓸 말투. 한국어라면 반말/존댓말 중 무엇이 맞는지,
  일본어라면 だ체/です체 중 무엇이 맞는지 근거와 함께 한 문장
- hooks: 반복되는 후렴이나 훅 문장을 원문 그대로 최대 3개
- glossary: 곡 전체에서 일관되게 옮겨야 할 핵심 낱말이나 이미지 최대 6개.
  각 항목은 {{"term": "원문", "note": "어떤 뜻/느낌으로 다뤄야 하는지"}}

가사에 없는 내용을 상상해서 채우지 마세요. 판단이 어려우면 빈 문자열로 두세요.

반환 형식은 반드시 아래 JSON 하나뿐입니다.
{{"speaker":"","situation":"","mood":"","register":"","hooks":[],"glossary":[]}}

가사:
{payload}
"""

    @classmethod
    def _format_brief(cls, brief: dict | None) -> str:
        if not isinstance(brief, dict):
            return ""
        lines: list[str] = []
        if brief.get("speaker"):
            lines.append(f"- 화자/청자: {brief['speaker']}")
        if brief.get("situation"):
            lines.append(f"- 상황: {brief['situation']}")
        if brief.get("mood"):
            lines.append(f"- 정서: {brief['mood']}")
        if brief.get("register"):
            lines.append(f"- 말투: {brief['register']}")

        hooks = brief.get("hooks")
        if isinstance(hooks, list) and hooks:
            joined = " / ".join(str(h).strip() for h in hooks[:3] if str(h).strip())
            if joined:
                lines.append(f"- 반복 훅(전곡에서 동일하게 번역): {joined}")

        glossary = brief.get("glossary")
        if isinstance(glossary, list) and glossary:
            items = []
            for g in glossary[:6]:
                if isinstance(g, dict) and str(g.get("term", "")).strip():
                    items.append(f"{str(g['term']).strip()} = {str(g.get('note', '')).strip()}")
            if items:
                lines.append("- 용어 일관성: " + " / ".join(items))

        if not lines:
            return ""
        return "이 곡에 대한 사전 분석 (번역 시 반영하세요):\n" + "\n".join(lines) + "\n"

    @classmethod
    def _prompt(cls, texts: list[str], src: str, dst: str, title: str, brief: dict | None = None) -> str:
        src_name = LANG_NAMES_KO.get(src, src)
        dst_name = LANG_NAMES_KO.get(dst, dst)
        soft, hard = cls.LINE_BUDGET.get(dst, (24, 36))
        items = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
        payload = json.dumps(items, ensure_ascii=False, indent=2)
        brief_block = cls._format_brief(brief)

        # 분석 호출이 실패하면 참조할 메모가 없으므로 문구를 바꿉니다.
        if brief_block:
            register_rule = "5. 말투는 위 사전 분석의 지침을 곡 전체에서 흔들림 없이 유지합니다."
        else:
            register_rule = (
                "5. 말투는 가사 전체를 먼저 훑어 하나로 정한 뒤, 곡 전체에서 "
                "흔들림 없이 유지합니다. 중간에 반말과 존댓말을 섞지 마세요."
            )

        return f"""당신은 노래 가사와 영상 자막을 전문적으로 번역하는 번역가입니다.

작업: {src_name} 노래 가사를 {dst_name} 영상 자막으로 번역합니다.
곡 제목: {title or '(제목 없음)'}

{brief_block}
반드시 지킬 원칙:
1. 단어를 1:1로 옮기는 직역을 피하고, {dst_name} 원어민이 실제 노래 자막에서 자연스럽게 읽을 표현으로 번역합니다.
2. 원문의 의미, 장면, 감정, 시대감, 인물 관계와 호칭은 보존합니다.
3. 시적 이미지는 자연스럽게 살리되 원문에 없는 사건이나 해석을 새로 만들지 않습니다.
4. 반복되는 후렴/훅은 같은 의미와 말투로 일관되게 번역합니다.
{register_rule}
6. 원문의 대명사나 생략된 주어는 문맥을 보고 {dst_name}에서 자연스럽게 처리합니다.

자막 길이 규칙 (매우 중요):
7. 번역문 한 줄은 공백 포함 {soft}자 내외로 씁니다. 어떤 경우에도 {hard}자를 넘기지 마세요.
8. 원문이 짧으면 번역도 짧게 씁니다. 짧은 원문을 길고 설명적인 문장으로 부풀리지 마세요.
9. 길이를 줄일 때는 뜻을 빼는 대신 조사, 군더더기 수식어, 당연한 주어를 생략합니다.
10. 문장 끝의 마침표는 생략합니다. 물음표와 느낌표는 필요할 때만 씁니다.

형식 규칙:
11. 각 입력 id마다 정확히 하나의 번역을 반환합니다. 줄을 합치거나 쪼개지 마세요.
12. 줄바꿈 문자를 번역문 안에 넣지 마세요. 한 줄로만 씁니다.
13. 번역문만 반환하고 해설, 주석, 로마자 발음은 넣지 마세요.

반환 형식은 반드시 아래 JSON 하나뿐입니다.
{{"translations":[{{"id":1,"text":"번역문"}},{{"id":2,"text":"번역문"}}]}}

입력:
{payload}
"""

    # 번역용으로 부적합한 모델(임베딩, 이미지, 음성 등)을 걸러냅니다.
    _MODEL_EXCLUDE = (
        "embedding", "aqa", "imagen", "image", "tts",
        "veo", "audio", "live", "nano-banana",
    )

    def list_available_models(self) -> list[str]:
        """이 API 키로 실제 호출 가능한 텍스트 모델 이름을 가져옵니다."""
        names: list[str] = []
        page_token = ""
        try:
            while True:
                url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
                if page_token:
                    url += "&pageToken=" + urllib.parse.quote(page_token)
                req = urllib.request.Request(url, headers={"x-goog-api-key": self.api_key})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                for m in obj.get("models") or []:
                    name = str(m.get("name", "")).replace("models/", "")
                    methods = m.get("supportedGenerationMethods") or m.get("supportedActions") or []
                    if "generateContent" not in methods:
                        continue
                    if any(bad in name.lower() for bad in self._MODEL_EXCLUDE):
                        continue
                    names.append(name)
                page_token = obj.get("nextPageToken") or ""
                if not page_token:
                    break
        except Exception:
            return names
        return names

    def _model_candidates(self) -> list[str]:
        """대체 모델 후보를 무료 한도가 넉넉한 순서로 돌려줍니다.

        예전에는 '가장 최신 flash'를 골랐는데, 이게 거꾸로였습니다.
        무료 등급에서는 최신 모델일수록 분당/하루 한도가 더 빡빡하고,
        lite 계열이 가장 여유롭습니다. 가사 번역은 프롬프트로 맥락과 규칙을
        충분히 주기 때문에 lite로도 품질이 잘 나옵니다.
        """
        available = self.list_available_models()
        if not available:
            return []

        def version_of(name: str) -> float:
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", name.lower())
            try:
                return float(m.group(1)) if m else 0.0
            except Exception:
                return 0.0

        def rank(name: str) -> tuple:
            low = name.lower()
            if "flash" in low and "lite" in low:
                tier = 0          # 무료 한도가 가장 넉넉함
            elif "flash" in low:
                tier = 1
            elif "pro" in low:
                tier = 2          # 무료 등급에서 한도가 가장 낮거나 유료 전용
            else:
                tier = 3
            preview = 1 if ("preview" in low or "exp" in low) else 0
            # 같은 등급 안에서는 안정 버전 중 최신을 씁니다.
            return (tier, preview, -version_of(name), name)

        available.sort(key=rank)
        return [n for n in available if n != self.model_name]

    def _resolve_model(self) -> str | None:
        candidates = self._model_candidates()
        return candidates[0] if candidates else None

    # 무료 등급은 분당 요청수(RPM)가 10~15회로 제한됩니다. 곡 15개를 두 언어로
    # 번역하면 분석까지 합쳐 45회가 되는데, 이를 연속으로 던지면 몇 초 만에
    # 분당 한도를 넘습니다. 하루 한도는 넉넉하므로 간격만 두면 해결됩니다.
    MIN_REQUEST_INTERVAL = 5.0    # 초. 시작 간격
    MAX_REQUEST_INTERVAL = 45.0   # 초. 적응해서 올릴 수 있는 상한
    MAX_RATE_LIMIT_RETRIES = 6
    # 서버 혼잡(503)은 우리가 아낀다고 풀리는 게 아니라 기다리면 풀립니다.
    # 한도 초과보다 넉넉하게 재시도합니다.
    MAX_SERVER_ERROR_RETRIES = 6

    def _sleep(self, seconds: float) -> None:
        """중단 버튼에 반응할 수 있도록 잘게 나눠 잡니다.

        한 번에 180초를 자면 중단을 눌러도 그만큼 기다려야 합니다.
        """
        end = time.monotonic() + max(0.0, seconds)
        while True:
            self.check_cancel()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self._interval - elapsed
        if wait > 0:
            self._sleep(wait)
        self._last_request_at = time.monotonic()

    def _learn_from_retry_delay(self, seconds: float) -> None:
        """구글이 요구한 대기 시간을 보고 요청 간격을 스스로 올립니다.

        모델마다 분당 한도가 다르고 공개 문서와도 어긋나는 경우가 있어서,
        고정 간격은 너무 짧거나 너무 길기 쉽습니다. 실제 응답을 근거로 맞춥니다.
        """
        if seconds <= 0:
            return
        target = min(self.MAX_REQUEST_INTERVAL, max(self._interval, seconds * 0.6))
        if target > self._interval + 0.5:
            self._interval = target
            self.progress(f"요청 간격을 {self._interval:.0f}초로 늘렸습니다.")

    @staticmethod
    def _retry_delay_seconds(detail: str, attempt: int) -> float:
        """응답 본문에 구글이 알려준 대기 시간이 있으면 그걸 씁니다."""
        try:
            obj = json.loads(detail)
            for item in (obj.get("error", {}).get("details") or []):
                value = str(item.get("retryDelay", "")).strip()
                if value.endswith("s"):
                    parsed = float(value[:-1])
                    if parsed > 0:
                        return min(120.0, parsed + 1.0)
        except Exception:
            pass
        # 지수 백오프: 15초, 30초, 60초, 120초
        return min(120.0, 15.0 * (2 ** attempt))

    def _call_api(self, prompt: str) -> str:
        last_error: _RateLimited | None = None
        quota_tries = 0
        server_tries = 0

        while True:
            try:
                return self._call_once(prompt)
            except _RateLimited as e:
                last_error = e

                if e.kind == "server":
                    server_tries += 1
                    if server_tries >= self.MAX_SERVER_ERROR_RETRIES:
                        break
                    # 서버 혼잡은 20초, 40초, 80초... 로 물러섭니다.
                    delay = min(180.0, 20.0 * (2 ** (server_tries - 1)))
                    self.progress(
                        f"Gemini 서버가 혼잡합니다(일시적). {delay:.0f}초 후 재시도 "
                        f"({server_tries}/{self.MAX_SERVER_ERROR_RETRIES - 1})"
                    )
                else:
                    quota_tries += 1
                    if quota_tries >= self.MAX_RATE_LIMIT_RETRIES:
                        break
                    delay = self._retry_delay_seconds(e.detail, quota_tries - 1)
                    self._learn_from_retry_delay(delay)
                    self.progress(
                        f"Gemini 분당 요청 한도에 걸렸습니다. {delay:.0f}초 후 재시도 "
                        f"({quota_tries}/{self.MAX_RATE_LIMIT_RETRIES - 1})"
                    )

                self._sleep(delay)
                self._last_request_at = time.monotonic()

        if last_error is not None and last_error.kind == "server":
            raise RuntimeError(
                "Gemini 서버가 계속 혼잡합니다(HTTP 503).\n"
                "구글 쪽 일시적인 문제이며 설정이나 API 키 문제가 아닙니다.\n"
                "잠시 후 다시 실행하거나, 'AI 모델' 칸에 다른 모델 이름을 넣어 보세요.\n"
                "이미 번역된 곡은 캐시에 남아 있어 처음부터 다시 하지 않습니다."
            ) from last_error

        raise RuntimeError(
            "Gemini API 사용량 제한이 계속됩니다.\n"
            "무료 등급의 하루 요청 한도를 모두 썼을 수 있습니다. "
            "내일 다시 시도하거나, 생성할 자막 언어를 하나만 선택해 호출 수를 줄여보세요."
        ) from last_error

    def _call_once(self, prompt: str) -> str:
        try:
            return self._post(prompt, self.model_name)
        except _ModelUnavailable as e:
            if self._tried_models is None:
                self._tried_models = self._model_candidates()

            # 첫 후보도 막힐 수 있으므로 될 때까지 순서대로 시도합니다.
            while self._tried_models:
                candidate = self._tried_models.pop(0)
                self.progress(f"'{self.model_name}' 사용 불가 → '{candidate}'로 전환합니다.")
                try:
                    result = self._post(prompt, candidate)
                except _ModelUnavailable:
                    continue
                self.model_name = candidate
                return result

            available = self.list_available_models()
            hint = ("\n지금 사용 가능한 모델: " + ", ".join(available[:10])) if available else ""
            raise RuntimeError(
                f"'{self.model_name}' 모델을 사용할 수 없고, 대체할 모델도 찾지 못했습니다.\n"
                f"프로그램의 'AI 모델' 칸에 사용 가능한 이름을 입력하세요.{hint}"
            ) from e

    def _post(self, prompt: str, model_name: str) -> str:
        self._throttle()
        model = urllib.parse.quote(model_name, safe="-._")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.35,
                "responseMimeType": "application/json",
                # 응답 길이 상한을 넉넉히 잡습니다. 지정하지 않으면 모델 기본값이
                # 작아서 가사가 긴 곡에서 JSON이 중간에 잘립니다.
                "maxOutputTokens": 16384,
            },
        }
        if self._supports_thinking_config:
            # 최신 flash 계열은 '생각하기'가 기본으로 켜져 있고, 그 토큰이
            # 출력 한도를 같이 먹어 JSON이 중간에 잘립니다. 번역은 긴 추론이
            # 필요 없으므로 끕니다. 이 항목을 모르는 모델은 400을 내므로
            # 그때는 빼고 다시 보냅니다.
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if e.code == 400 and "thinking" in detail.lower() and self._supports_thinking_config:
                # 이 모델은 thinkingConfig를 모릅니다. 빼고 한 번만 다시 보냅니다.
                self._supports_thinking_config = False
                self.progress("이 모델은 thinkingConfig를 지원하지 않아 해당 설정 없이 재시도합니다.")
                return self._post(prompt, model_name)
            if e.code in {401, 403}:
                raise RuntimeError(
                    "Gemini API 인증에 실패했습니다. API 키를 확인하세요."
                ) from e
            if e.code == 429:
                # 분당 한도라면 잠시 기다렸다 재시도하면 풀립니다.
                raise _RateLimited(detail, kind="quota") from e
            if e.code in (500, 502, 503, 504):
                # 구글 서버가 일시적으로 붐비거나 불안정한 경우입니다.
                # 우리 잘못이 아니고 대개 몇십 초 뒤에 정상화됩니다.
                raise _RateLimited(detail, kind="server") from e
            if e.code == 404:
                # 모델이 은퇴했거나 이 계정에 열려 있지 않은 경우입니다.
                # 키 자체는 정상이므로 다른 모델로 재시도할 수 있습니다.
                raise _ModelUnavailable(
                    f"'{model_name}' 모델을 사용할 수 없습니다: {detail[:300]}"
                ) from e
            raise RuntimeError(f"Gemini API 오류 HTTP {e.code}: {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Gemini API에 연결할 수 없습니다. 인터넷 연결을 확인하세요: {e.reason}"
            ) from e

        try:
            obj = json.loads(raw)
            candidates = obj.get("candidates") or []
            parts = candidates[0]["content"]["parts"] if candidates else []
            text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
            if not text.strip():
                raise ValueError("empty response text")
            return text
        except Exception as e:
            raise RuntimeError(f"Gemini 응답을 읽지 못했습니다: {raw[:500]}") from e

    @staticmethod
    def _salvage_objects(text: str) -> list[dict]:
        """잘리거나 깨진 JSON에서 온전한 항목만 건져냅니다.

        응답이 길이 제한으로 중간에 끊기면 전체 json.loads는 실패하지만,
        끊기기 전까지의 항목들은 멀쩡합니다. 곡 전체를 버리는 대신 살립니다.
        """
        found: list[dict] = []
        decoder = json.JSONDecoder()
        i = 0
        n = len(text)
        while i < n:
            start = text.find("{", i)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(text, start)
            except Exception:
                i = start + 1
                continue
            if isinstance(obj, dict):
                if "id" in obj:
                    found.append(obj)
                # 최상위 {"translations":[...]} 형태면 안쪽을 펼칩니다.
                inner = obj.get("translations")
                if isinstance(inner, list):
                    found.extend(x for x in inner if isinstance(x, dict) and "id" in x)
            i = max(end, start + 1)
        return found

    @staticmethod
    def _extract_json(text: str) -> object:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            # 마지막 수단: 온전한 항목만 회수합니다.
            salvaged = GeminiNaturalTranslator._salvage_objects(cleaned)
            if salvaged:
                return {"translations": salvaged, "_salvaged": True}
            raise

    @classmethod
    def _parse_translations(cls, text: str, expected_count: int) -> list[str]:
        obj = cls._extract_json(text)
        if isinstance(obj, dict):
            values = obj.get("translations")
        else:
            values = obj

        if not isinstance(values, list):
            raise ValueError("translations 배열이 없습니다.")

        # Preferred format: [{"id":1,"text":"..."}, ...]
        if values and all(isinstance(x, dict) for x in values):
            mapping: dict[int, str] = {}
            for x in values:
                try:
                    idx = int(x.get("id"))
                except Exception:
                    continue
                value = str(x.get("text", "")).strip()
                if value:
                    mapping[idx] = value
            if all(i in mapping for i in range(1, expected_count + 1)):
                return [mapping[i] for i in range(1, expected_count + 1)]

        # Compatibility: ["line1", "line2", ...]
        if len(values) == expected_count and all(not isinstance(x, (dict, list)) for x in values):
            out = [str(x).strip() for x in values]
            if all(out):
                return out

        raise ValueError(
            f"번역 줄 수가 맞지 않습니다. 기대 {expected_count}줄 / 응답 {len(values)}개"
        )

    def analyze_song(self, texts: list[str], src: str, title: str) -> dict | None:
        """곡 전체를 먼저 읽고 화자, 상황, 말투, 반복 훅을 파악합니다.

        번역 언어와 무관하므로 곡당 한 번만 호출되고, 결과는 캐시에 남습니다.
        영어+한국어와 영어+일본어를 같이 뽑아도 이 호출은 곡당 1회입니다.
        실패해도 번역 자체는 계속 진행합니다. 어디까지나 품질 보조입니다.
        """
        if not texts:
            return None

        key = self.cache.batch_key(
            self.engine_name, self.model_name, src, "_analysis", title, texts,
            variant=self.PROMPT_VERSION,
        )
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            return cached

        try:
            raw = self._call_api(self._analysis_prompt(texts, src, title))
            obj = self._extract_json(raw)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        self.cache.set(key, obj)
        self.cache.save()
        return obj

    @classmethod
    def _multi_prompt(cls, texts: list[str], src: str, dsts: list[str], title: str) -> str:
        """분석과 여러 언어 번역을 한 번의 호출로 처리하는 프롬프트.

        예전에는 곡마다 분석 1회 + 언어별 1회씩 호출했습니다. 15곡을 두 언어로
        뽑으면 45회가 되어 무료 등급의 분당 한도에 계속 걸렸습니다. 한 번에
        처리하면 15회로 줄어듭니다. 품질도 오히려 유리한데, 모델이 두 언어를
        같은 해석 위에서 옮기기 때문입니다.
        """
        src_name = LANG_NAMES_KO.get(src, src)
        items = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
        payload = json.dumps(items, ensure_ascii=False, indent=2)

        lang_rules = []
        fields = []
        for d in dsts:
            name = LANG_NAMES_KO.get(d, d)
            soft, hard = cls.LINE_BUDGET.get(d, (24, 36))
            lang_rules.append(
                f'  - "{d}" ({name}): 한 줄 {soft}자 내외, 최대 {hard}자를 넘기지 마세요.'
            )
            fields.append(f'"{d}":"{name} 번역"')
        rules_block = "\n".join(lang_rules)
        example = '{"id":1,' + ",".join(fields) + "}"
        dst_names = ", ".join(LANG_NAMES_KO.get(d, d) for d in dsts)

        return f"""당신은 노래 가사와 영상 자막을 전문적으로 번역하는 번역가입니다.

작업: {src_name} 노래 가사를 {dst_names} 자막으로 번역합니다.
곡 제목: {title or '(제목 없음)'}

먼저 (겉으로 쓰지 말고) 가사 전체를 읽고 다음을 스스로 판단하세요.
- 화자가 누구이고 누구에게 말하는지
- 곡 전체의 상황과 장면, 정서
- 어울리는 말투 (한국어는 반말/존댓말, 일본어는 だ체/です체)
- 반복되는 후렴과 곡 전체에서 일관되게 옮겨야 할 핵심 이미지

그 판단을 바탕으로 아래 원칙에 따라 번역하세요.
1. 단어를 1:1로 옮기는 직역을 피하고, 원어민이 노래 자막에서 자연스럽게 읽을 표현으로 씁니다.
2. 원문의 의미, 장면, 감정, 인물 관계와 호칭은 보존합니다.
3. 시적 이미지는 살리되 원문에 없는 사건이나 해석을 만들지 않습니다.
4. 반복되는 후렴은 같은 의미와 말투로 일관되게 번역합니다.
5. 말투는 곡 전체에서 흔들림 없이 유지합니다. 반말과 존댓말을 섞지 마세요.
6. 대명사나 생략된 주어는 문맥을 보고 각 언어에서 자연스럽게 처리합니다.
7. 여러 언어 사이에서도 해석이 어긋나지 않게, 같은 원문은 같은 뜻으로 옮깁니다.

자막 길이 규칙 (매우 중요):
{rules_block}
8. 원문이 짧으면 번역도 짧게 씁니다. 짧은 원문을 길게 부풀리지 마세요.
9. 줄일 때는 뜻을 빼는 대신 조사, 군더더기 수식어, 당연한 주어를 생략합니다.
10. 문장 끝 마침표는 생략합니다. 물음표와 느낌표는 필요할 때만 씁니다.

형식 규칙:
11. 각 입력 id마다 정확히 하나의 항목을 반환하고, 요청된 모든 언어를 채웁니다.
12. 줄바꿈 문자를 번역문 안에 넣지 마세요. 한 줄로만 씁니다.
13. 해설, 주석, 로마자 발음은 넣지 마세요.
14. (instrumental)이나 (guitar solo)처럼 가사가 아닌 연주 구간 표시는
    번역하지 말고 원문을 그대로 옮겨 적으세요. 비워 두지 마세요.

반환 형식은 반드시 아래 JSON 하나뿐입니다.
{{"translations":[{example}]}}

입력:
{payload}
"""

    @staticmethod
    def _is_passthrough(text: str) -> bool:
        """번역할 내용이 없는 줄인지 판단합니다.

        (instrumental), (outro), ~~~ 같은 연주 구간 표시나 기호 줄이 여기 해당합니다.
        모델은 이런 줄에 빈 값을 돌려주는 게 정상인데, 예전에는 그걸 오류로 보고
        곡 전체를 실패시켰습니다.
        """
        stripped = str(text).strip()
        if not stripped:
            return True
        # 괄호로만 둘러싸인 지시문: (instrumental), (guitar solo) 등
        if re.fullmatch(r"[\(\[][^)\]]*[\)\]]", stripped):
            return True
        # 글자가 하나도 없는 줄 (기호, 숫자만)
        if not re.search(r"[A-Za-z가-힣ぁ-んァ-ヶ一-龯]", stripped):
            return True
        return False

    @classmethod
    def _parse_multi(cls, text: str, sources: list[str], dsts: list[str],
                     strict: bool = True) -> dict[str, list[str]]:
        obj = cls._extract_json(text)
        values = obj.get("translations") if isinstance(obj, dict) else obj
        if not isinstance(values, list):
            raise ValueError("translations 배열이 없습니다.")

        by_id: dict[int, dict] = {}
        for x in values:
            if not isinstance(x, dict):
                continue
            try:
                by_id[int(x.get("id"))] = x
            except Exception:
                continue

        expected = len(sources)
        # 실제로 번역이 필요한 줄 중 몇 개나 채워졌는지 봅니다.
        translatable = [i for i in range(expected) if not cls._is_passthrough(sources[i])]
        filled = sum(
            1 for i in translatable
            if by_id.get(i + 1) and any(str(by_id[i + 1].get(d, "")).strip() for d in dsts)
        )
        if strict and translatable and filled < len(translatable) * 0.6:
            # 절반 넘게 비었으면 응답 자체가 잘못된 것이므로 재시도할 가치가
            # 있습니다. 다만 마지막 시도(strict=False)에서는 건진 만큼이라도
            # 쓰는 편이 곡 전체를 버리는 것보다 낫습니다.
            raise ValueError(
                f"번역이 너무 많이 비어 있습니다 ({filled}/{len(translatable)}줄)."
            )

        out: dict[str, list[str]] = {d: [] for d in dsts}
        for i in range(expected):
            row = by_id.get(i + 1) or {}
            for d in dsts:
                value = str(row.get(d, "")).strip()
                if not value:
                    # 연주 구간 표시나 기호 줄은 원문 그대로 두는 게 맞습니다.
                    # 그 외에 빠진 줄도 원문으로 채워 자막에 구멍이 나지 않게 합니다.
                    value = str(sources[i]).strip()
                out[d].append(value)
        return out

    # 한 번의 응답에 담을 최대 가사 줄 수.
    # 잘림의 실제 원인은 길이가 아니라 thinking 토큰이었으므로(위에서 껐습니다)
    # 보통 곡은 한 번에 처리하고, 유난히 긴 곡만 나눕니다. 너무 잘게 쪼개면
    # 호출 수가 늘어 무료 등급의 분당 한도에 다시 걸립니다.
    MAX_LINES_PER_CALL = 45

    def translate_multi(
        self,
        texts: list[str],
        src: str,
        dsts: list[str],
        *,
        title: str = "",
    ) -> dict[str, list[str]]:
        """여러 대상 언어를 API 호출 한 번으로 번역합니다.

        가사가 길면 응답이 잘릴 수 있으므로 여러 덩어리로 나눠 보냅니다.
        나눌 때도 곡 제목을 함께 넘겨 문맥이 끊기지 않게 합니다.
        """
        targets = sorted({d for d in dsts if d and d != src})
        if not targets or not texts:
            return {}

        if len(texts) <= self.MAX_LINES_PER_CALL:
            return self._translate_chunk(texts, src, targets, title)

        merged: dict[str, list[str]] = {d: [] for d in targets}
        total = (len(texts) + self.MAX_LINES_PER_CALL - 1) // self.MAX_LINES_PER_CALL
        for n, start in enumerate(range(0, len(texts), self.MAX_LINES_PER_CALL), 1):
            chunk = texts[start:start + self.MAX_LINES_PER_CALL]
            self.progress(f"번역 {n}/{total} 구간 ({len(chunk)}줄): {title}")
            part = self._translate_chunk(chunk, src, targets, title)
            for d in targets:
                merged[d].extend(part.get(d, []))
        return merged

    def _translate_chunk(
        self,
        texts: list[str],
        src: str,
        targets: list[str],
        title: str,
    ) -> dict[str, list[str]]:
        key = self.cache.batch_key(
            self.engine_name, self.model_name, src, "+".join(targets), title, texts,
            variant=self.PROMPT_VERSION + "-multi",
        )
        cached = self.cache.get(key)
        if isinstance(cached, dict) and all(
            isinstance(cached.get(d), list) and len(cached[d]) == len(texts) for d in targets
        ):
            return {d: [str(x) for x in cached[d]] for d in targets}

        prompt = self._multi_prompt(texts, src, targets, title)
        last_error: Exception | None = None
        attempts = 3
        for attempt in range(attempts):
            is_last = attempt == attempts - 1
            try:
                raw = self._call_api(prompt)
                parsed = self._parse_multi(raw, texts, targets, strict=not is_last)
                if is_last:
                    self.progress(
                        f"'{title}': 응답이 불완전해 번역되지 않은 줄은 원문으로 채웁니다."
                    )
                parsed = {d: [self._tidy_line(x) for x in v] for d, v in parsed.items()}
                self.cache.set(key, parsed)
                self.cache.save()
                return parsed
            except Exception as e:
                last_error = e
                prompt += (
                    "\n\n중요: 직전 응답의 형식이 맞지 않았습니다. "
                    f"id 1부터 {len(texts)}까지 정확히 {len(texts)}개 항목을 반환하고, "
                    f"각 항목에 {', '.join(targets)} 키를 모두 채우세요."
                )

        raise RuntimeError(f"자연번역 실패: {last_error}") from last_error

    def translate_many(
        self,
        texts: list[str],
        src: str,
        dst: str,
        *,
        title: str = "",
    ) -> list[str]:
        if src == dst:
            return list(texts)
        if not texts:
            return []

        brief = self.analyze_song(texts, src, title)

        # 분석 결과가 달라지면 번역도 달라져야 하므로 캐시 키에 함께 넣습니다.
        variant = self.PROMPT_VERSION
        if brief:
            variant += "|" + hashlib.sha256(
                json.dumps(brief, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]

        key = self.cache.batch_key(
            self.engine_name, self.model_name, src, dst, title, texts, variant=variant,
        )
        cached = self.cache.get(key)
        if isinstance(cached, list) and len(cached) == len(texts):
            return [str(x) for x in cached]

        prompt = self._prompt(texts, src, dst, title, brief)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = self._call_api(prompt)
                out = self._parse_translations(raw, len(texts))
                out = [self._tidy_line(x) for x in out]
                self.cache.set(key, out)
                self.cache.save()
                return out
            except Exception as e:
                last_error = e
                # One retry with an explicit correction request appended.
                prompt += (
                    "\n\n중요: 직전 응답의 형식이 맞지 않았습니다. "
                    f"반드시 id 1부터 {len(texts)}까지 정확히 {len(texts)}개의 번역을 JSON으로 반환하세요."
                )

        raise RuntimeError(f"자연번역 실패: {last_error}") from last_error

    @staticmethod
    def _tidy_line(text: str) -> str:
        """자막 한 줄로 안전하게 다듬습니다.

        모델이 규칙을 어기고 줄바꿈을 넣거나 마침표를 붙이는 경우가 가끔
        있습니다. SRT는 줄바꿈이 곧 두 번째 자막 줄이 되므로 여기서 정리합니다.
        뜻을 바꾸는 자동 축약은 하지 않습니다.
        """
        value = str(text).replace("\r", " ").replace("\n", " ")
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"[。.]+$", "", value).strip()
        return value


def build_translator(
    engine: str,
    cache_path: Path,
    *,
    api_key: str = "",
    model_name: str = "gemini-2.5-flash",
    progress=None,
    check_cancel=None,
):
    engine = (engine or "gemini").lower().strip()
    if engine == "gemini":
        return GeminiNaturalTranslator(
            cache_path=cache_path,
            api_key=api_key,
            model_name=model_name,
            progress=progress,
            check_cancel=check_cancel,
        )
    if engine == "argos":
        return ArgosTranslator(cache_path=cache_path)
    raise ValueError(f"지원하지 않는 번역 엔진입니다: {engine}")
