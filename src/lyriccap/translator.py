from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LANG_NAMES_KO = {
    "en": "영어",
    "ko": "한국어",
    "ja": "일본어",
}


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
    def batch_key(engine: str, model: str, src: str, dst: str, title: str, texts: list[str]) -> str:
        raw = json.dumps(
            {
                "engine": engine,
                "model": model,
                "src": src,
                "dst": dst,
                "title": title,
                "texts": texts,
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
    ):
        self.cache = TranslationCache(cache_path)
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model_name or "gemini-2.5-flash").strip()
        self.timeout = int(timeout)
        if not self.api_key:
            raise RuntimeError(
                "자연번역을 사용하려면 Gemini API 키가 필요합니다.\n"
                "프로그램의 'Gemini API 키' 칸에 키를 한 번 입력하고 저장하세요."
            )

    @staticmethod
    def _prompt(texts: list[str], src: str, dst: str, title: str) -> str:
        src_name = LANG_NAMES_KO.get(src, src)
        dst_name = LANG_NAMES_KO.get(dst, dst)
        items = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
        payload = json.dumps(items, ensure_ascii=False, indent=2)
        return f"""당신은 노래 가사와 영상 자막을 전문적으로 번역하는 번역가입니다.

작업: {src_name} 노래 가사를 {dst_name} 자막으로 번역합니다.
곡 제목: {title or '(제목 없음)'}

반드시 지킬 원칙:
1. 단어를 1:1로 옮기는 직역을 피하고, {dst_name} 원어민이 실제 노래 자막에서 자연스럽게 읽을 표현으로 번역합니다.
2. 원문의 의미, 장면, 감정, 시대감, 인물 관계와 호칭은 보존합니다.
3. 시적 이미지는 자연스럽게 살리되 원문에 없는 사건이나 해석을 새로 만들지 않습니다.
4. 반복되는 후렴/훅은 같은 의미와 말투로 일관되게 번역합니다.
5. 자막이므로 지나치게 길거나 설명조인 문장은 피하고 읽기 좋은 길이로 다듬습니다.
6. 영어의 대명사나 생략된 주어는 문맥을 보고 한국어/일본어에서 자연스럽게 처리합니다.
7. 각 입력 id마다 정확히 하나의 번역을 반환합니다. 줄을 합치거나 쪼개지 마세요.
8. 번역문만 반환하고 해설, 주석, 로마자 발음은 넣지 마세요.

반환 형식은 반드시 아래 JSON 하나뿐입니다.
{{"translations":[{{"id":1,"text":"번역문"}},{{"id":2,"text":"번역문"}}]}}

입력:
{payload}
"""

    def _call_api(self, prompt: str) -> str:
        model = urllib.parse.quote(self.model_name, safe="-._")
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
            },
        }
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
            if e.code in {401, 403}:
                raise RuntimeError(
                    "Gemini API 인증에 실패했습니다. API 키를 확인하세요."
                ) from e
            if e.code == 429:
                raise RuntimeError(
                    "Gemini API 사용량 제한에 도달했습니다. 잠시 후 다시 시도하세요."
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
    def _extract_json(text: str) -> object:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            if m:
                return json.loads(m.group(0))
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

        key = self.cache.batch_key(self.engine_name, self.model_name, src, dst, title, texts)
        cached = self.cache.get(key)
        if isinstance(cached, list) and len(cached) == len(texts):
            return [str(x) for x in cached]

        prompt = self._prompt(texts, src, dst, title)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = self._call_api(prompt)
                out = self._parse_translations(raw, len(texts))
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


def build_translator(
    engine: str,
    cache_path: Path,
    *,
    api_key: str = "",
    model_name: str = "gemini-2.5-flash",
):
    engine = (engine or "gemini").lower().strip()
    if engine == "gemini":
        return GeminiNaturalTranslator(
            cache_path=cache_path,
            api_key=api_key,
            model_name=model_name,
        )
    if engine == "argos":
        return ArgosTranslator(cache_path=cache_path)
    raise ValueError(f"지원하지 않는 번역 엔진입니다: {engine}")
