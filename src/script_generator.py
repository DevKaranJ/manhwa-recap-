from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

import requests
from pydantic import BaseModel

from src.logging_setup import get_logger
from src.models import Segment, StoryManifest

log = get_logger("nove.script")


class ScriptGenerationError(RuntimeError):
    pass


class _RetryableError(RuntimeError):
    pass


class _SegmentResponse(BaseModel):
    id: int
    text: str
    tone: str = "neutral"
    visual_prompt: str = ""
    est_duration_sec: float = 6.0


class _ManifestResponse(BaseModel):
    title: str
    premise: str = ""
    segments: list[_SegmentResponse] = []


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_SYSTEM_PROMPT = (
    "You write immersive first-person narration for an anime-style recap video. "
    'Return ONLY a JSON object matching exactly this schema: {"title": str, "premise": str, '
    '"segments": [{"id": int, "text": str, "tone": str, "visual_prompt": str, '
    '"est_duration_sec": float}]}. Each segment is one scene of 40-80 words told in '
    "first-person, vividly describing what the narrator sees and feels. The visual_prompt is "
    "an English anime-style image-generation prompt describing that scene; it must contain "
    "no text, letters, or captions rendered in the image. est_duration_sec must be between "
    "5 and 8."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_visual_prompt(segment_text: str, title: str) -> str:
    text = " ".join(segment_text.split())
    return (
        f"anime-style scene illustration for '{title}': {text} — "
        "cinematic lighting, rich color, no text or captions in the image"
    )


def _group_sentences(sentences: list[str]) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words >= 40 and current_words + words > 80:
            groups.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        groups.append(" ".join(current))
    return groups


def build_deterministic_manifest(text: str, title: str | None = None) -> StoryManifest:
    text = text.strip()
    if not text:
        raise ScriptGenerationError("cannot build a script from empty text")
    title = title or " ".join(text.split()[:6])
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]
    segments: list[Segment] = []
    for group in _group_sentences(sentences):
        word_count = len(group.split())
        duration = min(max(word_count * 0.18, 3.0), 12.0)
        segments.append(
            Segment(
                id=len(segments) + 1,
                text=group,
                tone="neutral",
                visual_prompt=default_visual_prompt(group, title),
                est_duration_sec=round(duration, 2),
            )
        )
    return StoryManifest(
        title=title,
        premise=text,
        segments=segments,
        meta={"generated_by": "deterministic", "created_at": _utc_now()},
    )


def save_manifest(manifest: StoryManifest, path) -> None:
    data = {
        "title": manifest.title,
        "premise": manifest.premise,
        "segments": [
            {
                "id": seg.id,
                "text": seg.text,
                "tone": seg.tone,
                "visual_prompt": seg.visual_prompt,
                "est_duration_sec": seg.est_duration_sec,
            }
            for seg in manifest.segments
        ],
        "meta": manifest.meta,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


class ScriptGenerator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def generate_from_prompt(self, prompt: str, title: str | None = None) -> StoryManifest:
        user_message = (
            "Dramatize the following premise into an immersive first-person narration "
            "script for an anime-style recap video, following the required JSON schema:\n\n"
            + prompt
        )
        return self._generate(user_message, title)

    def generate_from_text(self, text: str, title: str | None = None) -> StoryManifest:
        return self._generate(text, title)

    def _generate(self, user_message: str, title: str | None) -> StoryManifest:
        llm = self.cfg.get("llm", {})
        provider = llm.get("provider", "none")
        api_key = llm.get("api_key", "")
        if provider not in ("openrouter", "zenmux") or not api_key:
            log.warning(
                "LLM generation unavailable (provider=%r, has_api_key=%s); "
                "falling back to deterministic generation",
                provider,
                bool(api_key),
            )
            raise ScriptGenerationError(
                f"LLM generation unavailable: provider={provider!r}, api_key is empty. "
                "Falling back to deterministic generation."
            )

        base_url = (llm.get("base_url") or "").rstrip("/")
        models = llm.get("models") or []
        temperature = llm.get("temperature", 0.8)
        max_tokens = llm.get("max_tokens", 2048)
        timeout_sec = llm.get("timeout_sec", 60)
        retry_delays = llm.get("retry_delays") or [2.0, 4.0]

        last_error = "no LLM models configured"
        for model in models:
            delay_index = 0
            while True:
                try:
                    content = self._post_chat(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_sec=timeout_sec,
                        user_message=user_message,
                    )
                    manifest = self._parse_manifest(content, title)
                    log.info(
                        "Generated script via LLM (model=%s, segments=%d)",
                        model,
                        len(manifest.segments),
                    )
                    return manifest
                except (_RetryableError, requests.RequestException) as exc:
                    last_error = str(exc)
                    if delay_index >= len(retry_delays):
                        break
                    time.sleep(retry_delays[delay_index])
                    delay_index += 1
                except Exception as exc:
                    last_error = str(exc)
                    break
        raise ScriptGenerationError(
            f"LLM script generation failed after trying all models: {last_error}"
        )

    def _post_chat(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
        user_message: str,
    ) -> str:
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
        if resp.status_code >= 500:
            raise _RetryableError(
                f"LLM request failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise ScriptGenerationError(
                f"LLM request failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_manifest(self, content: str, title: str | None) -> StoryManifest:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScriptGenerationError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ScriptGenerationError("LLM response is not a JSON object")
        response = _ManifestResponse.model_validate(payload)
        if not response.segments:
            raise ScriptGenerationError("LLM response contained no segments")

        fallback_title = response.title.strip() or (title or "")
        segments: list[Segment] = []
        for index, seg in enumerate(response.segments, start=1):
            if not seg.text.strip():
                raise ScriptGenerationError("LLM response contained an empty segment text")
            visual = seg.visual_prompt.strip() or default_visual_prompt(seg.text, fallback_title)
            duration = min(max(float(seg.est_duration_sec), 3.0), 12.0)
            segments.append(
                Segment(
                    id=index,
                    text=seg.text.strip(),
                    tone=seg.tone.strip() or "neutral",
                    visual_prompt=visual,
                    est_duration_sec=round(duration, 2),
                )
            )
        return StoryManifest(
            title=fallback_title,
            premise=response.premise,
            segments=segments,
            meta={"generated_by": "llm", "created_at": _utc_now()},
        )
