import json

import pytest

import src.script_generator as sg
from src.models import Segment, StoryManifest
from src.script_generator import (
    ScriptGenerationError,
    ScriptGenerator,
    build_deterministic_manifest,
    default_visual_prompt,
    save_manifest,
)


def cfg_for(provider="openrouter", api_key="sk-test"):
    return {
        "llm": {
            "provider": provider,
            "base_url": "https://api.test/v1",
            "api_key": api_key,
            "models": ["test-model"],
            "temperature": 0.8,
            "max_tokens": 1024,
            "timeout_sec": 10,
        },
        "paths": {"scripts_dir": "workspace/scripts"},
    }


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def ok_response(manifest_dict):
    content = json.dumps(manifest_dict)
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def long_text():
    return "The hero woke in a cold sweat. The dream felt more real than memory. " * 12


# ---------------------------------------------------------------------------
# deterministic path
# ---------------------------------------------------------------------------


def test_deterministic_manifest_shape():
    manifest = build_deterministic_manifest(long_text(), title="My Title")
    assert isinstance(manifest, StoryManifest)
    assert len(manifest.segments) >= 1
    assert [seg.id for seg in manifest.segments] == list(
        range(1, len(manifest.segments) + 1)
    )
    for seg in manifest.segments:
        assert seg.text.strip()
        assert seg.tone == "neutral"
        assert "anime" in seg.visual_prompt.lower()
        assert 3.0 <= seg.est_duration_sec <= 12.0
    assert manifest.title == "My Title"
    assert manifest.meta["generated_by"] == "deterministic"
    assert "created_at" in manifest.meta


def test_deterministic_default_title_from_text():
    text = "Once upon a time in a faraway land there was a brave warrior."
    manifest = build_deterministic_manifest(text)
    assert manifest.title == " ".join(text.split()[:6])


def test_deterministic_segments_group_sentences():
    manifest = build_deterministic_manifest(long_text())
    assert len(manifest.segments) > 1
    for seg in manifest.segments:
        assert 3.0 <= seg.est_duration_sec <= 12.0
        assert "anime" in seg.visual_prompt.lower()
        assert manifest.title in seg.visual_prompt


def test_default_visual_prompt_mentions_title_and_anime():
    vp = default_visual_prompt("The hero draws his sword.", "Blade of Dawn")
    assert "anime" in vp.lower()
    assert "Blade of Dawn" in vp


# ---------------------------------------------------------------------------
# LLM gating
# ---------------------------------------------------------------------------


def test_generator_requires_llm_provider():
    generator = ScriptGenerator(cfg_for(provider="none", api_key=""))
    with pytest.raises(ScriptGenerationError):
        generator.generate_from_text("some text")


def test_generator_requires_api_key():
    generator = ScriptGenerator(cfg_for(provider="openrouter", api_key=""))
    with pytest.raises(ScriptGenerationError):
        generator.generate_from_text("some text")


# ---------------------------------------------------------------------------
# LLM path (stubbed network)
# ---------------------------------------------------------------------------


def test_llm_generation_parses_manifest(monkeypatch):
    payload = {
        "title": "Test Story",
        "premise": "A premise.",
        "segments": [
            {
                "id": 1,
                "text": "First scene narration.",
                "tone": "dramatic",
                "visual_prompt": "anime hero portrait",
                "est_duration_sec": 6.0,
            },
            {
                "id": 2,
                "text": "Second scene.",
                "tone": "neutral",
                "visual_prompt": "",
                "est_duration_sec": 2.0,
            },
        ],
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["body"] = kwargs.get("json")
        return ok_response(payload)

    monkeypatch.setattr(sg.requests, "post", fake_post)
    generator = ScriptGenerator(cfg_for(provider="openrouter", api_key="sk-test"))
    manifest = generator.generate_from_text("raw source text")

    assert manifest.title == "Test Story"
    assert len(manifest.segments) == 2
    assert manifest.segments[0].text == "First scene narration."
    assert manifest.segments[0].visual_prompt == "anime hero portrait"
    assert [seg.id for seg in manifest.segments] == [1, 2]
    assert manifest.segments[1].est_duration_sec == 3.0
    assert "anime" in manifest.segments[1].visual_prompt

    assert captured["url"] == "https://api.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["model"] == "test-model"


def test_llm_rejects_empty_segments(monkeypatch):
    payload = {"title": "T", "premise": "P", "segments": []}
    monkeypatch.setattr(sg.requests, "post", lambda *a, **k: ok_response(payload))
    generator = ScriptGenerator(cfg_for(provider="openrouter", api_key="sk-test"))
    with pytest.raises(ScriptGenerationError):
        generator.generate_from_text("x")


def test_llm_rejects_empty_segment_text(monkeypatch):
    payload = {
        "title": "T",
        "premise": "P",
        "segments": [{"id": 1, "text": "   ", "est_duration_sec": 6.0}],
    }
    monkeypatch.setattr(sg.requests, "post", lambda *a, **k: ok_response(payload))
    generator = ScriptGenerator(cfg_for(provider="openrouter", api_key="sk-test"))
    with pytest.raises(ScriptGenerationError):
        generator.generate_from_text("x")


def test_retries_on_5xx_then_succeeds(monkeypatch):
    payload = {
        "title": "T",
        "premise": "P",
        "segments": [{"id": 1, "text": "ok", "est_duration_sec": 6.0}],
    }
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse(502, {})
        return ok_response(payload)

    monkeypatch.setattr(sg.requests, "post", fake_post)
    cfg = cfg_for(provider="openrouter", api_key="sk-test")
    cfg["llm"]["retry_delays"] = [0.0, 0.0]
    manifest = ScriptGenerator(cfg).generate_from_text("x")
    assert calls["n"] == 3
    assert manifest.title == "T"


def test_tries_models_in_order(monkeypatch):
    used = []

    def fake_post(url, **kwargs):
        used.append(kwargs["json"]["model"])
        return FakeResponse(400, {})

    monkeypatch.setattr(sg.requests, "post", fake_post)
    cfg = cfg_for(provider="openrouter", api_key="sk-test")
    cfg["llm"]["models"] = ["model-a", "model-b"]
    cfg["llm"]["retry_delays"] = [0.0, 0.0]
    with pytest.raises(ScriptGenerationError):
        ScriptGenerator(cfg).generate_from_text("x")
    assert used == ["model-a", "model-b"]


def test_generate_from_prompt_sends_dramatization(monkeypatch):
    payload = {
        "title": "T",
        "premise": "P",
        "segments": [{"id": 1, "text": "s", "est_duration_sec": 6.0}],
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return ok_response(payload)

    monkeypatch.setattr(sg.requests, "post", fake_post)
    generator = ScriptGenerator(cfg_for(provider="openrouter", api_key="sk-test"))
    manifest = generator.generate_from_prompt("A hero awakens.")
    assert manifest.title == "T"
    user_message = captured["messages"][1]["content"]
    assert "A hero awakens." in user_message


def test_system_prompt_mentions_schema(monkeypatch):
    payload = {
        "title": "T",
        "premise": "P",
        "segments": [{"id": 1, "text": "s", "est_duration_sec": 6.0}],
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured["system"] = kwargs["json"]["messages"][0]["content"]
        return ok_response(payload)

    monkeypatch.setattr(sg.requests, "post", fake_post)
    ScriptGenerator(cfg_for(provider="openrouter", api_key="sk-test")).generate_from_text(
        "x"
    )
    for token in ('"segments"', "visual_prompt", "est_duration_sec", "first-person"):
        assert token in captured["system"]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_save_manifest_roundtrip(tmp_path):
    manifest = build_deterministic_manifest(
        "Hello world. This is a test. Another sentence here.", title="Test"
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == manifest.title
    assert len(data["segments"]) == len(manifest.segments)
    assert data["segments"][0]["visual_prompt"] == manifest.segments[0].visual_prompt
