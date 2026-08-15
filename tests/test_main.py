import pytest
from pathlib import Path

from src.models import Segment, StoryManifest

from main import (
    _chunk_text,
    _merge_manifests,
    apply_character_hints,
    apply_cli_overrides,
    clean_workspace,
    load_source,
    truncate_to_budget,
)


def test_load_source_file(tmp_path):
    f = tmp_path / "story.txt"
    f.write_text("The ashes fell like snow.", encoding="utf-8")
    source, mode = load_source(_args(file=str(f)))
    assert mode == "text"
    assert source == "The ashes fell like snow."


def test_load_source_missing_file_exits_with_input_code():
    with pytest.raises(SystemExit) as exc_info:
        load_source(_args(file="does_not_exist.txt"))
    assert exc_info.value.code == 2


def test_clean_workspace_removes_artifacts(tmp_path):
    cfg = {
        "paths": {
            "scripts_dir": str(tmp_path / "scripts"),
            "audio_dir": str(tmp_path / "audio"),
            "images_dir": str(tmp_path / "images"),
            "output_dir": str(tmp_path / "output"),
        }
    }
    for key in ("scripts_dir", "audio_dir", "images_dir", "output_dir"):
        d = Path(cfg["paths"][key])
        d.mkdir(parents=True)
        (d / "artifact.bin").write_bytes(b"x")
    clean_workspace(cfg)
    for key in ("scripts_dir", "audio_dir", "images_dir", "output_dir"):
        d = Path(cfg["paths"][key])
        assert list(d.iterdir()) == []


def test_truncate_to_budget_none_keeps_all():
    m = _manifest(4)
    assert len(truncate_to_budget(m, None).segments) == 4


def test_truncate_to_budget_caps_segments():
    m = _manifest(4)
    # each segment est 6.0s -> 4*6=24s; budget 20s keeps 3
    out = truncate_to_budget(m, 20.0 / 60.0)
    assert len(out.segments) == 3


def test_truncate_to_budget_always_keeps_one():
    m = _manifest(2)
    out = truncate_to_budget(m, 0.01)
    assert len(out.segments) == 1


def test_apply_cli_overrides_offline():
    cfg = {"llm": {"provider": "openrouter", "api_key": "k"}}
    out = apply_cli_overrides(cfg, _args(offline=True))
    assert out["llm"]["provider"] == "none"


def test_apply_cli_overrides_burn():
    cfg = {"video": {"burn_subtitles": False}}
    out = apply_cli_overrides(cfg, _args(burn_subtitles=True))
    assert out["video"]["burn_subtitles"] is True


def test_apply_cli_overrides_provider_uses_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    cfg = {"llm": {"provider": "openrouter", "base_url": "x", "api_key": "old"}}
    out = apply_cli_overrides(cfg, _args(provider="openrouter"))
    assert out["llm"]["api_key"] == "env-key"
    assert out["llm"]["base_url"] == "https://openrouter.ai/api/v1"


def test_apply_cli_overrides_llm_model():
    cfg = {"llm": {"models": ["model-a", "model-b"]}}
    out = apply_cli_overrides(cfg, _args(llm_model="custom/model"))
    assert out["llm"]["models"] == ["custom/model"]


def test_chunk_text_splits_long_story():
    paragraphs = [f"Paragraph {i} " + "word " * 60 for i in range(20)]
    text = "\n".join(paragraphs)
    chunks = _chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 510 for c in chunks)
    assert " ".join("\n".join(chunks).split()) == " ".join(text.split())


def test_chunk_text_short_story_single_chunk():
    assert _chunk_text("Short story here.") == ["Short story here."]


def test_merge_manifests_renumbers_and_keeps_title():
    m1 = StoryManifest(title="A", premise="p1", segments=[Segment(id=1, text="x"), Segment(id=2, text="y")])
    m2 = StoryManifest(title="B", premise="p2", segments=[Segment(id=1, text="z")])
    merged = _merge_manifests([m1, m2], title="Final")
    assert merged.title == "Final"
    assert [s.id for s in merged.segments] == [1, 2, 3]
    assert merged.meta["chunks"] == "2"


def test_apply_character_hints_appends_matching_character():
    manifest = StoryManifest(
        title="t",
        premise="p",
        segments=[
            Segment(id=1, text="Kael drew his sword", visual_prompt="knight in a courtyard")
        ],
    )
    apply_character_hints(
        manifest,
        {"Kael": "silver-haired knight", "Mira": "brown-haired woman"},
    )
    assert "Kael: silver-haired knight" in manifest.segments[0].visual_prompt
    assert "Mira" not in manifest.segments[0].visual_prompt


def _manifest(n):
    return StoryManifest(
        title="T",
        premise="P",
        segments=[
            Segment(id=i + 1, text=f"Segment {i + 1}", est_duration_sec=6.0)
            for i in range(n)
        ],
    )


def _args(**overrides):
    defaults = dict(
        prompt=None,
        file=None,
        title=None,
        output=None,
        bgm=None,
        max_minutes=None,
        burn_subtitles=False,
        skip_render=False,
        dry_run=False,
        keep_temp=False,
        clean=False,
        debug=False,
        provider=None,
        llm_model=None,
        offline=False,
    )
    defaults.update(overrides)
    return type("Args", (), defaults)()
