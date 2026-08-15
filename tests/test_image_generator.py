import time
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import pytest
import requests
from PIL import Image

from src.image_generator import ImageGenerator, generate_placeholder

IMG_CFG = {
    "provider": "pollinations",
    "base_url": "https://image.pollinations.ai/prompt",
    "width": 1920,
    "height": 1080,
    "model": "flux",
    "nologo": True,
    "seed": 42,
    "timeout_sec": 180,
}


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


class _ImageResponse:
    status_code = 200
    headers = {"Content-Type": "image/png"}
    raw = BytesIO(_png_bytes())

    def __init__(self):
        self.raw = BytesIO(_png_bytes())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def cfg(tmp_path):
    return {"image": dict(IMG_CFG), "paths": {"images_dir": str(tmp_path)}}


@pytest.fixture
def gen(cfg):
    return ImageGenerator(cfg)


@pytest.mark.parametrize(
    "size",
    [(800, 600), (2000, 800), (600, 1000)],
    ids=["small-landscape", "wide-landscape", "portrait"],
)
def test_process_image_normalizes_to_1920x1080_rgb(tmp_path, gen, size):
    src = tmp_path / "in.png"
    Image.new("RGB", size, (120, 60, 200)).save(src)

    out = gen.process_image(src, tmp_path / "out.png")

    assert out == tmp_path / "out.png"
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (1920, 1080)
        assert im.mode == "RGB"
        assert im.format == "PNG"


def test_process_image_flattens_alpha_to_white_background(tmp_path, gen):
    src = tmp_path / "rgba.png"
    img = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    for x in range(200, 400):
        for y in range(150, 300):
            img.putpixel((x, y), (255, 0, 0, 255))
    img.save(src)

    out = gen.process_image(src, tmp_path / "flat.png")

    with Image.open(out) as im:
        assert im.mode == "RGB"
        assert "A" not in im.getbands()
        assert im.getpixel((0, 0)) == (255, 255, 255)
        assert im.getpixel((720, 300)) == (255, 0, 0)


def test_generate_placeholder_returns_valid_1920x1080_rgb(tmp_path):
    out = generate_placeholder("anime style, moonlit forest, drifting fireflies", 3, tmp_path, title="Test Title")

    assert isinstance(out, Path)
    assert out.exists()
    assert out.name == "scene_003_placeholder.png"
    with Image.open(out) as im:
        assert im.size == (1920, 1080)
        assert im.mode == "RGB"
        assert im.format == "PNG"


def test_generate_scene_cache_hit_does_not_touch_network(tmp_path, cfg, monkeypatch):
    gen = ImageGenerator(cfg)
    out_dir = tmp_path / "imgs"
    out_dir.mkdir()
    cached = out_dir / "scene_001.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(cached)

    def fail(*args, **kwargs):
        raise AssertionError("network should not be contacted when cache is valid")

    monkeypatch.setattr("src.image_generator.requests.get", fail)

    result = gen.generate_scene("any prompt that would hit the network", 1, out_dir)

    assert result == cached
    assert result.exists()


def test_generate_scene_falls_back_to_placeholder_on_network_failure(tmp_path, cfg, monkeypatch):
    gen = ImageGenerator(cfg)
    out_dir = tmp_path / "imgs"

    def conn_error(*args, **kwargs):
        raise requests.ConnectionError("provider unreachable")

    monkeypatch.setattr("src.image_generator.requests.get", conn_error)
    monkeypatch.setattr("src.image_generator.time.sleep", lambda s: None)

    result = gen.generate_scene("anime style, stormy sea at night", 2, out_dir)

    assert result.exists()
    assert result.name == "scene_002_placeholder.png"
    with Image.open(result) as im:
        assert im.size == (1920, 1080)
        assert im.mode == "RGB"


def test_generate_scene_tries_secondary_pollinations_model(tmp_path, monkeypatch):
    cfg = {"image": dict(IMG_CFG, fallback_models=["turbo"]), "paths": {"images_dir": str(tmp_path)}}
    gen = ImageGenerator(cfg)
    out_dir = tmp_path / "imgs"

    seen = []

    def fake_get(url, timeout=None, stream=None):
        seen.append(url)
        if "model=flux" in url:
            raise requests.ConnectionError("primary model down")
        return _ImageResponse()

    monkeypatch.setattr("src.image_generator.requests.get", fake_get)
    monkeypatch.setattr("src.image_generator.time.sleep", lambda s: None)

    result = gen.generate_scene("anime style, neon city", 1, out_dir)

    assert result.name == "scene_001.png"
    assert result.exists()
    assert any("model=flux" in u for u in seen)
    assert any("model=turbo" in u for u in seen)
    with Image.open(result) as im:
        assert im.size == (1920, 1080)


def test_generate_scene_uses_huggingface_fallback_when_key_present(tmp_path, monkeypatch):
    cfg = {
        "image": dict(
            IMG_CFG,
            hf_api_key="hf-test-key",
            hf_base_url="https://api.test",
            hf_model="org/model",
        ),
        "paths": {"images_dir": str(tmp_path)},
    }
    gen = ImageGenerator(cfg)
    out_dir = tmp_path / "imgs"

    def conn_error(*args, **kwargs):
        raise requests.ConnectionError("pollinations down")

    monkeypatch.setattr("src.image_generator.requests.get", conn_error)
    monkeypatch.setattr("src.image_generator.time.sleep", lambda s: None)

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _ImageResponse()

    monkeypatch.setattr("src.image_generator.requests.post", fake_post)

    result = gen.generate_scene("anime style, cherry blossom rain", 3, out_dir)

    assert result.name == "scene_003.png"
    assert result.exists()
    assert captured["url"] == "https://api.test/models/org/model"
    assert captured["headers"]["Authorization"] == "Bearer hf-test-key"
    assert "inputs" in captured["json"]
    with Image.open(result) as im:
        assert im.size == (1920, 1080)


def test_generate_scene_skips_hf_when_no_key(tmp_path, monkeypatch):
    gen = ImageGenerator({"image": dict(IMG_CFG), "paths": {"images_dir": str(tmp_path)}})
    out_dir = tmp_path / "imgs"

    def conn_error(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr("src.image_generator.requests.get", conn_error)
    monkeypatch.setattr("src.image_generator.time.sleep", lambda s: None)

    def boom(*args, **kwargs):
        raise AssertionError("HF must not be contacted without a key")

    monkeypatch.setattr("src.image_generator.requests.post", boom)

    result = gen.generate_scene("anime style, desert", 4, out_dir)
    assert result.name == "scene_004_placeholder.png"


def test_style_prefix_and_character_applied_to_every_prompt(tmp_path, monkeypatch):
    cfg = {
        "image": dict(
            IMG_CFG,
            style_prefix="modern chinese manhua art style, urban romance webtoon",
            character="beautiful pale 24yo woman with long brown hair",
        ),
        "paths": {"images_dir": str(tmp_path)},
    }
    gen = ImageGenerator(cfg)
    out_dir = tmp_path / "imgs"
    seen = []

    def fake_get(url, timeout=None, stream=None):
        seen.append(unquote(url))
        return _ImageResponse()

    monkeypatch.setattr("src.image_generator.requests.get", fake_get)
    monkeypatch.setattr("src.image_generator.time.sleep", lambda s: None)

    gen.generate_scene("kneeling father handing a doll to a child", 1, out_dir)
    gen.generate_scene("mother sitting on a sofa with tea", 2, out_dir)

    assert len(seen) == 2
    for url in seen:
        assert "modern chinese manhua art style, urban romance webtoon" in url
        assert "beautiful pale 24yo woman with long brown hair" in url
    assert "kneeling father handing a doll to a child" in seen[0]
    assert "mother sitting on a sofa with tea" in seen[1]
