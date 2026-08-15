import time
from pathlib import Path

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
