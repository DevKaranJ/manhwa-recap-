from __future__ import annotations

import colorsys
import hashlib
import math
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.logging_setup import get_logger

log = get_logger("nove.image")


class ImageProcessingError(RuntimeError):
    pass


def _load_font(size: int):
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _vignette(width, height, strength=110):
    radius = int(math.hypot(width, height) / 2)
    cx, cy = width / 2, height / 2
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    steps = 72
    for i in range(steps, 0, -1):
        r = radius * i / steps
        alpha = int(strength * ((steps - i) / steps) ** 1.6)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
    black = Image.new("RGB", (width, height), (0, 0, 0))
    return black, mask


def generate_placeholder(visual_prompt: str, scene_id: int, out_dir, title: str = "") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    digest = hashlib.md5(f"{title}|{visual_prompt}".encode("utf-8")).digest()
    hue_a = digest[0] / 255.0
    hue_b = (hue_a + 0.55) % 1.0
    top = colorsys.hls_to_rgb(hue_a, 0.42, 0.85)
    bottom = colorsys.hls_to_rgb(hue_b, 0.32, 0.55)

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / (height - 1)
        row = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
        draw.line([(0, y), (width, y)], fill=row)

    black, mask = _vignette(width, height)
    img = Image.composite(black, img, mask)
    draw = ImageDraw.Draw(img)

    label = (title or visual_prompt or "Recap").strip()
    title_font = _load_font(64)
    caption_font = _load_font(28)
    max_width = width - 320
    lines = _wrap_text(draw, label, title_font, max_width)
    line_h = 78
    y = (height - line_h * len(lines)) // 2 - 30
    for line in lines:
        line_w = draw.textlength(line, font=title_font)
        draw.text(((width - line_w) / 2, y), line, fill=(255, 255, 255), font=title_font)
        y += line_h

    caption = f"Scene {scene_id}"
    cap_w = draw.textlength(caption, font=caption_font)
    draw.text(((width - cap_w) / 2, height - 150), caption, fill=(235, 235, 235), font=caption_font)

    out_path = out_dir / f"scene_{scene_id:03d}_placeholder.png"
    img.save(out_path, format="PNG")
    return out_path


class ImageGenerator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._image_cfg = cfg["image"]

    def _is_cached(self, path: Path) -> bool:
        try:
            with Image.open(path) as im:
                return im.mode == "RGB" and im.size == (self._image_cfg["width"], self._image_cfg["height"])
        except Exception:
            return False

    def _build_url(self, prompt: str, scene_id: int, seed: int) -> str:
        base = self._image_cfg["base_url"].rstrip("/")
        nologo = "true" if self._image_cfg.get("nologo", True) else "false"
        return (
            f"{base}/{quote(prompt, safe='')}"
            f"?width={self._image_cfg['width']}&height={self._image_cfg['height']}"
            f"&model={self._image_cfg['model']}&nologo={nologo}&seed={seed}"
        )

    def generate_scene(self, visual_prompt: str, scene_id: int, out_dir) -> Path:
        out_dir = Path(out_dir)
        out_path = out_dir / f"scene_{scene_id:03d}.png"
        if self._is_cached(out_path):
            return out_path
        out_dir.mkdir(parents=True, exist_ok=True)

        backoffs = (3.0, 6.0)
        for attempt in range(3):
            seed = self._image_cfg["seed"] + scene_id + attempt
            url = self._build_url(visual_prompt, scene_id, seed)
            temp = None
            try:
                resp = requests.get(url, timeout=self._image_cfg["timeout_sec"], stream=True)
                if resp.status_code != 200:
                    raise ImageProcessingError(f"HTTP {resp.status_code}")
                if not resp.headers.get("Content-Type", "").startswith("image/"):
                    raise ImageProcessingError(
                        f"unexpected Content-Type: {resp.headers.get('Content-Type')!r}"
                    )
                temp = out_dir / f".scene_{scene_id:03d}_{attempt}.tmp"
                with open(temp, "wb") as fh:
                    resp.raw.decode_content = True
                    shutil.copyfileobj(resp.raw, fh)
                if temp.stat().st_size == 0:
                    raise ImageProcessingError("empty response body")
                log.info("scene %d: downloaded %d bytes", scene_id, temp.stat().st_size)
                return self.process_image(temp, out_path)
            except Exception as exc:
                # broad catch: image-provider outages must never crash the pipeline
                if temp is not None and temp.exists():
                    try:
                        temp.unlink()
                    except OSError:
                        pass
                if attempt < 2:
                    time.sleep(backoffs[attempt])
                else:
                    log.warning("scene %d: image download failed after retries: %s", scene_id, exc)

        log.warning("scene %d: falling back to placeholder image", scene_id)
        return generate_placeholder(visual_prompt, scene_id, out_dir)

    def process_image(self, src, out_path) -> Path:
        out_path = Path(out_path)
        width = self._image_cfg["width"]
        height = self._image_cfg["height"]
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGBA")
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])
                im = background.convert("RGB")
                im = ImageOps.fit(im, (width, height), Image.LANCZOS)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_path, format="PNG")
        except Exception as exc:
            raise ImageProcessingError(f"could not process image {src}: {exc}") from exc
        return out_path
