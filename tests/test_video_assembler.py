import subprocess

import pytest
from PIL import Image

from src.models import SegmentTiming, TimingData
from src.video_assembler import RenderError, VideoAssembler


def _make_cfg(tmp_path) -> dict:
    return {
        "video": {
            "fps": 24,
            "codec": "libx264",
            "preset": "veryfast",
            "crf": 23,
            "audio_bitrate": "192k",
            "bgm_volume_db": -24,
            "duck_db": 10,
            "min_scene_sec": 1,
            "max_scene_sec": 12,
            "burn_subtitles": False,
            "font_file": "arial.ttf",
        },
        "paths": {"output_dir": str(tmp_path)},
    }


def _make_image(path, color=(255, 0, 0)) -> None:
    Image.new("RGB", (64, 64), color).save(path)


class TestZoompanFilter:
    def test_even_index_zoom_in(self, tmp_path):
        f = VideoAssembler(_make_cfg(tmp_path)).zoompan_filter(0, 6.0, 24)
        assert "zoompan=" in f
        assert "min(1.0+0.0008*on,1.12)" in f
        assert "d=144" in f
        assert "s=1920x1080" in f

    def test_odd_index_zoom_out(self, tmp_path):
        f = VideoAssembler(_make_cfg(tmp_path)).zoompan_filter(1, 6.0, 24)
        assert "max(1.12-0.0008*on,1.0)" in f

    def test_third_scene_pan_variant(self, tmp_path):
        f = VideoAssembler(_make_cfg(tmp_path)).zoompan_filter(2, 6.0, 24)
        assert "(on/144)" in f
        assert "x='(iw-iw/zoom)*(on/144)'" in f


class TestRender:
    def test_render_real_ffmpeg(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        asm = VideoAssembler(cfg)

        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        _make_image(img1, (255, 0, 0))
        _make_image(img2, (0, 0, 255))

        timing = TimingData(
            segments=[
                SegmentTiming(segment_id=0, audio_path="", duration_sec=1.5, words=[]),
                SegmentTiming(segment_id=1, audio_path="", duration_sec=1.5, words=[]),
            ],
            total_duration_sec=3.0,
        )
        out = tmp_path / "out.mp4"

        result = asm.render([img1, img2], timing, bgm=None, srt=None, out_path=out)

        assert result == out
        assert out.exists() and out.stat().st_size > 0

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert probe.returncode == 0, probe.stderr
        # ffprobe CSV orders by struct field, and codec_name is the codec
        # name ("h264"), never the encoder wrapper ("libx264").
        assert probe.stdout.strip() == "h264,1920,1080"
        enc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_tags=encoder",
                "-of", "default=nw=1", str(out),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert enc.returncode == 0, enc.stderr
        assert "libx264" in enc.stdout
        aprobe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert aprobe.returncode == 0, aprobe.stderr
        assert aprobe.stdout.strip() == ""

    def test_len_mismatch_raises_before_ffmpeg(self, tmp_path, monkeypatch):
        asm = VideoAssembler(_make_cfg(tmp_path))
        timing = TimingData(
            segments=[SegmentTiming(segment_id=0, audio_path="", duration_sec=1.5, words=[])],
            total_duration_sec=1.5,
        )

        def _explode(*args, **kwargs):
            raise AssertionError("subprocess.run must not be called for mismatched lengths")

        monkeypatch.setattr(subprocess, "run", _explode)
        with pytest.raises(RenderError):
            asm.render([], timing, bgm=None, srt=None, out_path=tmp_path / "out.mp4")

    def test_render_with_burned_subtitles(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        cfg["video"]["burn_subtitles"] = True
        asm = VideoAssembler(cfg)

        img = tmp_path / "a.png"
        _make_image(img)
        srt = tmp_path / "subs.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,400\nHello world\n",
            encoding="utf-8",
        )
        timing = TimingData(
            segments=[SegmentTiming(segment_id=0, audio_path="", duration_sec=1.5, words=[])],
            total_duration_sec=1.5,
        )
        out = tmp_path / "out_subs.mp4"

        asm.render([img], timing, bgm=None, srt=srt, out_path=out)

        assert out.exists() and out.stat().st_size > 0
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == "h264,1920,1080"
        enc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_tags=encoder",
                "-of", "default=nw=1", str(out),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert enc.returncode == 0, enc.stderr
        assert "libx264" in enc.stdout
