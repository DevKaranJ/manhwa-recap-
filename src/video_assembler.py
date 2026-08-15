from __future__ import annotations

import subprocess
import time
from pathlib import Path

from src.logging_setup import get_logger
from src.models import TimingData

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = get_logger("nove.video")


class RenderError(RuntimeError):
    pass


def _num(value: float) -> str:
    return f"{value:g}"


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter option (no shell here).

    Filter options split on ':' and treat '\\' as an escape, so convert to
    forward slashes and backslash-escape the drive letter colon (C:\\ -> C\\:).
    """
    s = str(path).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = s[0] + "\\:" + s[2:]
    return s


def _concat_path(path: Path) -> str:
    """Concat demuxer accepts forward slashes and resolves relative to the
    concat file's directory, so always emit absolute paths."""
    return str(path.resolve()).replace("\\", "/")


class VideoAssembler:
    def __init__(self, cfg: dict) -> None:
        self.video = cfg["video"]
        self.paths = cfg["paths"]
        self.fps = int(self.video["fps"])
        self.keep_temp = False

    def zoompan_filter(self, scene_index: int, duration_sec: float, fps: int) -> str:
        d = round(duration_sec * fps)
        if scene_index % 2 == 0:
            z = "min(1.0+0.0008*on,1.12)"
            x = "iw/2-(iw/zoom/2)"
            y = "ih/2-(ih/zoom/2)"
        else:
            z = "max(1.12-0.0008*on,1.0)"
            x = "(iw-iw/zoom)/2"
            y = "(ih-ih/zoom)/2"
        if scene_index % 3 == 2:
            x = f"(iw-iw/zoom)*(on/{d})"
            y = "(ih-ih/zoom)/2"
        return (
            f"zoompan=z='{z}':x='{x}':y='{y}':d={d}:s=1920x1080:fps={fps}"
        )

    def render(
        self,
        images: list[Path],
        timing: TimingData,
        bgm: Path | None,
        srt: Path | None,
        out_path: Path,
    ) -> Path:
        if len(images) != len(timing.segments):
            raise RenderError(
                f"image/scene count mismatch: {len(images)} images "
                f"vs {len(timing.segments)} segments"
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        v = self.video
        min_d = float(v["min_scene_sec"])
        max_d = float(v["max_scene_sec"])

        scene_clips = []
        concat_txt = out_path.parent / "_concat.txt"
        full = out_path.parent / "_scenes_full.mp4"
        success = False
        try:
            t0 = time.perf_counter()
            for i, (img, seg) in enumerate(zip(images, timing.segments)):
                # Narration drives scene length: use the real audio duration.
                # Only pad *silent* scenes up to the minimum so the viewer has
                # something to watch, and cap at the config safety ceiling.
                audio_path = Path(seg.audio_path) if seg.audio_path else None
                has_audio = audio_path is not None and audio_path.exists()
                d = seg.duration_sec
                if not has_audio:
                    d = max(d, min_d)
                d = min(d, max_d)
                temp = out_path.parent / f"_scene_{i:03d}.mp4"
                scene_clips.append(temp)
                self._render_scene(img, seg.audio_path, d, has_audio, i, temp)
            log.info(
                "Stage 1: rendered %d scene clips in %.2fs",
                len(scene_clips),
                time.perf_counter() - t0,
            )

            t1 = time.perf_counter()
            lines = "".join(
                f"file '{_concat_path(c)}'\n" for c in scene_clips
            )
            concat_txt.write_text(lines, encoding="utf-8")
            self._run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_txt), "-c", "copy", str(full),
                ]
            )
            log.info(
                "Stage 2: concatenated %d clips in %.2fs",
                len(scene_clips),
                time.perf_counter() - t1,
            )

            t2 = time.perf_counter()
            self._finalize(full, bgm, srt, timing.total_duration_sec, out_path)
            log.info(
                "Stage 3: finalized output in %.2fs",
                time.perf_counter() - t2,
            )
            success = True
        finally:
            if success and not self.keep_temp:
                for f in scene_clips + [concat_txt, full]:
                    f.unlink(missing_ok=True)
        return out_path

    def _render_scene(
        self,
        img: Path,
        audio_path: str,
        d: float,
        has_audio: bool,
        index: int,
        temp: Path,
    ) -> None:
        zoompan = self.zoompan_filter(index, d, self.fps)
        vfilter = (
            "[0:v]scale=3840:2160:force_original_aspect_ratio=increase,"
            f"crop=3840:2160,{zoompan}[v]"
        )
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-t", _num(d), "-i", str(img),
        ]
        if has_audio:
            cmd += ["-i", str(Path(audio_path))]
            vfilter += (
                ";[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a]"
            )
        cmd += ["-filter_complex", vfilter, "-map", "[v]"]
        if has_audio:
            cmd += ["-map", "[a]"]
        v = self.video
        cmd += [
            "-c:v", v["codec"],
            "-preset", v["preset"],
            "-crf", str(v["crf"]),
            "-c:a", "aac",
            "-b:a", str(v["audio_bitrate"]),
            "-pix_fmt", "yuv420p",
            "-shortest",
            # zoompan emits d frames per input frame; without audio to stop the
            # stream this would over-produce, so cap the output to the scene.
            "-t", _num(d),
            str(temp),
        ]
        self._run(cmd)

    def _finalize(
        self,
        full: Path,
        bgm: Path | None,
        srt: Path | None,
        total_duration: float,
        out: Path,
    ) -> None:
        v = self.video
        burn = bool(v.get("burn_subtitles")) and srt is not None and Path(srt).exists()
        subtitles_vf = self._subtitles_filter(srt) if burn else None

        bgm_path = Path(bgm) if bgm is not None else None
        if bgm_path is not None and bgm_path.exists():
            if self._has_audio(full):
                # Normalize narration loudness first so it is clearly audible on
                # any device, then duck the BGM against it and remix.
                fc = (
                    f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[voice];"
                    f"[1:a]volume={v['bgm_volume_db']}dB,"
                    f"atrim=0:{_num(total_duration)}[bg];"
                    f"[bg][voice]sidechaincompress=threshold=0.03:ratio=8:"
                    f"attack=100:release=400[duck];"
                    f"[voice][duck]amix=inputs=2:duration=first:"
                    f"dropout_transition=0[a]"
                )
            else:
                # No narration audio to duck against, so just mix BGM alone.
                fc = (
                    f"[1:a]volume={v['bgm_volume_db']}dB,"
                    f"atrim=0:{_num(total_duration)}[a]"
                )
            cmd = [
                "ffmpeg", "-y", "-i", str(full),
                "-stream_loop", "-1", "-i", str(bgm_path),
                "-filter_complex", fc,
                "-map", "0:v", "-map", "[a]",
            ]
            if subtitles_vf is not None:
                cmd += ["-vf", subtitles_vf]
                cmd += [
                    "-c:v", v["codec"], "-preset", v["preset"],
                    "-crf", str(v["crf"]), "-pix_fmt", "yuv420p",
                ]
            else:
                cmd += ["-c:v", "copy"]
            cmd += [
                "-c:a", "aac", "-b:a", str(v["audio_bitrate"]), "-ac", "2",
                "-shortest", "-movflags", "+faststart", str(out),
            ]
            self._run(cmd)
        elif subtitles_vf is not None:
            cmd = [
                "ffmpeg", "-y", "-i", str(full),
                "-map", "0:v", "-map", "0:a?",
                "-vf", subtitles_vf,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:v", v["codec"], "-preset", v["preset"],
                "-crf", str(v["crf"]), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", str(v["audio_bitrate"]), "-ac", "2",
                "-movflags", "+faststart", str(out),
            ]
            self._run(cmd)
        else:
            self._run(
                [
                    "ffmpeg", "-y", "-i", str(full),
                    "-c", "copy", "-movflags", "+faststart", str(out),
                ]
            )
        self._validate_output(out)

    def _validate_output(self, out: Path) -> None:
        if not out.exists() or out.stat().st_size == 0:
            raise RenderError(f"render produced no output file: {out}")
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RenderError(f"output validation failed, no readable video stream: {out}")
        fields = proc.stdout.strip().split(",")
        if len(fields) < 3 or fields[0] != "h264" or fields[1] != "1920" or fields[2] != "1080":
            raise RenderError(
                f"output validation failed, expected h264 1920x1080, got: {proc.stdout.strip()}"
            )
        log.info("output validated: %s (%s, size=%d bytes)", out, proc.stdout.strip(), out.stat().st_size)

    def _resolve_font_assets(self) -> tuple[Path | None, str]:
        font_name = str(self.video.get("font_file", "arial.ttf"))
        search_dirs = [
            PROJECT_ROOT / "assets" / "fonts",
            Path("C:/Windows/Fonts"),
            Path("/usr/share/fonts/truetype/dejavu"),
        ]
        for d in search_dirs:
            candidate = d / font_name
            if candidate.is_file():
                return candidate.parent, candidate.stem
        return None, "Arial"

    def _subtitles_filter(self, srt: Path) -> str:
        # FFmpeg 8.x parses -vf strings twice: the graph parser strips single
        # quotes but preserves backslashes inside them, then the filter's own
        # option parser splits on ':' and unescapes backslashes. A bare
        # 'C:/...' breaks because both layers treat ':' as a separator, so
        # Windows drive paths need quotes AND an escaped colon ('C\:/...').
        def val(path: Path) -> str:
            return f"'{_escape_filter_path(path)}'"

        parts = [f"subtitles={val(srt)}"]
        fontsdir, style_font = self._resolve_font_assets()
        if fontsdir is not None:
            parts.append(f"fontsdir={val(fontsdir)}")
        style = (
            f"FontName={style_font},"
            f"FontSize={self.video.get('subtitle_font_size', 24)},"
            f"Outline={self.video.get('subtitle_outline', 2)},"
            f"Shadow={self.video.get('subtitle_shadow', 1)},"
            f"MarginV={self.video.get('subtitle_margin_v', 40)},"
            f"Alignment=2"
        )
        parts.append(f"force_style='{style}'")
        return ":".join(parts)

    @staticmethod
    def _has_audio(path: Path) -> bool:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        return bool(proc.stdout.strip())

    @staticmethod
    def _run(cmd: list, timeout: float = 1800.0) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RenderError(f"ffmpeg subprocess failed: {exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-300:]
            raise RenderError(
                f"ffmpeg failed with exit code {proc.returncode}; "
                f"command: {' '.join(map(str, cmd))} ... stderr tail: {tail}"
            )
        return proc
