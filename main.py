from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import ConfigError, load_config
from src.image_generator import ImageGenerator
from src.logging_setup import get_logger, setup_logging
from src.models import StoryManifest
from src.script_generator import ScriptGenerationError, ScriptGenerator, save_manifest
from src.subtitle_manager import write_srt
from src.tts_engine import TtsEngine
from src.video_assembler import RenderError, VideoAssembler

log = get_logger("nove.main")

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_INPUT = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Turn a story prompt or raw web-novel text into a 1080p anime "
        "audio-drama MP4 with narration, Ken Burns scenes, ducked BGM and SRT subtitles.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="high-level plot prompt to dramatize")
    source.add_argument("--file", help="path to a raw novel/story text file")
    parser.add_argument("--title", default=None, help="title override for the video")
    parser.add_argument("--output", default=None, help="output MP4 path")
    parser.add_argument("--bgm", default=None, help="background music file (.mp3/.wav)")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="cap estimated video length in minutes (truncates scenes)",
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="hardcode SRT subtitles into the video (requires libass)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="run script/audio/images/subtitles but skip ffmpeg rendering",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep intermediate scene clips after rendering",
    )
    parser.add_argument(
        "--provider",
        choices=["none", "openrouter", "zenmux"],
        default=None,
        help="override the LLM provider from config.yaml",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force deterministic generation; make no LLM API calls",
    )
    return parser.parse_args(argv)


def load_source(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.prompt:
        return args.prompt, "prompt"
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"input file not found: {path}")
        try:
            return path.read_text(encoding="utf-8"), "text"
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"could not read {path}: {exc}")
    raise SystemExit("provide either --prompt or --file")


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.offline:
        cfg["llm"]["provider"] = "none"
    elif args.provider:
        cfg["llm"]["provider"] = args.provider
        if args.provider == "zenmux":
            import os

            from dotenv import load_dotenv

            load_dotenv(Path(__file__).resolve().parent / ".env")
            cfg["llm"]["base_url"] = "https://zenmux.com/v1"
            cfg["llm"]["api_key"] = os.environ.get("ZENMUX_API_KEY", "")
        else:
            import os

            from dotenv import load_dotenv

            load_dotenv(Path(__file__).resolve().parent / ".env")
            cfg["llm"]["base_url"] = "https://openrouter.ai/api/v1"
            cfg["llm"]["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    if args.burn_subtitles:
        cfg["video"]["burn_subtitles"] = True
    return cfg


def generate_manifest(cfg: dict, source: str, mode: str, title: str | None) -> StoryManifest:
    generator = ScriptGenerator(cfg)
    try:
        if mode == "prompt":
            return generator.generate_from_prompt(source, title)
        return generator.generate_from_text(source, title)
    except ScriptGenerationError as exc:
        log.warning("LLM generation unavailable, using deterministic path: %s", exc)
        from src.script_generator import build_deterministic_manifest

        return build_deterministic_manifest(source, title)


def truncate_to_budget(manifest: StoryManifest, max_minutes: float | None) -> StoryManifest:
    if max_minutes is None:
        return manifest
    budget_sec = max_minutes * 60.0
    kept = []
    total = 0.0
    for seg in manifest.segments:
        if total + seg.est_duration_sec > budget_sec and kept:
            break
        kept.append(seg)
        total += seg.est_duration_sec
    if not kept:
        kept = manifest.segments[:1]
    log.info(
        "budget %.0fs keeps %d/%d segments (~%.0fs)",
        budget_sec,
        len(kept),
        len(manifest.segments),
        total,
    )
    manifest.segments = kept
    return manifest


def run_pipeline(args: argparse.Namespace) -> Path | None:
    cfg = load_config()
    cfg = apply_cli_overrides(cfg, args)

    source, mode = load_source(args)

    manifest = generate_manifest(cfg, source, mode, args.title)
    manifest = truncate_to_budget(manifest, args.max_minutes)
    log.info("story: %r with %d scene segments", manifest.title, len(manifest.segments))

    scripts_dir = Path(cfg["paths"]["scripts_dir"])
    manifest_path = scripts_dir / "story_manifest.json"
    save_manifest(manifest, manifest_path)
    log.info("manifest written: %s", manifest_path)

    audio_dir = Path(cfg["paths"]["audio_dir"])
    log.info("synthesizing %d segments with edge-tts voice %r ...", len(manifest.segments), cfg["tts"]["voice"])
    timing = asyncio.run(TtsEngine(cfg).synthesize_all(manifest, audio_dir))
    log.info("voiceover done: total %.1fs across %d segments", timing.total_duration_sec, len(timing.segments))

    images_dir = Path(cfg["paths"]["images_dir"])
    image_gen = ImageGenerator(cfg)
    images = []
    for seg in manifest.segments:
        path = image_gen.generate_scene(seg.visual_prompt, seg.id, images_dir)
        images.append(path)
    log.info("scene images ready: %d", len(images))

    srt_path = Path(cfg["paths"]["output_dir"]) / "subtitles.srt"
    write_srt(timing, srt_path)
    log.info("subtitles written: %s", srt_path)

    if args.skip_render:
        log.info("--skip-render: done (no video assembled)")
        return None

    output = Path(args.output) if args.output else Path(cfg["paths"]["output_dir"]) / "final_recap.mp4"
    bgm = Path(args.bgm) if args.bgm else None
    if bgm is None:
        default_bgm = Path(__file__).resolve().parent / "assets" / "bgm" / "demo_pad.mp3"
        if default_bgm.exists():
            bgm = default_bgm
            log.info("using default BGM: %s", bgm)
    if bgm is not None and not bgm.exists():
        log.warning("bgm file not found, continuing without music: %s", bgm)
        bgm = None

    assembler = VideoAssembler(cfg)
    assembler.keep_temp = args.keep_temp
    log.info("rendering video (this can take a while)...")
    result = assembler.render(images, timing, bgm, srt_path, output)
    log.info("render complete: %s", result)
    return result


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    try:
        result = run_pipeline(args)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return EXIT_FATAL
    except (RenderError, ScriptGenerationError) as exc:
        log.error("pipeline failed: %s", exc)
        return EXIT_FATAL
    except Exception:
        log.exception("unexpected pipeline failure")
        return EXIT_FATAL
    if result is not None:
        print(f"OUTPUT_VIDEO={result}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
