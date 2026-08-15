from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import edge_tts

from src.config import load_config
from src.logging_setup import get_logger
from src.models import SegmentTiming, StoryManifest, TimingData, WordToken

log = get_logger("nove.tts")

TICKS_PER_SECOND = 1e7


class TtsError(RuntimeError):
    pass


def word_tokens_from_boundaries(boundaries: list[dict]) -> list[WordToken]:
    tokens = []
    for boundary in boundaries:
        if boundary.get("type") == "SentenceBoundary":
            continue
        text = boundary.get("text", "")
        if not any(ch.isalnum() for ch in text):
            continue
        offset = boundary.get("offset")
        duration = boundary.get("duration")
        if not isinstance(offset, (int, float)) or not isinstance(duration, (int, float)):
            continue
        tokens.append(
            WordToken(
                word=text,
                start=offset / TICKS_PER_SECOND,
                end=(offset + duration) / TICKS_PER_SECOND,
            )
        )
    return tokens


def _estimate_duration(text: str) -> float:
    return max(1.0, min(len(text.split()) * 0.18, 12.0))


def _stream_audio(communicate):
    return communicate.stream_sync()


def _synthesize(communicate) -> tuple[bytes, list[dict]]:
    audio = bytearray()
    boundaries = []
    for chunk in _stream_audio(communicate):
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
            boundaries.append(chunk)
    return bytes(audio), boundaries


class TtsEngine:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.tts = cfg["tts"]

    async def synthesize_all(self, manifest: StoryManifest, audio_dir) -> TimingData:
        audio_dir = Path(audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        total = 0.0
        for segment in manifest.segments:
            audio_path = audio_dir / f"segment_{segment.id:03d}.mp3"
            audio, boundaries = self._render(segment.text)
            if not audio:
                log.warning(
                    "segment %d tts failed after retries; writing ~1s silence fallback",
                    segment.id,
                )
                audio = self._silence_mp3(audio_path)
                boundaries = []
            audio_path.write_bytes(audio)

            words = word_tokens_from_boundaries(boundaries)
            duration = words[-1].end if words else _estimate_duration(segment.text)
            total += duration
            log.info(
                "segment %d tts duration=%.3fs words=%d",
                segment.id,
                duration,
                len(words),
            )
            segments.append(
                SegmentTiming(
                    segment_id=segment.id,
                    audio_path=str(audio_path),
                    duration_sec=duration,
                    words=words,
                )
            )

        timing = TimingData(segments=segments, total_duration_sec=total)
        (audio_dir / "timestamps.json").write_text(
            json.dumps(asdict(timing), indent=2), encoding="utf-8"
        )
        return timing

    def _render(self, text: str) -> tuple[bytes, list[dict]]:
        for attempt in range(1, 3):
            # boundary="WordBoundary" is required: the default only emits
            # SentenceBoundary, which would leave us with no per-word timing.
            # A fresh Communicate is needed per attempt because stream()
            # rejects a second call on the same instance.
            communicate = edge_tts.Communicate(
                text,
                voice=self.tts["voice"],
                rate=self.tts["rate"],
                pitch=self.tts["pitch"],
                boundary="WordBoundary",
            )
            try:
                audio, boundaries = _synthesize(communicate)
                if audio:
                    return audio, boundaries
                log.warning("tts returned zero audio (attempt %d); retrying", attempt)
            except Exception as exc:
                log.warning("tts stream failed (attempt %d): %s", attempt, exc)
        return b"", []

    def _silence_mp3(self, path: Path, seconds: float = 1.0) -> bytes:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise TtsError("ffmpeg not found; cannot synthesize silence fallback")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=24000:cl=mono",
                    "-t",
                    f"{seconds}",
                    "-q:a",
                    "9",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TtsError(f"silence fallback failed: {exc}") from exc
        return path.read_bytes()


def synthesize_voiceover(manifest: StoryManifest, audio_dir) -> TimingData:
    return asyncio.run(TtsEngine(load_config()).synthesize_all(manifest, audio_dir))
