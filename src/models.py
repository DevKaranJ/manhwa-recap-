from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    id: int
    text: str
    tone: str = "neutral"
    visual_prompt: str = ""
    est_duration_sec: float = 6.0


@dataclass
class StoryManifest:
    title: str
    premise: str
    segments: list[Segment] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class WordToken:
    word: str
    start: float
    end: float


@dataclass
class SegmentTiming:
    segment_id: int
    audio_path: str
    duration_sec: float
    words: list[WordToken] = field(default_factory=list)


@dataclass
class TimingData:
    segments: list[SegmentTiming] = field(default_factory=list)
    total_duration_sec: float = 0.0


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
