from pathlib import Path

from src.models import SegmentTiming, SubtitleCue, TimingData


def cues_from_timing(
    timing: TimingData, max_chars: int = 42, max_seconds: float = 3.0
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    segment_start = 0.0
    for seg in timing.segments:
        if not seg.words:
            cues.append(SubtitleCue(segment_start, segment_start + seg.duration_sec, "…"))
            segment_start += seg.duration_sec
            continue
        group: list = []
        group_start = seg.words[0].start
        for word in seg.words:
            candidate_text = " ".join(w.word for w in [*group, word])
            would_exceed_chars = bool(group) and len(candidate_text) > max_chars
            would_exceed_time = bool(group) and word.end - group_start > max_seconds
            if would_exceed_chars or would_exceed_time:
                cues.append(
                    SubtitleCue(group[0].start, group[-1].end, " ".join(w.word for w in group))
                )
                group = [word]
                group_start = word.start
            else:
                group.append(word)
        if group:
            cues.append(
                SubtitleCue(group[0].start, group[-1].end, " ".join(w.word for w in group))
            )
        cues[-1].end = max(cues[-1].end, segment_start + seg.duration_sec)
        segment_start += seg.duration_sec
    return cues


def build_srt(
    timing: TimingData, max_chars: int = 42, max_seconds: float = 3.0
) -> str:
    if hasattr(timing, "segments"):
        cues = cues_from_timing(timing, max_chars, max_seconds)
    else:
        cues = list(timing)
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(f"{i}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}")
    return "\n\n".join(blocks) + "\n"


def write_srt(timing: TimingData, out_path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(build_srt(timing), encoding="utf-8")
    return out_path


def _format_timestamp(seconds: float) -> str:
    total_ms = int(max(0.0, seconds) * 1000)
    total_s = total_ms // 1000
    ms = total_ms % 1000
    hh = total_s // 3600
    mm = total_s // 60
    ss = total_s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
