from pathlib import Path

from src.models import SegmentTiming, SubtitleCue, TimingData


def _refine_cue_text(cue_text: str, original: str) -> str:
    """Reattach the case and punctuation of the original narration to a word-only cue.

    edge-tts word boundaries strip punctuation and lower-case words, which makes
    burned subtitles look like raw ASR output. Aligning the cue back to the source
    sentence restores a polished, captioned look.
    """
    needle = " ".join(cue_text.split()).lower()
    hay = " ".join(original.split())
    start = hay.lower().find(needle)
    if start < 0:
        fallback = " ".join(cue_text.split())
        return fallback[0].upper() + fallback[1:] if fallback else fallback
    end = start + len(needle)
    if end < len(hay) and hay[end] in ".,!?…;:)]}":
        end += 1
    return hay[start:end].strip()


def cues_from_timing(
    timing: TimingData,
    max_chars: int = 42,
    max_seconds: float = 3.0,
    text_by_id: dict[int, str] | None = None,
) -> list[SubtitleCue]:
    """Build caption cues from per-segment word timings.

    edge-tts word timestamps are relative to each segment's own audio (each
    segment restarts at 0.0s), so every cue is offset by the accumulated
    `segment_start` to land on the global video timeline.
    """
    text_by_id = text_by_id or {}
    cues: list[SubtitleCue] = []
    segment_start = 0.0
    for seg in timing.segments:
        offset = segment_start
        original = text_by_id.get(seg.segment_id, "")
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
                    SubtitleCue(
                        group[0].start + offset,
                        group[-1].end + offset,
                        _refine_cue_text(" ".join(w.word for w in group), original),
                    )
                )
                group = [word]
                group_start = word.start
            else:
                group.append(word)
        if group:
            cues.append(
                SubtitleCue(
                    group[0].start + offset,
                    group[-1].end + offset,
                    _refine_cue_text(" ".join(w.word for w in group), original),
                )
            )
        cues[-1].end = max(cues[-1].end, segment_start + seg.duration_sec)
        segment_start += seg.duration_sec

    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start:
            cues[i].end = max(cues[i].start, cues[i + 1].start - 0.01)
    return cues


def build_srt(
    timing: TimingData,
    max_chars: int = 42,
    max_seconds: float = 3.0,
    text_by_id: dict[int, str] | None = None,
) -> str:
    if hasattr(timing, "segments"):
        cues = cues_from_timing(timing, max_chars, max_seconds, text_by_id)
    else:
        cues = list(timing)
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(f"{i}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}")
    return "\n\n".join(blocks) + "\n"


def write_srt(timing: TimingData, out_path, text_by_id: dict[int, str] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.write_text(build_srt(timing, text_by_id=text_by_id), encoding="utf-8")
    return out_path


def _format_timestamp(seconds: float) -> str:
    total_ms = int(max(0.0, seconds) * 1000)
    total_s = total_ms // 1000
    ms = total_ms % 1000
    hh = total_s // 3600
    mm = total_s // 60
    ss = total_s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
