from pathlib import Path

import pytest

from src.models import SegmentTiming, SubtitleCue, TimingData, WordToken
from src.subtitle_manager import build_srt, cues_from_timing, write_srt

SEG1_WORDS = [
    "The",
    "silent",
    "forest",
    "whispered",
    "secrets",
    "beneath",
    "the",
    "moonlit",
    "canopy",
    "tonight",
]
SEG2_WORDS = ["shadows", "creeping", "between", "branches"]


def _word(text, start, end):
    return WordToken(word=text, start=start, end=end)


@pytest.fixture
def two_segment_timing():
    seg1_words = []
    for i, w in enumerate(SEG1_WORDS):
        seg1_words.append(_word(w, i * 0.6, i * 0.6 + 0.6))
    seg2_words = []
    for i, w in enumerate(SEG2_WORDS):
        seg2_words.append(_word(w, i * 0.5, i * 0.5 + 0.4))
    return TimingData(
        segments=[
            SegmentTiming(
                segment_id=1, audio_path="seg1.mp3", duration_sec=6.0, words=seg1_words
            ),
            SegmentTiming(
                segment_id=2, audio_path="seg2.mp3", duration_sec=4.0, words=seg2_words
            ),
        ],
        total_duration_sec=10.0,
    )


def test_segment_one_produces_multiple_cues_within_limits(two_segment_timing):
    cues = cues_from_timing(two_segment_timing, max_chars=20)
    seg1_cues = [c for c in cues if c.start < 6.0]
    assert len(seg1_cues) >= 3
    for cue in seg1_cues:
        assert len(cue.text) <= 20
        assert cue.end - cue.start <= 3.0
    assert seg1_cues[-1].end == pytest.approx(6.0)


def test_segment_cues_never_mix_and_seg2_starts_on_boundary(two_segment_timing):
    cues = cues_from_timing(two_segment_timing, max_chars=20)
    seg2_terms = set(SEG2_WORDS)
    seg1_cues = [c for c in cues if c.start < 6.0]
    seg2_cues = [c for c in cues if c.start >= 6.0]
    assert seg2_cues, "segment 2 must produce cues"
    assert seg2_cues[0].start == pytest.approx(6.0)
    for cue in seg1_cues:
        assert not any(t.lower() in seg2_terms for t in cue.text.split())
    for cue in seg2_cues:
        assert all(t.lower() in seg2_terms for t in cue.text.split())


def test_cue_starts_are_ascending(two_segment_timing):
    cues = cues_from_timing(two_segment_timing)
    starts = [c.start for c in cues]
    assert starts == sorted(starts)
    assert len(set(starts)) >= 2


def test_words_never_split_across_cues(two_segment_timing):
    cues = cues_from_timing(two_segment_timing)
    all_words = set(SEG1_WORDS) | set(SEG2_WORDS)
    for cue in cues:
        for token in cue.text.split():
            assert token.lower() in all_words


def test_cue_refinement_restores_case_and_punctuation(two_segment_timing):
    original = {
        1: "The silent forest whispered secrets beneath the moonlit canopy tonight.",
        2: "Shadows creeping between branches.",
    }
    cues = cues_from_timing(
        two_segment_timing, max_chars=1000, max_seconds=10.0, text_by_id=original
    )
    assert len(cues) == 2
    assert cues[0].text == "The silent forest whispered secrets beneath the moonlit canopy tonight."
    assert cues[1].text == "Shadows creeping between branches."


def test_cue_refinement_fallback_capitalizes_without_match(two_segment_timing):
    cues = cues_from_timing(two_segment_timing, max_chars=1000, max_seconds=10.0)
    assert cues[0].text.startswith("The ") or cues[0].text[0].isupper()


def test_relative_word_timings_are_offset_to_global_timeline(two_segment_timing):
    cues = cues_from_timing(two_segment_timing, max_chars=1000, max_seconds=10.0)
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(6.0)
    assert cues[1].start == pytest.approx(6.0)
    assert cues[1].end == pytest.approx(10.0)


def test_no_cue_overlaps_even_when_segments_crowded():
    seg1_words = [_word("a", i * 0.5, i * 0.5 + 0.3) for i in range(3)]
    seg2_words = [_word("b", i * 0.5, i * 0.5 + 0.3) for i in range(3)]
    timing = TimingData(
        segments=[
            SegmentTiming(segment_id=1, audio_path="s1.mp3", duration_sec=1.5, words=seg1_words),
            SegmentTiming(segment_id=2, audio_path="s2.mp3", duration_sec=1.5, words=seg2_words),
        ],
        total_duration_sec=3.0,
    )
    cues = cues_from_timing(timing, max_chars=1000, max_seconds=10.0)
    for i in range(len(cues) - 1):
        assert cues[i].end <= cues[i + 1].start + 1e-9


def test_empty_words_segment_emits_placeholder_cue():
    timing = TimingData(
        segments=[SegmentTiming(segment_id=1, audio_path="s.mp3", duration_sec=2.5, words=[])],
        total_duration_sec=2.5,
    )
    cues = cues_from_timing(timing)
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(2.5)
    assert cues[0].text == "…"


def test_max_seconds_cap_closes_cue():
    words = [_word("x", i * 1.5, i * 1.5 + 0.5) for i in range(4)]
    timing = TimingData(
        segments=[SegmentTiming(segment_id=1, audio_path="s.mp3", duration_sec=5.0, words=words)],
        total_duration_sec=5.0,
    )
    cues = cues_from_timing(timing, max_chars=1000, max_seconds=2.5)
    assert len(cues) == 2
    for cue in cues:
        assert cue.end - cue.start <= 2.5


def test_build_srt_format_and_trailing_newline():
    cues = [
        SubtitleCue(start=0.0, end=0.6, text="The silent forest"),
        SubtitleCue(start=1.5, end=4.2, text="whispered secrets"),
        SubtitleCue(start=3661.5, end=3662.0, text="long video"),
        SubtitleCue(start=-0.5, end=0.2, text="clamped start"),
    ]
    srt = build_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,600\nThe silent forest\n\n")
    assert "2\n00:00:01,500 --> 00:00:04,200\nwhispered secrets" in srt
    assert "3\n01:61:01,500 --> 01:61:02,000\nlong video" in srt
    assert "4\n00:00:00,000 --> 00:00:00,200\nclamped start" in srt
    assert "\n\n" in srt
    assert srt.endswith("\n")
    assert srt.count("\n\n") == 3


def test_build_srt_rounds_milliseconds_down():
    cues = [SubtitleCue(start=1.9999, end=2.0, text="floor ms")]
    srt = build_srt(cues)
    assert "00:00:01,999 --> 00:00:02,000\nfloor ms" in srt


def test_write_srt_writes_build_srt_output(tmp_path, two_segment_timing):
    out = tmp_path / "out.srt"
    returned = write_srt(two_segment_timing, out)
    assert returned == out
    assert isinstance(returned, Path)
    assert out.read_text(encoding="utf-8") == build_srt(two_segment_timing)
