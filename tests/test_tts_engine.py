import json
from pathlib import Path

import pytest

import src.tts_engine as te
from src.models import Segment, StoryManifest

AUDIO = b"\xff\xe3\x18\xc4" * 16
AUDIO_CHUNK = {"type": "audio", "data": AUDIO}

GOOD_CHUNKS = [
    AUDIO_CHUNK,
    {"type": "WordBoundary", "offset": 0, "duration": 20_000_000, "text": "Hello"},
    AUDIO_CHUNK,
    {"type": "WordBoundary", "offset": 20_000_000, "duration": 10_000_000, "text": "world?"},
    {"type": "SentenceBoundary", "offset": 30_000_000, "duration": 5_000_000, "text": "Hello world."},
]

NO_AUDIO_CHUNKS = [
    {"type": "SentenceBoundary", "offset": 0, "duration": 10_000_000, "text": "no audio"},
]


class FakeCommunicate:
    _count = 0
    fail_first = False
    fail_all = False
    chunks = GOOD_CHUNKS
    instance_chunks = {}

    def __init__(self, *args, **kwargs):
        self._no = FakeCommunicate._count
        FakeCommunicate._count += 1

    def stream_sync(self):
        if self._no == 0 and FakeCommunicate.fail_first:
            raise ConnectionError("simulated tts failure")
        if FakeCommunicate.fail_all:
            raise ConnectionError("simulated tts failure")
        if self._no in FakeCommunicate.instance_chunks:
            yield from FakeCommunicate.instance_chunks[self._no]
        else:
            yield from FakeCommunicate.chunks


@pytest.fixture
def fake_communicate(monkeypatch):
    FakeCommunicate._count = 0
    FakeCommunicate.fail_first = False
    FakeCommunicate.fail_all = False
    FakeCommunicate.chunks = GOOD_CHUNKS
    FakeCommunicate.instance_chunks = {}
    monkeypatch.setattr(te.edge_tts, "Communicate", FakeCommunicate)


def _cfg():
    return {"tts": {"voice": "en-US-TestNeural", "rate": "+0%", "pitch": "-2Hz"}}


def _manifest():
    return StoryManifest(
        title="t",
        premise="p",
        segments=[
            Segment(id=1, text="Hello world."),
            Segment(id=2, text="Second line here."),
        ],
    )


def _single_segment_manifest():
    return StoryManifest(title="t", premise="p", segments=[Segment(id=1, text="Hello world.")])


def test_word_tokens_from_boundaries_converts_ticks_to_seconds():
    tokens = te.word_tokens_from_boundaries(
        [
            {"type": "WordBoundary", "offset": 0, "duration": 20_000_000, "text": "Hello"},
            {"type": "WordBoundary", "offset": 20_000_000, "duration": 10_000_000, "text": "world?"},
        ]
    )
    assert tokens[0].word == "Hello"
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[0].end == pytest.approx(2.0)
    assert tokens[1].word == "world?"
    assert tokens[1].start == pytest.approx(2.0)
    assert tokens[1].end == pytest.approx(3.0)


def test_word_tokens_drop_punctuation_only_and_sentence_boundary():
    tokens = te.word_tokens_from_boundaries(
        [
            {"type": "WordBoundary", "offset": 0, "duration": 20_000_000, "text": "Hello"},
            {"type": "WordBoundary", "offset": 20_000_000, "duration": 10_000_000, "text": "?"},
            {"type": "SentenceBoundary", "offset": 30_000_000, "duration": 5_000_000, "text": "Hello world."},
        ]
    )
    assert [t.word for t in tokens] == ["Hello"]
    assert len(tokens) == 1
    assert tokens[0].end == pytest.approx(2.0)


def test_word_tokens_skip_non_boundary_chunks():
    tokens = te.word_tokens_from_boundaries(
        [
            {"type": "audio", "data": b"x"},
            {"type": "WordBoundary", "duration": 20_000_000, "text": "missing-offset"},
        ]
    )
    assert tokens == []


def test_estimate_duration_20_words():
    assert te._estimate_duration(" ".join(["word"] * 20)) == pytest.approx(6.0)


def test_estimate_duration_clamped():
    assert te._estimate_duration("one") == pytest.approx(1.0)
    assert te._estimate_duration(" ".join(["word"] * 500)) == pytest.approx(12.0)
    assert te._estimate_duration("") == pytest.approx(1.0)


def test_synthesize_all_builds_timing_and_writes_files(fake_communicate, tmp_path):
    engine = te.TtsEngine(_cfg())
    timing = te.asyncio.run(engine.synthesize_all(_manifest(), tmp_path))

    assert len(timing.segments) == 2
    seg1 = timing.segments[0]
    assert seg1.segment_id == 1
    assert [w.word for w in seg1.words] == ["Hello", "world?"]
    assert seg1.duration_sec == pytest.approx(3.0)
    assert seg1.audio_path == str(tmp_path / "segment_001.mp3")

    audio = (tmp_path / "segment_001.mp3").read_bytes()
    assert audio == AUDIO * 2

    assert timing.total_duration_sec == pytest.approx(
        sum(s.duration_sec for s in timing.segments)
    )

    ts_path = tmp_path / "timestamps.json"
    assert ts_path.exists()
    data = json.loads(ts_path.read_text(encoding="utf-8"))
    assert len(data["segments"]) == 2
    assert data["total_duration_sec"] == pytest.approx(6.0)
    assert data["segments"][0]["words"][0]["start"] == 0.0


def test_synthesize_all_retries_after_connection_error(fake_communicate, tmp_path):
    FakeCommunicate.fail_first = True
    engine = te.TtsEngine(_cfg())
    timing = te.asyncio.run(engine.synthesize_all(_single_segment_manifest(), tmp_path))

    assert FakeCommunicate._count == 2
    assert len(timing.segments) == 1
    assert (tmp_path / "segment_001.mp3").exists()


def test_synthesize_all_retries_after_zero_audio(fake_communicate, tmp_path):
    FakeCommunicate.instance_chunks = {0: NO_AUDIO_CHUNKS}
    engine = te.TtsEngine(_cfg())
    timing = te.asyncio.run(engine.synthesize_all(_single_segment_manifest(), tmp_path))

    assert FakeCommunicate._count == 2
    assert len(timing.segments) == 1
    assert timing.segments[0].duration_sec == pytest.approx(3.0)


def test_synthesize_all_writes_silence_after_two_failures(fake_communicate, tmp_path, monkeypatch, caplog):
    FakeCommunicate.fail_all = True
    monkeypatch.setattr(te.TtsEngine, "_silence_mp3", lambda self: b"SILENCE")
    engine = te.TtsEngine(_cfg())
    timing = te.asyncio.run(engine.synthesize_all(_manifest(), tmp_path))

    assert len(timing.segments) == 2
    assert (tmp_path / "segment_001.mp3").read_bytes() == b"SILENCE"
    assert timing.segments[0].duration_sec == pytest.approx(
        te._estimate_duration("Hello world.")
    )
    assert any("fallback" in r.message or "failed" in r.message for r in caplog.records)
    assert (tmp_path / "timestamps.json").exists()


def test_synthesize_all_raises_tts_error_when_silence_unavailable(fake_communicate, tmp_path):
    FakeCommunicate.fail_all = True
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(te.TtsEngine, "_silence_mp3", lambda self: (_ for _ in ()).throw(te.TtsError("no silence")))
    engine = te.TtsEngine(_cfg())
    with pytest.raises(te.TtsError):
        te.asyncio.run(engine.synthesize_all(_manifest(), tmp_path))
    monkeypatch.undo()


def test_synthesize_voiceover_wrapper(fake_communicate, tmp_path, monkeypatch):
    monkeypatch.setattr(te, "load_config", lambda: _cfg())
    timing = te.synthesize_voiceover(_manifest(), tmp_path)

    assert isinstance(timing, te.TimingData)
    assert len(timing.segments) == 2
    assert timing.segments[0].duration_sec == pytest.approx(3.0)
