# Novel Video Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated Python CLI pipeline that turns a story prompt or raw web-novel text into a 30-60+ minute 1080p YouTube-ready MP4 audio-drama video (narration voiceover, Ken Burns anime scenes, ducked BGM, SRT subtitles) using only free-tier APIs and FFmpeg.

**Architecture:** Six orchestrated layers — script generation (LLM), TTS (edge-tts), scene images (Pollinations + Pillow), subtitles (SRT from word timestamps), video assembly (FFmpeg zoompan + sidechaincompress), and CLI orchestration (`main.py`). Every network/LLM layer has a deterministic fallback so the pipeline always produces a renderable result without API keys.

**Tech Stack:** Python 3.14 (`py` launcher), `requests`, `pydantic`, `edge-tts`, `Pillow`, `python-dotenv`, `pyyaml`, FFmpeg 8.1.2 CLI via subprocess. No GPU libraries.

## Global Constraints

- Python invoked as `py -3.14` on this machine (Windows, pwsh). FFmpeg 8.1.2 full build available at `ffmpeg`.
- **No heavy GPU libs** — no PyTorch, no CUDA, no tensorflow. Local rendering = FFmpeg CLI only.
- All external APIs are free-tier: OpenRouter/ZenMux (LLM), edge-tts (TTS), Pollinations.ai (images).
- Target resolution: 1920x1080, 24 fps, `libx264`, `-preset veryfast`, `-crf 23`, `-pix_fmt yuv420p`, `-c:a aac -b:a 192k`, `-movflags +faststart`.
- Output: `workspace/output/final_recap.mp4`. Working artifacts in `workspace/` (gitignored).
- Every module logs through a shared logger to stderr and raises typed exceptions; API failures retry 2x with backoff then fall back to deterministic generation — never a hard crash for network reasons.
- No comments in code unless they explain a non-obvious FFmpeg/timing decision.
- Per-scene video strategy: render each scene to an intermediate MP4, then `concat` demuxer with `-c copy` for the full-length video (stable on low-spec CPUs, avoids a giant filtergraph).

## Shared Contracts (definitive — all teams build against these)

`src/models.py` (dataclasses, created in Task 0):

```python
@dataclass
class Segment:
    id: int                      # 1-based
    text: str                    # narration paragraph (one image scene)
    tone: str                    # emotional tone tag, e.g. "somber"
    visual_prompt: str           # anime visual prompt for image gen
    est_duration_sec: float      # 5-8s target pacing

@dataclass
class StoryManifest:
    title: str
    premise: str
    segments: list[Segment]
    meta: dict[str, str]         # {"generated_by": "...", "created_at": "..."}

@dataclass
class WordToken:
    word: str
    start: float                 # seconds
    end: float                   # seconds

@dataclass
class SegmentTiming:
    segment_id: int
    audio_path: str
    duration_sec: float
    words: list[WordToken]       # may be empty if timestamps unavailable

@dataclass
class TimingData:
    segments: list[SegmentTiming]
    total_duration_sec: float

@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str
```

### story_manifest.json schema (Task 1 output)

```json
{
  "title": "The Last Ember",
  "premise": "one-line premise",
  "segments": [
    {"id": 1, "text": "I woke to ash on my tongue...", "tone": "somber",
     "visual_prompt": "anime style, first person view of a burned village at dawn, embers drifting, muted colors",
     "est_duration_sec": 6.0}
  ],
  "meta": {"generated_by": "deepseek/deepseek-chat", "created_at": "2026-08-15T00:00:00Z"}
}
```

### Module interfaces

- `src/config.py`: `load_config(path=None) -> dict` (merges config.yaml + .env). `resolve(...)` helpers for workspace paths.
- `src/script_generator.py`: `ScriptGenerator(cfg)`, `generate_from_prompt(prompt: str, title: str|None) -> StoryManifest`, `generate_from_text(text: str, title: str|None) -> StoryManifest`, `build_deterministic_manifest(text: str, title: str|None) -> StoryManifest`, module fn `save_manifest(manifest, path)`.
- `src/tts_engine.py`: `TtsEngine(cfg)`, async `synthesize_all(manifest: StoryManifest, audio_dir: Path) -> TimingData`. Emits `workspace/audio/segment_001.mp3`... and `workspace/audio/timestamps.json` (dump of TimingData via `dataclasses.asdict`).
- `src/image_generator.py`: `ImageGenerator(cfg)`, `generate_scene(visual_prompt: str, scene_id: int, out_dir: Path) -> Path`, `process_image(src: Path, out_path: Path) -> Path` (strict 1920x1080 RGB). Cache by sha1 of prompt; placeholder fallback image on network failure.
- `src/subtitle_manager.py`: `build_srt(timing: TimingData) -> str`, `write_srt(timing: TimingData, out_path: Path) -> Path`. Group words into cues <= ~42 chars / <= 3s, never crossing segment boundaries.
- `src/video_assembler.py`: `VideoAssembler(cfg)`, `render(images: list[Path], timing: TimingData, bgm: Path|None, srt: Path|None, out_path: Path) -> Path`.
- `main.py`: argparse CLI `--prompt | --file | --title | --output | --bgm | --max-minutes | --burn-subtitles | --keep-temp | --skip-render`.

---

## Task 0: Foundation & Scaffolding

**Files:**
- Create: `.env.example`, `requirements.txt`, `config.yaml`, `.gitignore`, `src/__init__.py`, `src/models.py`, `src/config.py`, `src/logging_setup.py`, `assets/fonts/.gitkeep`, `assets/bgm/.gitkeep`

**Interfaces:** Produces the shared dataclasses in `src/models.py` and the config schema below; everything else builds on them.

### Config schema (config.yaml)

```yaml
llm:
  provider: openrouter            # openrouter | zenmux | none
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY # or ZENMUX_API_KEY
  models: ["deepseek/deepseek-chat", "google/gemini-2.0-flash-exp:free"]
  temperature: 0.8
  max_tokens: 8192
  timeout_sec: 90
tts:
  engine: edge-tts                # edge-tts | openrouter
  voice: en-US-ChristopherNeural
  rate: +0%
  pitch: -2Hz
  openrouter_model: deepgram/flux-tts:free
image:
  provider: pollinations
  base_url: https://image.pollinations.ai/prompt
  width: 1920
  height: 1080
  model: flux
  nologo: true
  seed: 42
  timeout_sec: 180
video:
  fps: 24
  codec: libx264
  preset: veryfast
  crf: 23
  audio_bitrate: 192k
  bgm_volume_db: -24
  duck_db: 10
  min_scene_sec: 3
  max_scene_sec: 12
  burn_subtitles: false
  font_file: arial.ttf
paths:
  scripts_dir: workspace/scripts
  audio_dir: workspace/audio
  images_dir: workspace/images
  output_dir: workspace/output
```

**`.env.example`:** `OPENROUTER_API_KEY=`, `ZENMUX_API_KEY=` (empty, documented).

**`requirements.txt`:** `requests>=2.31`, `pydantic>=2.7`, `edge-tts>=6.1`, `Pillow>=10.2`, `python-dotenv>=1.0`, `pyyaml>=6.0`.

**Steps:**
- [ ] Create all scaffolding files above.
- [ ] `src/config.py`: loads config.yaml, overrides with env vars (dotenv), creates workspace dirs, validates ffmpeg presence via `ffmpeg -version` check with clear error.
- [ ] Generate `assets/bgm/demo_pad.mp3`: a 2-minute ambient drone via ffmpeg `sine` + `anoisesrc` filters (used as default BGM when `--bgm` not given).
- [ ] Verify: `py -c "from src.config import load_config; c = load_config(); print(c['video']['codec'])"` prints `libx264`.
- [ ] Commit: `feat: scaffold project foundation (config, models, logging, demo bgm)`

---

## Task 1: Team A — script_generator.py

**Files:**
- Create: `src/script_generator.py`

**Interfaces:**
- Consumes: `StoryManifest`, `Segment` from `src/models.py`; `load_config` from `src/config.py`.
- Produces: `ScriptGenerator(cfg)`, `generate_from_prompt(prompt, title=None) -> StoryManifest`, `generate_from_text(text, title=None) -> StoryManifest`, `build_deterministic_manifest(text, title=None) -> StoryManifest`, `save_manifest(manifest, path)`.

- [ ] TDD: tests for `build_deterministic_manifest` — sentence splitting into 5-8s segments, generic anime visual prompt per segment (contains "anime" + topic keyword), valid StoryManifest.
- [ ] TDD: tests for `ScriptGenerator` — when `cfg["llm"]["provider"] == "none"` or no API key, calls deterministic path. JSON-response parsing into StoryManifest via pydantic, strict segment validation (id sequential, est_duration clamped 3-12).
- [ ] Implement `ScriptGenerator`: OpenAI-compatible POST to `{base_url}/chat/completions` with `response_format: {"type": "json_object"}` prompt requesting the manifest JSON. Try models in order; retry 2x backoff; on total failure raise `ScriptGenerationError` (caught by main to fall back).
- [ ] LLM prompt (system): "You are a webnovel adaptation director. Return ONLY JSON matching this schema: {title, premise, segments: [{id, text, tone, visual_prompt, est_duration_sec}]}. Write immersive first-person narration. Each segment is one scene, 40-80 words, one visual_prompt in English for anime-style image generation (no text in image). est_duration_sec 5-8." User message = the source (prompt or raw text).
- [ ] Visual prompt helper: given a segment, `_default_visual_prompt(segment_text, title)` -> "anime style, {title}, dramatic cinematic, {first 8-10 keywords}, no text". Used in deterministic path and as fallback.
- [ ] `save_manifest(manifest, path)` writes pretty-printed JSON.
- [ ] Verify: `py -m pytest` for new tests + `py -c` smoke constructing a 3-segment manifest.
- [ ] Commit: `feat: add LLM script generator with deterministic fallback`

---

## Task 2: Team B — tts_engine.py

**Files:**
- Create: `src/tts_engine.py`

**Interfaces:**
- Consumes: `StoryManifest`, `TimingData`, `SegmentTiming`, `WordToken` from `src/models.py`.
- Produces: `TtsEngine(cfg)` with async `synthesize_all(manifest, audio_dir) -> TimingData`. Also writes `timestamps.json` (asdict of TimingData) into audio_dir.

- [ ] TDD: unit test building `TimingData` from fake word boundaries (offsets in 100ns units from edge-tts WordBoundary events) — verifies conversion math (offset/1e7 seconds).
- [ ] Implement `synthesize_all`: for each segment, run `edge_tts.Communicate(seg.text, voice=voice, rate=rate, pitch=pitch)`; collect `WordBoundary` events (offset 100ns, duration 100ns) and audio bytes; write `segment_001.mp3`. On any per-segment failure, retry once; if still failing, synthesize that segment via `Communicate` and use sentence-length estimate for duration (words empty).
- [ ] Word cleanup: strip punctuation-only tokens; skip `WordBoundary` of type SentenceBoundary.
- [ ] Concatenate nothing — each segment stays a separate MP3 (video assembler consumes one per scene).
- [ ] `timestamps.json` written with `json.dumps(asdict(timing), indent=2)`.
- [ ] Verify: `py -m pytest` + a real 1-segment synthesis smoke test (network required; if unavailable, mark test skip via env `NOVE_ENGINE_OFFLINE=1`).
- [ ] Commit: `feat: add edge-tts voiceover engine with word timestamps`

---

## Task 3: Team C — image_generator.py

**Files:**
- Create: `src/image_generator.py`

**Interfaces:**
- Consumes: `load_config`, `Path`.
- Produces: `ImageGenerator(cfg)`, `generate_scene(visual_prompt, scene_id, out_dir) -> Path`, `process_image(src, out_path) -> Path`.

- [ ] TDD: `process_image` tests — a 800x600 or portrait Pillow image must become exactly 1920x1080 RGB (center-crop + resize); RGBA gets alpha flattened.
- [ ] Implement `generate_scene`: build URL `{base_url}/{quote(prompt)}?width=1920&height=1080&model={model}&nologo=true&seed={seed+scene_id}`; `requests.get` with 180s timeout, stream to file. Retry 2x with backoff + seed increment.
- [ ] Cache: `out_dir/scene_{scene_id:03d}.png`; if exists and valid 1920x1080, return it (no re-download).
- [ ] On network failure or non-200: generate procedural placeholder with Pillow — vertical gradient from tone-derived colors, title text centered, subtle vignette — saved as `scene_00X_placeholder.png`; log warning. Never raise on image download failure.
- [ ] After every download: `process_image` normalize to 1920x1080 RGB, save as `.png`.
- [ ] Verify: pytest + offline smoke: with network blocked, still returns a valid 1920x1080 file.
- [ ] Commit: `feat: add Pollinations scene image generator with Pillow normalization`

---

## Task 4: Team D — subtitle_manager.py

**Files:**
- Create: `src/subtitle_manager.py`

**Interfaces:**
- Consumes: `TimingData`, `SubtitleCue` from `src/models.py`.
- Produces: `build_srt(timing) -> str`, `write_srt(timing, out_path) -> Path`.

- [ ] TDD: `build_srt` — words grouped into cues: max ~42 chars per line, break on word boundaries, hard cap 3.0s per cue (split long words), cues never span segment boundaries; start times strictly ascending; SRT time format `HH:MM:SS,mmm`; first cue starts >= 0.
- [ ] Implement per segment: fold segment-level duration into last cue if word data ends early (pad final cue end = segment duration).
- [ ] If a segment has no word data, create one cue spanning segment duration with first 42 chars of text.
- [ ] Verify: pytest with a synthetic TimingData (2 segments, mixed word coverage).
- [ ] Commit: `feat: add SRT subtitle builder from word timestamps`

---

## Task 5: Team E — video_assembler.py

**Files:**
- Create: `src/video_assembler.py`

**Interfaces:**
- Consumes: `TimingData` from `src/models.py`, `load_config`.
- Produces: `VideoAssembler(cfg)`, `render(images, timing, bgm, srt, out_path) -> Path`.

- [ ] TDD: helper `_zoompan_filter(i, duration, fps)` returns zoompan filter string — even scenes zoom-in (`z='min(1.0+0.0008*on,1.12)'`), odd scenes zoom-out; alternating horizontal pan (x: `'iw/2-(iw/zoom/2)'` and `'(iw-iw/zoom)'` variants), `s=1920x1080`, `fps=fps`, `d=duration*fps`.
- [ ] Implement `render`:
  1. Per scene: `ffmpeg -loop 1 -t D -i scene.png -i segment.mp3 -filter_complex "[0:v]scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,<zoompan>[v];[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a]" -map "[v]" -map "[a]" -c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 192k -shortest scene_i.mp4`.
  2. Write concat list file; `ffmpeg -f concat -safe 0 -i list.txt -c copy scenes_full.mp4`.
  3. If bgm given: `ffmpeg -i scenes_full.mp4 -stream_loop -1 -i bgm -filter_complex "[1:a]volume={bgm_volume_db}dB,atrim=0:{total}[bg];[bg][0:a]sidechaincompress=threshold=0.03:ratio=8:attack=100:release=400[duck];[duck][0:a]amix=inputs=2:duration=first:dropout_transition=0[a]" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k final.mp4`. Else just `-c copy` scenes_full to final.
  4. If `srt` and `cfg.video.burn_subtitles`: `subtitles='{srt}':fontsdir=assets/fonts:force_style='FontName=Arial,FontSize=20,Outline=1,Shadow=1'` appended to a final `-vf` pass.
  5. `-movflags +faststart`.
- [ ] Clean up intermediate `scene_i.mp4` and concat list unless `--keep-temp`.
- [ ] Verify: pytest for `_zoompan_filter`; full 2-scene 6s smoke render via ffmpeg (should take < 60s on veryfast).
- [ ] Commit: `feat: add FFmpeg video assembler with Ken Burns and BGM ducking`

---

## Task 6: Integration — main.py

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: everything above.
- Produces: CLI entrypoint.

- [ ] argparse: `--prompt`, `--file`, `--title`, `--output`, `--bgm`, `--max-minutes`, `--burn-subtitles`, `--keep-temp`, `--skip-render`, `--provider none|openrouter|zenmux` (override), `--offline` (forces deterministic path, no network).
- [ ] Pipeline: load config → generate manifest (LLM, falling back to deterministic on any `ScriptGenerationError` or provider=none) → save manifest → TTS → images → subtitles → assemble → cleanup → print final path + duration.
- [ ] `--max-minutes`: truncate manifest segments to fit estimated duration (sum est_duration_sec <= max*60).
- [ ] Logging: INFO progress lines per stage; WARNING for every fallback; ERROR + traceback for fatal; exit codes 0 success, 1 fatal, 2 input error.
- [ ] Verify: `py main.py --help`; run `py main.py --prompt "..." --offline --max-minutes 0.2 --skip-render` prints stage lines without crashing.
- [ ] Commit: `feat: wire end-to-end CLI pipeline`

---

## Task 7: QA — end-to-end verification

- [ ] `py -m pip install -r requirements.txt` from clean state.
- [ ] Unit suite green: `py -m pytest -q` (skip network tests when offline).
- [ ] Full offline demo: `py main.py --prompt "A swordsman wakes in a city of ash and remembers his name." --offline --bgm assets/bgm/demo_pad.mp3 --burn-subtitles --max-minutes 0.5 --output workspace/output/final_recap.mp4`.
- [ ] Assert output MP4 exists, `ffprobe` shows 1920x1080, h264, aac, duration ~30s, has audio stream, srt file generated.
- [ ] Commit: `docs: end-to-end verification notes` (or fix commits as needed).

---

## Self-Review Notes

- Spec coverage: script gen (T1), TTS (T2), images (T3), subtitles (T4), assembly (T5), CLI+cleanup (T6), E2E (T7). Env/config (T0). ✓
- Fallbacks keep pipeline alive without any API keys (deterministic manifest, placeholder images, no-LLM path).
- Sidechaincompress ducking + concat-demuxer approach keeps the FFmpeg command stable for 30-60 min outputs on low-spec CPUs.
