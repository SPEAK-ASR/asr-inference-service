# Realtime ASR Backend — Live Progress Tracker

> This file is updated continuously as the implementation progresses.
> Source-of-truth design: [investigated_detail.md](investigated_detail.md).

**Last updated:** 2026-05-05 (Phase 1 — Code complete, paused for review)
**Current phase:** Phase 1 — MVP (Done, awaiting your end-to-end verification)

---

## Phase Overview

| Phase | Title | Status |
|------:|---------------------------|----------------------------------------|
| 1 | MVP (Working Realtime Demo) | Code complete — awaiting your run |
| 2 | Quality Stabilization | Not started |
| 3 | Performance and Scale | Not started |
| 4 | Production Hardening | Not started |

Status legend: `Not started` · `In progress` · `Code complete` · `Done` · `Blocked`

---

## Phase 1 — MVP

Goal: end-to-end Sinhala live transcription over WebSocket with stable partial
and final events, driven by `SPEAK-ASR/whisper-si-exp-10-medium-all`.

### Checklist

- [x] Bootstrap project (directories, `requirements.txt`, `.gitignore`, `README.md`, `PROGRESS.md`)
- [x] `app/core/config.py` + `app/core/logging.py`
- [x] `app/sessions/schemas.py` (Pydantic discriminated unions)
- [x] `app/asr/model_loader.py` (auto CUDA/CPU)
- [x] `app/asr/streaming_engine.py` (sliding-window inference)
- [x] `app/asr/decoder.py` (stable-prefix partial logic)
- [x] `app/sessions/manager.py` (per-session state + reaper)
- [x] `app/api/ws_transcribe.py` (WebSocket gateway)
- [x] `app/main.py` (entry, health, mounts)
- [x] `tests/manual/client.html` (mic + WebSocket test page)
- [x] Static / unit-style verification (see below)
- [ ] **End-to-end mic verification in browser** *(needs you, see "How to Verify")*

### What I verified locally

- `python -m py_compile` passes for every module under `app/`.
- `pydantic` round-trips a sample `start` message through the
  `ClientMessage` discriminated union (returns `StartMsg`).
- `IncrementalDecoder.observe_hypothesis("hello")` then `("hello world")`
  emits a `DecoderOutput(is_stable=True, text="hello world")` — confirms the
  longest-common-prefix logic.
- `IncrementalDecoder.finalize("hello world.")` resets internal state
  cleanly.

### What still needs your run

First boot needs to download the base model (`openai/whisper-medium`) plus the
adapter weights from `SPEAK-ASR/whisper-si-exp-10-medium-all`.
The loader now detects PEFT adapters and composes base+adapter at startup.
Once you run the steps in **How to Verify** below, Phase 1 graduates from
`Code complete` to `Done`.

### Exit criteria

- `uvicorn app.main:app` boots, model loads, `/health/ready` returns 200.
- `http://localhost:8000/client` captures mic, streams to `/ws/transcribe`,
  renders `partial_transcript` and `final_transcript` events.
- 5-min single-session stability run with clean disconnect.

---

## Phase 2 — Quality Stabilization (planned)

- Silero VAD-based end-of-utterance detection (`app/asr/vad.py`)
- Tunable partial-stability threshold (char-delta + min-stable-window)
- Per-segment confidence logging + optional fallback decode hook
- Punctuation/casing post-processor (off by default)

## Phase 3 — Performance and Scale (planned)

- ASR worker pool with sticky session routing (`app/workers/asr_worker.py`)
- Bounded queues + `warning` events on backpressure
- Synthetic load-test driver (`tests/load_test.py`) and benchmark sweep

## Phase 4 — Production Hardening (planned)

- WS auth + per-IP rate limiting
- Prometheus metrics + OpenTelemetry traces
- Dockerfile (CUDA base) + compose
- Runbook, error-code table, rollback notes

---

## Design Decisions Log

| When | Decision | Rationale |
|------|----------|-----------|
| Phase 1 bootstrap | Auto CUDA/CPU device selection | Allow Windows dev workflow; fall back gracefully if no GPU. |
| Phase 1 bootstrap | Single-process, in-memory session state | Simplest viable architecture for MVP; multi-worker comes in Phase 3. |
| Phase 1 streaming | Whisper buffered-chunk + overlap (not native streaming) | Matches model family; section 5 of investigation calls this the practical baseline. |
| Phase 1 client | HTML+JS test client served from FastAPI at `/client` | Single-command dev loop, no separate build step. |
| Phase 1 audio | Client downsamples to 16 kHz mono PCM16, base64-encoded `audio_chunk` every ~320 ms | Matches the section 6 wire format; keeps server normalization-free. |
| Phase 1 partials | Stable-prefix algorithm with `min_partial_char_delta = 1` | Lightweight first cut; re-tuned with VAD + windowing in Phase 2. |
| Phase 1 inference | All `pipeline(...)` calls run on the default executor under an `asyncio.Lock` | Whisper isn't safe to invoke in parallel on a single CUDA context. |
| Phase 1 sessions | Idle reaper at 5 s ticks, 60 s timeout | Bounds memory under abandoned connections; tunable in `Settings`. |

---

## How to Verify (Phase 1)

### 1) Install dependencies

CPU-only (works on Windows out of the box):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU (CUDA 12.1) — install torch from PyTorch's index first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -r requirements.txt
```

### 2) Boot the server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

First boot downloads `openai/whisper-medium` plus
`SPEAK-ASR/whisper-si-exp-10-medium-all` adapter weights.
You should see structured JSON log lines `asr_model_loading` then
`asr_model_loaded` then `app_ready`.

### 3) Smoke-check health

```powershell
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

`/health/ready` should return HTTP 200 with `model_id` and `device` fields.

### 4) Open the test client

Visit **http://localhost:8000/client** in Chrome/Edge/Firefox.

- Click **Start streaming** → grants mic permission, opens WebSocket,
  starts shipping ~320 ms PCM16 chunks.
- Speak Sinhala (and a stray English word if you want); the **Live partial**
  card should update every ~500 ms with the rolling hypothesis.
- Click **End utterance** → server runs a final decode; the result is
  appended to **Finals**.
- Click **Stop** → graceful shutdown; you'll see a `session_summary` log line.

### 5) Stability run

Leave one session connected for ~5 minutes, speaking intermittently. Confirm:
- No exceptions in server logs.
- Memory stays bounded (the engine buffer is capped to
  `ASR_MAX_BUFFER_SECONDS` = 30 s by default).
- Disconnect cleans up: log shows `session_unregistered`.

When that all looks good, mark Phase 1 as **Done** and tell me to start
Phase 2.

---

## Next Steps

**Pausing here for your review (per the plan).** When you give the go-ahead,
Phase 2 picks up:

1. Add Silero VAD in `app/asr/vad.py` for silence-driven utterance
   finalization.
2. Tighten `IncrementalDecoder` with a stability time-window so partials
   stop flickering on noisy frames.
3. Surface per-segment confidence in logs; wire an optional fallback decode
   hook for low-confidence windows (Sinhala-English code-mix policy in
   section 4.3).
4. Add a no-op punctuation/casing post-processor scaffold.
