# Realtime ASR Backend (Sinhala-First)

FastAPI backend that streams live audio from a client over WebSocket and returns
near real-time partial and final transcripts using a Whisper stack tuned for
Sinhala (default adapter: `SPEAK-ASR/whisper-si-exp-10-medium-all`).

> Detailed design: [investigated_detail.md](investigated_detail.md)
> Live progress: [PROGRESS.md](PROGRESS.md)
> HTTP upload API details: [docs/http-transcription.md](docs/http-transcription.md)

## Quick start

```powershell
# 1) Create a venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps (CPU)
pip install -r requirements.txt

#    For CUDA (recommended on a GPU server), install torch separately first:
#    pip install --index-url https://download.pytorch.org/whl/cu121 torch
#    pip install -r requirements.txt

# 3) Set how you load the model (see "Model loading" below), then run:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open the test client in your browser:

- Test page: http://localhost:8000/client
- Health (live): http://localhost:8000/health/live
- Health (ready): http://localhost:8000/health/ready
- HTTP transcription: `POST /api/transcribe`

## Synchronous HTTP transcription

Use this when the frontend records audio first and uploads it on demand.

Request:
- `multipart/form-data`
- `audio_file` (required)
- `language` (optional; defaults to `ASR_LANGUAGE_HINT`)

Example (`curl`):

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "audio_file=@./sample.wav;type=audio/wav" \
  -F "language=si"
```

Example response:

```json
{
  "text": "....",
  "language": "si",
  "duration_ms": 482,
  "model_kind": "peft"
}
```

## Model loading

The service supports **three** ways to load a model. Choose one with
`ASR_MODEL_KIND` (or copy a template from `config/` — see the end of this section).

All settings use the `ASR_` prefix. You can set them in the shell for a single
run or put them in a `.env` file at the project root.

### 1) PEFT adapter + base model (`ASR_MODEL_KIND=peft`)

Use this when `ASR_MODEL_ID` is a Hugging Face **LoRA/PEFT adapter** repo. The
base Whisper checkpoint is read from the adapter’s `adapter_config.json`
(`base_model_name_or_path`), unless you override it.

```powershell
$env:ASR_MODEL_KIND = "peft"
$env:ASR_MODEL_ID = "SPEAK-ASR/whisper-si-exp-10-medium-all"
# Optional explicit base if auto-detection is wrong or you want a pin:
# $env:ASR_BASE_MODEL_ID = "openai/whisper-medium"
# Merge LoRA into base before inference (default true; matches typical Space setup):
# $env:ASR_MERGE_PEFT_ADAPTER = "true"
$env:ASR_DEVICE = "auto"
$env:ASR_CUDA_DTYPE = "float16"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Single merged / full checkpoint (`ASR_MODEL_KIND=merged`)

Use this when `ASR_MODEL_ID` is **one** Hugging Face repo with full weights
(merged fine-tune or native full model), not a LoRA-only adapter.

```powershell
$env:ASR_MODEL_KIND = "merged"
$env:ASR_MODEL_ID = "your-org/your-merged-whisper-medium-si"
$env:ASR_DEVICE = "auto"
$env:ASR_CUDA_DTYPE = "float16"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Faster Whisper / CTranslate2 (`ASR_MODEL_KIND=faster_whisper`)

Use this when `ASR_MODEL_ID` is a **CTranslate2** export (e.g. `model.bin` on
Hugging Face).

```powershell
$env:ASR_MODEL_KIND = "faster_whisper"
$env:ASR_MODEL_ID = "irudachirath/faster-whisper-medium-si-exp10-fp16"
$env:ASR_DEVICE = "auto"
$env:ASR_FASTER_WHISPER_CUDA_COMPUTE_TYPE = "float16"
$env:ASR_FASTER_WHISPER_CPU_COMPUTE_TYPE = "int8"
# Optional: beam size (1 = greedy; good for streaming latency)
# $env:ASR_FASTER_WHISPER_BEAM_SIZE = "1"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Example `.env` files

Ready-made templates (copy to `.env` or merge into yours):

| Mode | Template |
|------|-----------|
| PEFT adapter | [config/transformers-adapter.env.example](config/transformers-adapter.env.example) |
| Merged / full HF | [config/transformers-merged-full.env.example](config/transformers-merged-full.env.example) |
| faster-whisper | [config/faster-whisper-ct2.env.example](config/faster-whisper-ct2.env.example) |

### Legacy env vars

If you still use `ASR_BACKEND` and `ASR_TRANSFORMERS_LOAD_MODE` without
`ASR_MODEL_KIND`, the app maps them to the new `model_kind` the same way as
before (`faster_whisper` vs `transformers` + `full` → merged).

## Project layout

```
app/
  main.py                 # FastAPI entrypoint
  api/ws_transcribe.py    # WebSocket gateway
  core/{config,logging}.py
  asr/{model_loader,streaming_engine,decoder,vad}.py
  sessions/{manager,schemas}.py
config/
  *.env.example           # Copy to project root `.env` if you like
tests/
  manual/client.html      # Mic + WebSocket test client
```

## Phase status

See [PROGRESS.md](PROGRESS.md). Phase 1 (MVP) is being implemented now.
