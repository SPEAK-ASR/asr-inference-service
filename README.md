# Realtime ASR Backend (Sinhala-First)

FastAPI backend that streams live audio from a client over WebSocket and returns
near real-time partial and final transcripts using a Hugging Face Whisper model
fine-tuned for Sinhala (`SPEAK-ASR/whisper-si-exp-10-medium-all`).

> Detailed design: [investigated_detail.md](investigated_detail.md)
> Live progress: [PROGRESS.md](PROGRESS.md)

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

# 3) Run the dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open the test client in your browser:

- Test page: http://localhost:8000/client (now includes a **Speaker diarization** toggle — flips on per-session `enable_diarization` and renders speaker chips per turn)
- Health (live): http://localhost:8000/health/live
- Health (ready): http://localhost:8000/health/ready (reports `diarization.{capability,available,loaded}`)

## Project layout

```
app/
  main.py                 # FastAPI entrypoint
  api/ws_transcribe.py    # WebSocket gateway
  core/{config,logging}.py
  asr/{model_loader,streaming_engine,decoder}.py
  sessions/{manager,schemas}.py
  workers/                # (Phase 3)
gradio_app/
  app.py                  # standalone Gradio mic demo (Whisper + optional diarization)
  requirements.txt        # deps for the Gradio app (separate from this file)
tests/
  manual/client.html      # Mic + WebSocket test client
```

### Gradio demo (optional)

Browser UI for the same Sinhala Whisper+LoRA stack with optional diarization and noise removal:

```bash
pip install -r gradio_app/requirements.txt
python gradio_app/app.py
```

Details: [gradio_app/guide.md](gradio_app/guide.md).

## Phase status

See [PROGRESS.md](PROGRESS.md). Phase 1 (MVP) is being implemented now.
