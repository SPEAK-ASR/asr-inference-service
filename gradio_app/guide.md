# Gradio Demo — Sinhala ASR Real-time

A standalone browser UI for the Sinhala Whisper + LoRA stack with VAD-based segmentation.

## Prerequisites

Python 3.9+ and a working microphone. GPU is optional but speeds up inference.

## Setup

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# 2a) CPU install (default)
pip install -r gradio_app/requirements.txt

# 2b) GPU / CUDA install — install the CUDA-enabled torch wheels FIRST,
#     then install the remaining dependencies without overwriting them.
#     Check your CUDA version with: nvidia-smi
#     Pick the right wheel URL from https://pytorch.org/get-started/locally/
#     Example for CUDA 12.1:
#     pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
#     pip install gradio transformers peft numpy silero-vad
```

## Running

```bash
python gradio_app/app.py
```

Open your browser at **http://localhost:7860**.

## Usage

1. Click **Record** to start streaming audio from your microphone.
2. Speak in Sinhala. Transcription appears automatically after each pause.
3. Use the **Task** toggle to switch between *transcribe* (Sinhala text) and *translate* (English).
4. Adjust **Silence trigger** and **VAD sensitivity** under ⚙️ Advanced Settings to tune responsiveness.
5. Click **🗑️ Clear** to reset the transcript.

## Models used

| Component | Model |
|-----------|-------|
| Base ASR  | [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) |
| LoRA adapter | [`SPEAK-ASR/whisper-si-exp-10-medium-all`](https://huggingface.co/SPEAK-ASR/whisper-si-exp-10-medium-all) |
| VAD       | [Silero VAD](https://github.com/snakers4/silero-vad) |
