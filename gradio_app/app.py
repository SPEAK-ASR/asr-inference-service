import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from silero_vad import load_silero_vad, get_speech_timestamps

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL            = "openai/whisper-medium"
ADAPTER_ID            = "SPEAK-ASR/whisper-si-exp-10-medium-all"
TARGET_SR             = 16000

SILENCE_TRIGGER_SEC   = 1.0   # seconds of silence after speech → trigger transcription
MAX_BUFFER_SEC        = 30.0  # safety cap: transcribe even if silence never comes
MIN_SPEECH_SEC        = 0.5   # ignore buffers shorter than this (avoids noise blips)
VAD_THRESHOLD         = 0.5   # Silero confidence threshold (0–1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load Whisper ──────────────────────────────────────────────────────────────
print(f"[1/3] Loading base model : {BASE_MODEL}")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
model     = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

print(f"[2/3] Applying LoRA adapter: {ADAPTER_ID}")
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model = model.merge_and_unload()
model = model.to(device)
model.eval()

# ── Load Silero VAD ───────────────────────────────────────────────────────────
print("[3/3] Loading Silero VAD …")
vad_model = load_silero_vad()
vad_model.eval()
print("All models ready.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_has_speech(audio_chunk: np.ndarray, vad_threshold: float = VAD_THRESHOLD) -> bool:
    """Run VAD on a single small chunk. Returns True if any speech detected."""
    if len(audio_chunk) == 0:
        return False
    tensor = torch.from_numpy(audio_chunk)
    timestamps = get_speech_timestamps(
        tensor,
        vad_model,
        sampling_rate=TARGET_SR,
        threshold=vad_threshold,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
    )
    return len(timestamps) > 0


def run_whisper(audio_buffer: np.ndarray, task: str) -> str:
    """Run Whisper on a float32 mono 16 kHz buffer and return the text."""
    inputs = processor(
        audio_buffer,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sinhala",
        task=task,
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def reset_segment(state: dict) -> dict:
    """Clear per-segment tracking but keep accumulated text."""
    state["buffer"]         = np.array([], dtype=np.float32)
    state["has_speech"]     = False
    state["silent_samples"] = 0
    return state


# ── Streaming callback ────────────────────────────────────────────────────────

def stream_transcribe(audio_chunk, state: dict, task: str,
                      silence_trigger_sec: float = SILENCE_TRIGGER_SEC,
                      vad_threshold: float = VAD_THRESHOLD):
    """
    Called by Gradio on every microphone chunk.

    state keys
    ----------
    buffer         : np.ndarray  – speech audio accumulated since last transcription
    has_speech     : bool        – have we seen speech in the current segment?
    silent_samples : int         – consecutive silent samples counted after speech
    text           : str         – full transcription so far
    """
    # ── initialise state on first call ──
    if "buffer" not in state:
        state = reset_segment(state)
        state["text"] = ""

    if audio_chunk is None:
        return state, state["text"]

    sample_rate, waveform = audio_chunk

    # ── normalise to float32 mono ──
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if np.abs(waveform).max() > 1.0:
        waveform /= np.iinfo(np.int16).max

    # ── resample to 16 kHz if needed ──
    if sample_rate != TARGET_SR:
        t = torch.from_numpy(waveform).unsqueeze(0)
        t = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SR
        )(t)
        waveform = t.squeeze(0).numpy()

    # ── VAD: is this chunk speech or silence? ──
    speech_in_chunk = chunk_has_speech(waveform, vad_threshold)

    if speech_in_chunk:
        # Active speech → accumulate and reset silence counter
        state["buffer"]         = np.concatenate([state["buffer"], waveform])
        state["has_speech"]     = True
        state["silent_samples"] = 0

    else:
        if state["has_speech"]:
            # We were speaking — count silence and keep buffering
            # (so Whisper receives natural sentence endings)
            state["silent_samples"] += len(waveform)
            state["buffer"]          = np.concatenate([state["buffer"], waveform])
        else:
            # Silence before any speech — skip entirely
            return state, state["text"] or "🎤 Listening…"

    # ── decide whether to transcribe ──
    silent_sec  = state["silent_samples"] / TARGET_SR
    buffer_sec  = len(state["buffer"])    / TARGET_SR

    silence_triggered = state["has_speech"] and silent_sec >= silence_trigger_sec
    buffer_maxed      = buffer_sec >= MAX_BUFFER_SEC

    if silence_triggered or buffer_maxed:
        buf        = state["buffer"]
        speech_sec = buffer_sec - silent_sec   # approximate voiced duration

        if speech_sec >= MIN_SPEECH_SEC:
            segment_text = run_whisper(buf, task)
            if segment_text:
                prev         = state.get("text", "")
                state["text"] = (prev + " " + segment_text).strip()

        state = reset_segment(state)   # ready for next utterance

    return state, state["text"] or "🎤 Listening…"


def clear_all():
    return {}, ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Sinhala ASR — Real-time Whisper") as demo:
    gr.Markdown(
        """
        # 🎙️ Sinhala Speech Recognition — Real-time
        **Model:** [`SPEAK-ASR/whisper-si-exp-10-medium-all`](https://huggingface.co/SPEAK-ASR/whisper-si-exp-10-medium-all)

        Press **Record** and start speaking.
        Transcription fires automatically after you pause — silence is ignored.
        """
    )

    session_state = gr.State({})

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Microphone",
                streaming=True,
            )
            task = gr.Radio(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="Task",
                info="Transcribe → Sinhala text  |  Translate → English",
            )
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.Markdown(
                    "**Silence trigger** — how long a pause before transcription fires.  \n"
                    "**VAD sensitivity** — raise if background noise keeps triggering speech."
                )
                silence_slider = gr.Slider(
                    minimum=0.5, maximum=3.0, value=SILENCE_TRIGGER_SEC,
                    step=0.1, label="Silence trigger (seconds)",
                )
                vad_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=VAD_THRESHOLD,
                    step=0.05, label="VAD sensitivity",
                )
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column():
            output = gr.Textbox(
                label="Live Transcription",
                lines=14,
                placeholder="Start speaking — text will appear here after each pause…",
            )

    audio_input.stream(
        fn=stream_transcribe,
        inputs=[audio_input, session_state, task, silence_slider, vad_slider],
        outputs=[session_state, output],
    )

    clear_btn.click(fn=clear_all, outputs=[session_state, output])

    gr.Markdown(
        """
        ---
        Built with [Whisper](https://github.com/openai/whisper) · [Silero VAD](https://github.com/snakers4/silero-vad) · [PEFT](https://github.com/huggingface/peft) · [🤗 Transformers](https://github.com/huggingface/transformers)
        """
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)