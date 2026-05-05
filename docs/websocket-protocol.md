# WebSocket transcription contract (`/ws/transcribe`)

This document describes the **current** backend behavior so a Flutter client (or any WebSocket client) can implement streaming capture correctly.

For a **Flutter‑oriented UI and sample code**, see **[flutter-websocket-guide.md](flutter-websocket-guide.md)**.

## Transport

- **Endpoint:** `GET` upgrade to **`/ws/transcribe`** (e.g. `ws://HOST:PORT/ws/transcribe` or `wss://…`).
- **Framing:** Every message from the client and server is **a single WebSocket text frame** whose payload is **UTF‑8 JSON** (no multipart binary PCM channel in this service).
- **Audio format:** Signed **PCM16**, **little-endian** (`pcm_s16le`), **mono**, **fixed sample rate** that must equal the server setting **`ASR_TARGET_SAMPLE_RATE`** (default **`16000`** Hz).
- **How audio is carried:** PCM bytes are **`base64`‑encoded** and placed in the **`audio_b64`** field inside a JSON **`audio_chunk`** message (not raw binary WebSocket payloads).

So: **PCM16 @ 16 kHz is correct**, but it is shipped as **Base64-inside-JSON**, not as raw binary frames beside JSON.

---

## Typical client mental model ↔ server `type`

| Flutter-style idea | Server `type` |
|-------------------|---------------|
| Live hypothesis, may change | `partial_transcript` |
| Locked text for current utterance | `final_transcript` |
| — | Extra fields: `session_id`, `utterance_id`, `seq`, timestamps, etc. |

There is **no** top-level `{ isFinal }` flag; **`final_transcript`** means final for that utterance. Partials use **`partial_transcript`** and include **`is_stable`** (hints at decoder stability).

---

## Client → server messages

All validated with **`type`** as discriminator. Unknown/extra JSON keys cause validation errors (**`PROTOCOL_ERROR`**).

### 1. `start` (must be **first** message)

Opens the session.

```json
{
  "type": "start",
  "session_id": "opaque-string-from-client",
  "sample_rate": 16000,
  "encoding": "pcm_s16le",
  "channels": 1,
  "language_hint": "si"
}
```

- **`sample_rate`:** Must match **`ASR_TARGET_SAMPLE_RATE`** on the server (default **16000**), or you receive **`INVALID_AUDIO_FORMAT`**.
- **`language_hint`:** Optional override; Whisper language code string (often `"si"` for Sinhala). Server defaults come from **`ASR_LANGUAGE_HINT`** when omitted/overridden depending on validation.

### 2. `audio_chunk` (streaming)

Repeated after `start`; each chunk is a short slice of PCM16 LE mono compressed only by **Base64**.

```json
{
  "type": "audio_chunk",
  "seq": 0,
  "audio_b64": "<base64 of PCM16 bytes>",
  "duration_ms": 100
}
```

- **`seq`:** Non‑decreasing counter ( **`>= 0`** ) so you can correlate order / debug reordering on your side if needed.
- **`duration_ms`:** Declared chunk length **1–2000** ms per schema (helps sanity-check capture cadence).
- **Size limit:** Decoded PCM size must not exceed **`ASR_MAX_CHUNK_BYTES`** (default **256 KiB**) or you receive **`PAYLOAD_TOO_LARGE`**.
- Capture **many small chunks** rather than buffering whole utterances → lower latency for partial transcripts.

### 3. `end_utterance`

Ends the **current utterance**: server runs a full-buffer decode and sends **`final_transcript`**, then resets the internal utterance buffer for the next phrase.

```json
{ "type": "end_utterance", "seq": 123 }
```

`seq` is optional.

### 4. `ping`

Heartbeat; server replies with **`ack`** and **`message": "pong"`**.

### 5. `stop`

Graceful shutdown: finalizes current utterance (if any), then connection teardown with **`session_summary`**.

---

## Server → client messages

### `ack`

After **`start`** success:

```json
{ "type": "ack", "session_id": "…", "message": "stream_started" }
```

After **`ping`**:

```json
{ "type": "ack", "session_id": "…", "message": "pong" }
```

### `partial_transcript`

Emitted periodically while audio is flowing (server **`ASR_PARTIAL_INTERVAL_MS`**, often **500** ms), when the model yields non‑empty interim text:

```json
{
  "type": "partial_transcript",
  "session_id": "…",
  "utterance_id": "uxxxxxxxx",
  "seq": 1,
  "text": " … ",
  "start_ms": 0,
  "end_ms": 1234,
  "is_stable": false
}
```

### `final_transcript`

After **`end_utterance`** or **`stop`** (and after a successful final decode):

```json
{
  "type": "final_transcript",
  "session_id": "…",
  "utterance_id": "uxxxxxxxx",
  "text": " … ",
  "start_ms": 0,
  "end_ms": 5000
}
```

Then the server rotates **`utterance_id`** for subsequent speech unless the connection ends.

### `error`

Protocol or runtime problems:

```json
{ "type": "error", "code": "PROTOCOL_ERROR", "message": "…" }
```

Common **`code`** values: **`PROTOCOL_ERROR`**, **`INVALID_AUDIO_FORMAT`**, **`PAYLOAD_TOO_LARGE`**, **`SESSION_TIMEOUT`**, **`INTERNAL_ERROR`**.

### `session_summary`

Sent once when the connection is closing gracefully from the protocol’s perspective:

```json
{
  "type": "session_summary",
  "session_id": "…",
  "utterances": 3,
  "duration_ms": 120000,
  "reason": "client_stop"
}
```

---

## Recommended capture loop (conceptual)

1. Open WebSocket → send **`start`** once.
2. From the mic recorder, every **20–100 ms** (or similar), take **PCM16 LE mono @ 16 kHz** bytes → **`audio_chunk`** with **`audio_b64`**, increment **`seq`**.
3. Show UI updates from **`partial_transcript`**.
4. On phrase end / user tap “send segment” → **`end_utterance`** → **`final_transcript`**.
5. On exit → **`stop`**.

---

## Source of truth in code

- **Route:** [`app/api/ws_transcribe.py`](../app/api/ws_transcribe.py)
- **Schemas:** [`app/sessions/schemas.py`](../app/sessions/schemas.py)
