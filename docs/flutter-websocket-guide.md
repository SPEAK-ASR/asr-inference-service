# Flutter guide: realtime streaming to `/ws/transcribe`

This guide matches the **current** Python service implementation (`app/api/ws_transcribe.py`, `app/sessions/schemas.py`). Use it to build a Flutter UI with **live partial transcripts** and **per-utterance finals**.

**Related:** backend-neutral field reference in [websocket-protocol.md](websocket-protocol.md).

---

## What you are building

1. Open a WebSocket to **`/ws/transcribe`**.
2. Send **`start`** once (must be the **first** JSON message).
3. While the user speaks, repeatedly send **`audio_chunk`** with **Base64-encoded PCM16 mono** samples at **`ASR_TARGET_SAMPLE_RATE`** (server default **`16000` Hz**).
4. Read JSON messages from the socket: **`partial_transcript`** (interim text), **`final_transcript`** (committed line), **`ack`**, **`error`**, **`session_summary`**.
5. Optionally send **`end_utterance`** to finalize the current phrase without closing the socket, or **`stop`** to finalize and disconnect cleanly.

**Important:** The server expects **JSON text frames only**. It calls `receive_text()` and validates JSON. PCM is **not** sent as a binary WebSocket frame; it is embedded as **`audio_b64`** inside JSON.

---

## Endpoint URL

Same host as your REST API; path is **`/ws/transcribe`**.

- Local dev (Android emulator → host PC): **`ws://10.0.2.2:8000/ws/transcribe`**
- iOS Simulator → Mac: **`ws://127.0.0.1:8000/ws/transcribe`**
- Physical device on LAN: **`ws://<PC_LAN_IP>:8000/ws/transcribe`**
- HTTPS sites use **`wss://`** behind TLS.

---

## Server timing (affects UX)

These come from **`ASR_*`** env vars (defaults in `app/core/config.py`):

| Behavior | Setting | Default |
|----------|---------|--------|
| How often partial **decode** attempts run | `partial_interval_ms` | **500** ms |
| Minimum buffered audio before a partial can return non-empty text | driven by decoder + **`min_audio_for_partial_seconds`** | **~0.6** s typical |
| Max sliding buffer | `max_buffer_seconds` | **30** s |
| Closing session after **no** `audio_chunk` | `idle_timeout_seconds` | **60** s |
| Max decoded PCM per chunk | `max_chunk_bytes` | **262144** (256 KiB) |

**UI implication:** Expect **nothing useful for roughly the first 0.5–0.8 s** of speech; then **`partial_transcript`** events every ~500 ms **only when text changes** (see decoder below).

---

## Client → server JSON (strict)

Pydantic uses **`extra="forbid"`**. **Do not** add unknown keys (e.g. no `metadata` until the backend supports it).

### 1. `start` (required first message)

```json
{
  "type": "start",
  "session_id": "<your-id-max-128>",
  "sample_rate": 16000,
  "encoding": "pcm_s16le",
  "channels": 1,
  "language_hint": "si"
}
```

Rules enforced by the server:

- **`sample_rate`** must equal the server **`ASR_TARGET_SAMPLE_RATE`** (default **16000**) or you get **`INVALID_AUDIO_FORMAT`**.
- **`encoding`** must be exactly **`"pcm_s16le"`** (validated on the schema; omitting defaults it for some clients — safer to send explicitly).
- **`channels`** must be **`1`**.
- **`language_hint`:** optional-ish; schema default **`"si"`**. Server uses **`start.language_hint` or fallback `settings.language_hint`**.

Success → **`{"type":"ack","session_id":"…","message":"stream_started"}`**.

### 2. `audio_chunk` (streaming)

```json
{
  "type": "audio_chunk",
  "seq": 0,
  "audio_b64": "<BASE64(Standard) of raw PCM bytes>",
  "duration_ms": 40
}
```

- **`seq`:** **`>= 0`**, increment per chunk (recommended monotonic).
- **`audio_b64`:** PCM **signed 16‑bit**, **little‑endian**, **mono**, sampled at **`sample_rate`** from `start`. No WAV header — raw samples only.
- **`duration_ms`:** backend schema requires **`1 .. 2000`**. Prefer the **approximate duration** of that chunk \(\approx \mathrm{samples} \times 1000 / \mathrm{sampleRate}\)).

After Base64 decode, length must not exceed **`ASR_MAX_CHUNK_BYTES`** (256 KiB) or you receive **`PAYLOAD_TOO_LARGE`**.

Recommended chunk rhythm: send every **40–120 ms** of audio (~1280–3840 bytes PCM at 16 kHz) — small chunks keep latency low.

### 3. `end_utterance`

```json
{ "type": "end_utterance" }
```

Optional **`"seq": 123**. Triggers a **full-buffer** decode and a **`final_transcript`**, then the server starts a **new `utterance_id`** and clears the rolling buffer for the next phrase.

### 4. `ping` / `stop`

```json
{ "type": "ping" }
```

→ **`ack`** with **`"message": "pong"`**.

```json
{ "type": "stop" }
```

Runs finalize like **`end_utterance`** if there is audio, then closes the session (see **`session_summary`** below).

---

## Server → client JSON

Dispatch on **`type`**.

### `partial_transcript`

```json
{
  "type": "partial_transcript",
  "session_id": "…",
  "utterance_id": "uabc12ef",
  "seq": 3,
  "text": "interim …",
  "start_ms": 0,
  "end_ms": 4200,
  "is_stable": false
}
```

**Logic (current code):** Partials are emitted only when the incremental decoder accepts a new hypothesis (`app/asr/decoder.py`): text must be non-empty and **change** compared to the last emitted partial (length / equality rules with `min_partial_char_delta`). **`is_stable`** is `true` when the current text **starts with** an updated **longest common prefix** between the last two hypotheses — useful to style “likely stable” prefix vs tail in the UI.

### `final_transcript`

```json
{
  "type": "final_transcript",
  "session_id": "…",
  "utterance_id": "uabc12ef",
  "text": "Final line.",
  "start_ms": 0,
  "end_ms": 5100
}
```

Use this to **commit** text in your transcript list and **clear** interim UI for that utterance.

### `error`

```json
{ "type": "error", "code": "PROTOCOL_ERROR", "message": "…" }
```

**Note:** Invalid JSON/schema often gets **`PROTOCOL_ERROR`** and the socket **may stay open** (`continue` loop). Fix the outgoing message shape.

Idle timeout **`SESSION_TIMEOUT`**: server sends **`error`** then **`close(1001)`** (approximate timing — see SessionManager).

### `session_summary`

Sent in **`finally`** after a **`start`**‑accepted session closes (including **`stop`** and disconnect):

```json
{
  "type": "session_summary",
  "session_id": "…",
  "utterances": 2,
  "duration_ms": 45000,
  "reason": "client_stop"
}
```

**reason** examples: **`client_stop`**, **`client_disconnect`**, **`idle_timeout`**, **`internal_error`**, etc.

If **`start`** never succeeded (e.g. wrong first message), you may **not** get a **`session_summary`**.

---

## Flutter audio requirements

Your recorder must ultimately produce **`Int16`** samples, **little-endian** byte stream, **16000 Hz**, **mono**.

1. If the microphone plugin captures **44100 Hz** or **48000 Hz**, **resample to 16000 Hz** before encoding (Dart packages such as **`flutter_soloud`** / DSP libs, or platform-specific resamplers). Sending wrong-rate audio without resampling hurts accuracy.
2. If you get **float32** PCM, convert: \( s_{16} = \mathrm{clamp}(\mathrm{round}(f \times 32767), -32768, 32767) \).
3. Encode bytes with **`base64Encode`** (dart:convert) into **`audio_b64`**.

**Do not** wrap PCM in WAV inside **`audio_chunk`** unless you strip the header (server expects raw PCM only).

---

## Suggested Flutter packages

| Need | Typical package |
|------|----------------|
| WebSocket | **`web_socket_channel`** |
| JSON | **`dart:convert`** |

Microphone/recording varies by stack (`record`, `flutter_sound`, etc.). Regardless of plugin, satisfy **PCM16 mono 16 kHz** before **`audio_chunk`**.

**Android:** `RECORD_AUDIO` permission; **iOS:** `NSMicrophoneUsageDescription`.

---

## Minimal WebSocket wiring (Dart)

Use this pattern; plug your recorder into **`sendChunks`**.

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:web_socket_channel/web_socket_channel.dart';

const int sampleRate = 16000;

class TranscriptionClient {
  TranscriptionClient(this.uri);

  final Uri uri;
  WebSocketChannel? _ch;
  int _seq = 0;

  void connect({
    required String sessionId,
    String languageHint = 'si',
  }) {
    _ch = WebSocketChannel.connect(uri);

    _ch!.stream.listen(_onMessage, onError: onError, onDone: onDone);

    final start = {
      'type': 'start',
      'session_id': sessionId,
      'sample_rate': sampleRate,
      'encoding': 'pcm_s16le',
      'channels': 1,
      'language_hint': languageHint,
    };
    _ch!.sink.add(jsonEncode(start));
  }

  void _onMessage(dynamic message) {
    final raw = message is String ? message : utf8.decode(message as List<int>);
    final map = jsonDecode(raw) as Map<String, dynamic>;
    switch (map['type']) {
      case 'ack':
        onAck(map);
      case 'partial_transcript':
        onPartial(map);
      case 'final_transcript':
        onFinal(map);
      case 'error':
        onErrorPayload(map);
      case 'session_summary':
        onSessionSummary(map);
      default:
        onUnknown(map);
    }
  }

  /// [pcmLittleEndianMono] raw s16le bytes (multiple of 2).
  void sendAudioChunk(Uint8List pcmLittleEndianMono, {required int durationMs}) {
    final ch = _ch;
    if (ch == null) return;
    if (pcmLittleEndianMono.length % 2 != 0) {
      throw ArgumentError('PCM16 expects even byte length');
    }
    if (durationMs < 1 || durationMs > 2000) {
      throw ArgumentError('duration_ms must be 1–2000 per server schema');
    }
    final chunk = <String, dynamic>{
      'type': 'audio_chunk',
      'seq': _seq++,
      'audio_b64': base64Encode(pcmLittleEndianMono),
      'duration_ms': durationMs,
    };
    ch.sink.add(jsonEncode(chunk));
  }

  Future<void> endUtterance() async {
    _ch?.sink.add(jsonEncode({'type': 'end_utterance'}));
  }

  Future<void> stop() async {
    _ch?.sink.add(jsonEncode({'type': 'stop'}));
    await _ch?.sink.close();
    _ch = null;
  }

  // --- Override or wire with ValueNotifier / BLoC / Riverpod ---
  void onAck(Map<String, dynamic> m) {}
  void onPartial(Map<String, dynamic> m) {}
  void onFinal(Map<String, dynamic> m) {}
  void onErrorPayload(Map<String, dynamic> m) {}
  void onSessionSummary(Map<String, dynamic> m) {}
  void onUnknown(Map<String, dynamic> m) {}
  void onError(Object e, StackTrace st) {}
  void onDone() {}
}
```

**Realtime UI pattern**

- Maintain **`partialText`** and **`committedLines`** (`List<String>` or a single **`StringBuffer`** plus current line).
- On **`partial_transcript`**, update the **current line** overlay (often under **`utterance_id`**).
- On **`final_transcript`**, append **`text`** as a finalized row and clear **`partialText`** for **that utterance**.
- When **`utterance_id`** changes, treat it as a new line in the transcript view.

Optional: fade or bold the prefix while **`is_stable == true`** and keep the trailing characters in a tentative style.

---

## Keepalive vs idle shutdown

Backend closes idle connections after **`idle_timeout_seconds`** without **`audio_chunk`**. Strategies:

1. **`ping`** periodically while connected (answered with **`ack` / pong`) — idle timer is keyed off **`audio_chunk`** arrival (`last_chunk_at` in `SessionState`), **not ping**, so pings **alone do not refresh idle timeout** today.
2. Avoid long pauses **without chunks** unless you **`stop`** and reconnect later.

Design choice for your UX: **`stop`** when user leaves screen; reconnect with a new **`session_id`** next time.

---

## Quick validation checklist

- [ ] **`start`** first; **`sample_rate` == server `TARGET_SAMPLE_RATE` (16000 by default)**  
- [ ] PCM **s16le**, **mono**, **no extra JSON keys**  
- [ ] **`duration_ms` in 1…2000**  
- [ ] Base64 decodes to **≤ 256 KiB** per chunk  
- [ ] Handle **`partial_transcript`**, **`final_transcript`**, **`error`**, **`session_summary`**  
- [ ] Remember **~0.6 s** before useful partials  
- [ ] **`end_utterance`** or **`stop`** to get finals  

---

## Source of truth in this repo

| Area | File |
|------|------|
| Handlers & flow | `app/api/ws_transcribe.py` |
| JSON shapes | `app/sessions/schemas.py` |
| Idle timeout / session | `app/sessions/manager.py` |
| Partial emission policy | `app/asr/decoder.py` |
