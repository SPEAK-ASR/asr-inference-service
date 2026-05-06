# HTTP Audio Transcription API

This API accepts a recorded audio file and returns transcription text in the same response.

## Endpoint

- `POST /api/transcribe`

## Request

- Content type: `multipart/form-data`
- Form fields:
  - `audio_file` (required): recorded audio file
  - `language` (optional): language hint (defaults to server `ASR_LANGUAGE_HINT`)

Supported MIME types (configurable):
- `audio/wav`
- `audio/x-wav`
- `audio/webm`
- `audio/mpeg`
- `audio/mp3`
- `audio/mp4`
- `audio/x-m4a`
- `audio/aac`
- `audio/ogg`

## Response

`200 OK`

```json
{
  "text": "transcribed text",
  "language": "si",
  "duration_ms": 350,
  "model_kind": "peft"
}
```

## Errors

- `400`: invalid audio payload, empty upload, or unsupported MIME type
- `413`: upload exceeds `ASR_HTTP_TRANSCRIBE_MAX_UPLOAD_BYTES`
- `422`: missing required form fields
- `500`: transcription or timeout failure

## Environment Variables

- `ASR_HTTP_TRANSCRIBE_MAX_UPLOAD_BYTES` (default: `10485760`)
- `ASR_HTTP_TRANSCRIBE_ALLOWED_MIME_TYPES` (comma-separated list)
- `ASR_HTTP_TRANSCRIBE_TIMEOUT_SECONDS` (default: `120`)

## Frontend `fetch` Example

```javascript
const form = new FormData();
form.append("audio_file", recordedBlob, "recorded.webm");
form.append("language", "si");

const response = await fetch("http://localhost:8000/api/transcribe", {
  method: "POST",
  body: form,
});

const data = await response.json();
console.log(data.text);
```
