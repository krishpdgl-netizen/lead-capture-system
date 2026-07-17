"""
Event Lead Capture — FastAPI Backend
-------------------------------------
Receives lead submissions (form fields + audio + photo) from the frontend,
transcribes the audio directly via the Gemini API (no separate n8n step
needed), sends the media files + row data + transcript to a Google Apps
Script Web App (which saves them to Drive and Sheets under your own
Google account, with no Cloud billing/service-account setup required),
and triggers an n8n webhook with the transcript already included so n8n
only has to draft the quotation email and send it.

Deploy target: Render (or any ASGI host).
"""

import base64
import os
import subprocess
import tempfile
import uuid
import datetime
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# imageio_ffmpeg is used to convert browser-recorded audio (webm) into a
# format Gemini accepts (see _convert_to_wav below). Importing it is wrapped
# in a try/except so that if it's ever missing or fails to install for any
# reason, the WHOLE APP doesn't refuse to boot — leads still get saved, audio
# conversion just gets skipped and it's logged clearly at startup instead of
# crashing uvicorn. (This is exactly what caused the last deploy to roll back
# silently to old code — a missing dependency took the entire API down.)
try:
    import imageio_ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError as exc:
    imageio_ffmpeg = None
    _FFMPEG_AVAILABLE = False
    print(f"[STARTUP WARNING] imageio_ffmpeg not available ({exc}) — "
          f"audio will be sent to Gemini in its original format, which may be rejected.")

# ----------------------------------------------------------------------------
# Configuration (all via environment variables — see .env.example)
# ----------------------------------------------------------------------------
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")            # Google Apps Script /exec URL
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")          # Drive folder to store audio/photos
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")             # n8n workflow trigger URL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")               # Gemini API key (aistudio.google.com/apikey)
# Pinned model versions (gemini-2.0-flash, gemini-2.5-flash-lite, gemini-2.5-flash,
# etc.) keep getting closed off to new API keys as Google rotates its lineup.
# "gemini-flash-latest" is Google's own auto-updating alias for the current
# recommended Flash model, so it's the safer default — it won't need to be
# manually bumped every time a pinned version gets deprecated. A second,
# older model is kept as a fallback in case the alias itself ever has a
# transient issue. Override via GEMINI_MODELS (comma-separated) if needed.
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-flash-latest,gemini-2.0-flash").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not GEMINI_API_KEY:
    print("[STARTUP WARNING] GEMINI_API_KEY is not set — summaries will be skipped entirely.")
else:
    # Log a short, safe prefix only — never the full key — so it's possible to
    # confirm at a glance in Render's logs that the right key is loaded,
    # without exposing it. Standard Google AI Studio keys start with "AIza".
    print(f"[STARTUP] GEMINI_API_KEY loaded (starts with '{GEMINI_API_KEY[:6]}...', "
          f"length {len(GEMINI_API_KEY)})")

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _convert_to_wav(audio_bytes: bytes) -> Optional[bytes]:
    """Converts arbitrary audio bytes to WAV using a bundled static ffmpeg
    binary (via imageio-ffmpeg — no system/apt install needed on Render).

    This exists because the browser's MediaRecorder API (used on the
    frontend) produces audio/webm (Opus codec), but Gemini's generateContent
    API only officially supports WAV, MP3, AIFF, AAC, OGG Vorbis, and FLAC.
    Sending audio/webm as-is reliably gets rejected with an "Unsupported MIME
    type" error, which would otherwise look identical to any other failure.
    Converting to WAV first avoids that entirely.

    Returns None (rather than raising) if ffmpeg isn't available or
    conversion fails, so a conversion hiccup falls back to attempting the
    original bytes rather than blocking the lead from being saved."""
    if not _FFMPEG_AVAILABLE:
        return None

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    src_path = dst_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
            src.write(audio_bytes)
            src_path = src.name
        dst_path = src_path + ".wav"

        subprocess.run(
            [ffmpeg_path, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path],
            check=True, capture_output=True, timeout=30,
        )
        with open(dst_path, "rb") as f:
            return f.read()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        stderr = getattr(exc, "stderr", b"")
        stderr = stderr.decode(errors="ignore") if isinstance(stderr, bytes) else stderr
        print(f"[Audio conversion to WAV failed] {exc} | stderr: {stderr}")
        return None
    finally:
        for path in (src_path, dst_path):
            if path and os.path.exists(path):
                os.remove(path)


def summarize_audio_with_gemini(audio_bytes: bytes, mime_type: str) -> str:
    """Sends the raw audio bytes to Gemini and returns a short summary of
    what the lead said — no verbatim transcript, just the summary directly.
    This is intentionally simpler than a full transcribe-then-summarize
    prompt: fewer output tokens, nothing to parse out of a longer response,
    and it's the only part n8n actually uses downstream anyway.

    Tries each model in GEMINI_MODELS in order, falling through to the next
    on failure. Returns an empty string (rather than raising) if every model
    fails, so a Gemini hiccup never blocks the lead from being saved."""
    if not GEMINI_API_KEY:
        return ""

    # Convert to a Gemini-supported format first (see _convert_to_wav above).
    # If conversion fails or ffmpeg isn't available, fall back to sending the
    # original bytes/mime type — unlikely to succeed for webm, but a better
    # fallback than giving up before even trying.
    wav_bytes = _convert_to_wav(audio_bytes)
    if wav_bytes is not None:
        send_bytes, send_mime = wav_bytes, "audio/wav"
    else:
        send_bytes, send_mime = audio_bytes, mime_type

    prompt = (
        "Listen to this audio clip of a trade-show/event booth conversation. "
        "In 2-3 sentences, summarize what the customer is interested in, "
        "their company or use case, and any specific needs, quantities, or "
        "timelines they mentioned. Do not include a verbatim transcript — "
        "just the summary, as plain text with no headers or labels."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": send_mime, "data": base64.b64encode(send_bytes).decode("utf-8")}},
                    {"text": prompt},
                ]
            }
        ]
    }

    for model in GEMINI_MODELS:
        model = model.strip()
        if not model:
            continue
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                # Request succeeded (200 OK) but Gemini returned zero
                # candidates — usually a safety block or an empty/near-silent
                # clip. Log the full response so this is diagnosable instead
                # of just looking like a blank transcript again.
                print(f"[Gemini summary — no candidates returned, model={model}] full response: {data}")
                return "(No summary generated — audio may have been silent or too short)"

            finish_reason = candidates[0].get("finishReason", "")
            summary = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )

            if not summary:
                # Call succeeded and returned a candidate, but with no text —
                # log finishReason (e.g. SAFETY, MAX_TOKENS) so the cause is
                # visible rather than looking identical to a total failure.
                print(f"[Gemini summary — empty text, model={model}, finishReason={finish_reason}] full response: {data}")
                return "(No summary generated — audio may have been silent or too short)"

            print(f"[Gemini summary succeeded, model={model}, length={len(summary)}]")
            return summary
        except (requests.RequestException, KeyError, IndexError) as exc:
            # Logged (not raised) so a summarization failure never blocks the
            # lead from being saved — but printed so you can see the real
            # cause in Render's Logs tab. Falls through to the next model.
            error_body = getattr(getattr(exc, "response", None), "text", "")
            print(f"[Gemini summary failed, model={model}] {exc} | response body: {error_body}")
            continue

    return "(Summary generation failed — check Render logs)"


def save_lead_via_apps_script(
    lead_id: str,
    timestamp: str,
    name: str,
    email: str,
    phone: str,
    company: str,
    industry: str,
    products: str,
    audio_bytes: bytes,
    audio_filename: str,
    audio_mime_type: str,
    photo_bytes: bytes,
    photo_filename: str,
    photo_mime_type: str,
    transcript: str,
) -> dict:
    """Sends files + row data to the Apps Script Web App, which saves them
    to Drive and appends a row to Sheets under your own Google account."""
    # (transcript is threaded through so it lands in its own Sheet column)
    if not APPS_SCRIPT_URL:
        raise HTTPException(
            status_code=500,
            detail="APPS_SCRIPT_URL is not set — deploy the Apps Script Web App and set its /exec URL.",
        )
    if not GDRIVE_FOLDER_ID:
        raise HTTPException(status_code=500, detail="GDRIVE_FOLDER_ID is not set.")

    payload = {
        "gdrive_folder_id": GDRIVE_FOLDER_ID,
        "lead_id": lead_id,
        "timestamp": timestamp,
        "name": name,
        "email": email,
        "phone": phone,
        "company": company,
        "industry": industry,
        "products": products,
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "audio_filename": audio_filename,
        "audio_mime_type": audio_mime_type,
        "photo_base64": base64.b64encode(photo_bytes).decode("utf-8"),
        "photo_filename": photo_filename,
        "photo_mime_type": photo_mime_type,
        "transcript": transcript,
    }

    try:
        resp = requests.post(APPS_SCRIPT_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Apps Script request failed: {exc}")

    if result.get("status") != "success":
        raise HTTPException(status_code=502, detail=f"Apps Script error: {result.get('message')}")

    return result


def trigger_n8n(payload: dict):
    if not N8N_WEBHOOK_URL:
        return
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
    except requests.RequestException:
        # Don't fail the whole request if n8n is briefly unreachable —
        # the lead is already saved in Sheets/Drive at this point.
        pass


@app.get("/")
def health_check():
    return {"status": "ok", "service": "event-lead-capture-api"}


@app.get("/api/diagnostics")
def diagnostics():
    """Quick, no-secrets-exposed way to check what's configured, without
    digging through Render logs. Visit this URL directly in a browser."""
    return {
        "gemini_api_key_set": bool(GEMINI_API_KEY),
        "gemini_api_key_prefix": (GEMINI_API_KEY[:6] + "...") if GEMINI_API_KEY else None,
        "gemini_models": GEMINI_MODELS,
        "ffmpeg_available": _FFMPEG_AVAILABLE,
        "apps_script_url_set": bool(APPS_SCRIPT_URL),
        "gdrive_folder_id_set": bool(GDRIVE_FOLDER_ID),
        "n8n_webhook_url_set": bool(N8N_WEBHOOK_URL),
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.post("/api/submit-lead")
async def submit_lead(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    company: str = Form(...),
    industry: str = Form(...),
    products: Optional[str] = Form(""),
    audio: UploadFile = File(...),
    photo: UploadFile = File(...),
    client_lead_id: Optional[str] = Form(None),
):
    # If the frontend generated its own ID (used by the offline queue so a
    # lead captured with no connection keeps the same ID whenever it later
    # syncs), use that instead of minting a new one here.
    lead_id = (client_lead_id or str(uuid.uuid4()))[:8]
    # India Standard Time is a fixed UTC+5:30 offset with no daylight saving,
    # so a plain timedelta add is reliable here without needing pytz/zoneinfo
    # tzdata bundled in the deploy.
    IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
    timestamp = (datetime.datetime.now(datetime.timezone.utc) + IST_OFFSET).strftime("%Y-%m-%d %H:%M:%S IST")

    audio_bytes = await audio.read()
    photo_bytes = await photo.read()

    audio_filename = f"{lead_id}_{name.replace(' ', '_')}_audio.webm"
    photo_filename = f"{lead_id}_{name.replace(' ', '_')}_photo.jpg"

    # Summarize directly here, while we still have the raw bytes in memory —
    # this replaces the old plan of downloading the file again inside n8n.
    # Field name stays "transcript" throughout (Apps Script/Sheets/n8n) for
    # compatibility with the existing downstream setup — it just now holds a
    # short summary instead of a full verbatim transcript.
    transcript = summarize_audio_with_gemini(
        audio_bytes, audio.content_type or "audio/webm"
    )

    result = save_lead_via_apps_script(
        lead_id=lead_id,
        timestamp=timestamp,
        name=name,
        email=email,
        phone=phone,
        company=company,
        industry=industry,
        products=products,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        audio_mime_type=audio.content_type or "audio/webm",
        photo_bytes=photo_bytes,
        photo_filename=photo_filename,
        photo_mime_type=photo.content_type or "image/jpeg",
        transcript=transcript,
    )

    audio_link = result["audio_url"]
    photo_link = result["photo_url"]

    trigger_n8n(
        {
            "lead_id": lead_id,
            "timestamp": timestamp,
            "name": name,
            "email": email,
            "phone": phone,
            "company": company,
            "industry": industry,
            "products": products,
            "audio_url": audio_link,
            "photo_url": photo_link,
            "transcript": transcript,
        }
    )

    return {
        "status": "success",
        "lead_id": lead_id,
        "audio_url": audio_link,
        "photo_url": photo_link,
        "transcript": transcript,
    }
