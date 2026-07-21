"""
Event Lead Capture — FastAPI Backend
-------------------------------------
Receives lead submissions (form fields + audio + photo) from the frontend,
transcribes the audio directly via the Groq Whisper API, and sends the media
files + row data + transcript to a Google Apps Script Web App (which
saves them to Drive and Sheets under your own Google account, with no
Cloud billing/service-account setup required).

Deploy target: Render (or any ASGI host).
"""

import base64
import os
import uuid
import datetime
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Note: unlike the old Gemini path, no ffmpeg/webm-to-wav conversion is
# needed here — Groq's transcription endpoint accepts webm (along with flac,
# mp3, mp4, mpeg, mpga, m4a, ogg, wav) directly, so the raw browser-recorded
# bytes are sent as-is.

# ----------------------------------------------------------------------------
# Configuration (all via environment variables — see .env.example)
# ----------------------------------------------------------------------------
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")            # Google Apps Script /exec URL
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")          # Drive folder to store audio/photos
GROQ_API_KEY = os.getenv("GROQ_API_KEY")                   # Groq API key (console.groq.com/keys)
# whisper-large-v3-turbo is Groq's fastest/cheapest Whisper model and is
# accurate enough for this use case; whisper-large-v3 is kept as a slower,
# slightly more accurate fallback. Override via GROQ_WHISPER_MODELS
# (comma-separated) if needed.
GROQ_WHISPER_MODELS = os.getenv("GROQ_WHISPER_MODELS", "whisper-large-v3-turbo,whisper-large-v3").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not GROQ_API_KEY:
    print("[STARTUP WARNING] GROQ_API_KEY is not set — transcripts will be skipped entirely.")
else:
    # Log a short, safe prefix only — never the full key — so it's possible to
    # confirm at a glance in Render's logs that the right key is loaded,
    # without exposing it. Standard Groq keys start with "gsk_".
    print(f"[STARTUP] GROQ_API_KEY loaded (starts with '{GROQ_API_KEY[:6]}...', "
          f"length {len(GROQ_API_KEY)})")

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def transcribe_audio_with_groq(audio_bytes: bytes, mime_type: str, filename: str) -> str:
    """Sends the raw audio bytes to Groq's hosted Whisper API and returns the
    verbatim transcript.

    Unlike the old Gemini path (a general LLM prone to inventing plausible-
    sounding summaries for silent/unclear clips), Whisper is a dedicated
    speech-to-text model, so this returns an actual transcript rather than a
    summary. The "transcript" field name/column downstream is unchanged, it
    now just holds real transcribed text instead of a Gemini-written summary.

    Tries each model in GROQ_WHISPER_MODELS in order, falling through to the
    next on failure. Returns an empty string (rather than raising) if every
    model fails, so a Groq hiccup never blocks the lead from being saved."""
    if not GROQ_API_KEY:
        return ""

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    for model in GROQ_WHISPER_MODELS:
        model = model.strip()
        if not model:
            continue
        try:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (filename, audio_bytes, mime_type)},
                data={"model": model, "response_format": "json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data.get("text", "").strip()
            if not text:
                # Request succeeded (200 OK) but returned no text — usually a
                # silent/near-silent clip. Log the full response so this is
                # diagnosable instead of just looking like a blank transcript.
                print(f"[Groq transcript — empty text, model={model}] full response: {data}")
                return "(No transcript generated — audio may have been silent or too short)"

            print(f"[Groq transcript succeeded, model={model}, length={len(text)}]")
            return text
        except (requests.RequestException, KeyError) as exc:
            # Logged (not raised) so a transcription failure never blocks the
            # lead from being saved — but printed so you can see the real
            # cause in Render's Logs tab. Falls through to the next model.
            error_body = getattr(getattr(exc, "response", None), "text", "")
            print(f"[Groq transcript failed, model={model}] {exc} | response body: {error_body}")
            continue

    return "(Transcript generation failed — check Render logs)"


def save_lead_via_apps_script(
    lead_id: str,
    timestamp: str,
    rep_name: str,
    name: str,
    email: str,
    phone: str,
    company: str,
    industry: str,
    products: str,
    quantity: str,
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
        "rep_name": rep_name,
        "name": name,
        "email": email,
        "phone": phone,
        "company": company,
        "industry": industry,
        "products": products,
        "quantity": quantity,
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


@app.get("/")
def health_check():
    return {"status": "ok", "service": "event-lead-capture-api"}


@app.get("/api/diagnostics")
def diagnostics():
    """Quick, no-secrets-exposed way to check what's configured, without
    digging through Render logs. Visit this URL directly in a browser."""
    return {
        "groq_api_key_set": bool(GROQ_API_KEY),
        "groq_api_key_prefix": (GROQ_API_KEY[:6] + "...") if GROQ_API_KEY else None,
        "groq_whisper_models": GROQ_WHISPER_MODELS,
        "apps_script_url_set": bool(APPS_SCRIPT_URL),
        "gdrive_folder_id_set": bool(GDRIVE_FOLDER_ID),
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.post("/api/submit-lead")
async def submit_lead(
    rep_name: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    company: str = Form(...),
    industry: str = Form(...),
    products: Optional[str] = Form(""),
    quantity: Optional[str] = Form("1"),
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

    # Transcribe directly here, while we still have the raw bytes in memory.
    # Field name stays "transcript" throughout (Apps Script/Sheets) for
    # compatibility with the existing downstream setup.
    transcript = transcribe_audio_with_groq(
        audio_bytes, audio.content_type or "audio/webm", audio_filename
    )

    result = save_lead_via_apps_script(
        lead_id=lead_id,
        timestamp=timestamp,
        rep_name=rep_name,
        name=name,
        email=email,
        phone=phone,
        company=company,
        industry=industry,
        products=products,
        quantity=quantity,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        audio_mime_type=audio.content_type or "audio/webm",
        photo_bytes=photo_bytes,
        photo_filename=photo_filename,
        photo_mime_type=photo.content_type or "image/jpeg",
        transcript=transcript,
    )

    return {
        "status": "success",
        "lead_id": lead_id,
        "audio_url": result["audio_url"],
        "photo_url": result["photo_url"],
        "transcript": transcript,
    }
