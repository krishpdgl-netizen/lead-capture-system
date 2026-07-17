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
import uuid
import datetime
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-flash-latest,gemini-2.5-flash").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def transcribe_audio_with_gemini(audio_bytes: bytes, mime_type: str) -> str:
    """Sends the raw audio bytes to Gemini and returns a transcript + brief
    summary of what the lead said. Tries each model in GEMINI_MODELS in order,
    falling through to the next on failure. Returns an empty string (rather
    than raising) if every model fails, so a Gemini hiccup never blocks the
    lead from being saved."""
    if not GEMINI_API_KEY:
        return ""

    prompt = (
        "Transcribe this audio clip from a trade-show/event booth conversation. "
        "Then, on a new line starting with 'Summary:', give a 2-3 sentence summary "
        "of what the customer is interested in and any specific needs they mentioned."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode("utf-8")}},
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
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (requests.RequestException, KeyError, IndexError) as exc:
            # Logged (not raised) so a transcription failure never blocks the
            # lead from being saved — but printed so you can see the real
            # cause in Render's Logs tab. Falls through to the next model.
            error_body = getattr(getattr(exc, "response", None), "text", "")
            print(f"[Gemini transcription failed, model={model}] {exc} | response body: {error_body}")
            continue

    return ""


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
):
    lead_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.utcnow().isoformat()

    audio_bytes = await audio.read()
    photo_bytes = await photo.read()

    audio_filename = f"{lead_id}_{name.replace(' ', '_')}_audio.webm"
    photo_filename = f"{lead_id}_{name.replace(' ', '_')}_photo.jpg"

    # Transcribe directly here, while we still have the raw bytes in memory —
    # this replaces the old plan of downloading the file again inside n8n.
    transcript = transcribe_audio_with_gemini(
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
