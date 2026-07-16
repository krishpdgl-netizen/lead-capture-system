"""
Event Lead Capture — FastAPI Backend
-------------------------------------
Receives lead submissions (form fields + audio + photo) from the frontend,
sends the media files + row data to a Google Apps Script Web App (which
saves them to Drive and Sheets under your own Google account, with no
Cloud billing/service-account setup required), and triggers an n8n
webhook to kick off the AI follow-up pipeline.

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
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
) -> dict:
    """Sends files + row data to the Apps Script Web App, which saves them
    to Drive and appends a row to Sheets under your own Google account."""
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
        }
    )

    return {"status": "success", "lead_id": lead_id, "audio_url": audio_link, "photo_url": photo_link}
