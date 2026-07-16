"""
Event Lead Capture — FastAPI Backend
-------------------------------------
Receives lead submissions (form fields + audio + photo) from the frontend,
uploads the media files to Google Drive, appends a row to Google Sheets,
and triggers an n8n webhook to kick off the AI follow-up pipeline.

Deploy target: Render (or any ASGI host).
"""

import io
import os
import uuid
import datetime
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ----------------------------------------------------------------------------
# Configuration (all via environment variables — see .env.example)
# ----------------------------------------------------------------------------
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")          # Drive folder to store audio/photos
GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID") # Sheet that stores the lead database
GSHEET_TAB_NAME = os.getenv("GSHEET_TAB_NAME", "Leads")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")             # n8n workflow trigger URL
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _google_credentials():
    if not os.path.exists(GOOGLE_SERVICE_ACCOUNT_FILE):
        raise HTTPException(
            status_code=500,
            detail=(
                "Google service account file not found. Set GOOGLE_SERVICE_ACCOUNT_FILE "
                "and mount/upload the credentials JSON — see README."
            ),
        )
    return service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )


def upload_to_drive(file_bytes: bytes, filename: str, mime_type: str) -> str:
    """Uploads a file to the configured Drive folder and returns a shareable link."""
    creds = _google_credentials()
    drive = build("drive", "v3", credentials=creds)

    file_metadata = {"name": filename}
    if GDRIVE_FOLDER_ID:
        file_metadata["parents"] = [GDRIVE_FOLDER_ID]

    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=False)
    created = drive.files().create(body=file_metadata, media_body=media, fields="id").execute()
    file_id = created["id"]

    # Make it link-viewable (adjust to your org's sharing policy as needed)
    drive.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()

    return f"https://drive.google.com/file/d/{file_id}/view"


def append_to_sheet(row: list):
    creds = _google_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    sheets.spreadsheets().values().append(
        spreadsheetId=GSHEET_SPREADSHEET_ID,
        range=f"{GSHEET_TAB_NAME}!A:Z",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()


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

    audio_link = upload_to_drive(audio_bytes, audio_filename, audio.content_type or "audio/webm")
    photo_link = upload_to_drive(photo_bytes, photo_filename, photo.content_type or "image/jpeg")

    row = [lead_id, timestamp, name, email, phone, company, industry, products, audio_link, photo_link, "New"]
    append_to_sheet(row)

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
