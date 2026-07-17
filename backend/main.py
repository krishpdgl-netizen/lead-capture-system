"""
Event Lead Capture — FastAPI Backend
-------------------------------------
Receives lead submissions (form fields + audio + photo) from the frontend,
transcribes+summarizes the audio directly via the Gemini API, matches the
conversation against a product catalog (also via Gemini, in the same call),
sends everything to a Google Apps Script Web App (which saves files to
Drive and rows to Sheets under your own Google account), and triggers an
n8n webhook as a best-effort notification (n8n is currently run manually,
so this call is expected to no-op/fail silently — see README notes).

Deploy target: Render (or any ASGI host).
"""

import base64
import json
import os
import subprocess
import tempfile
import time
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
# crashing uvicorn.
try:
    import imageio_ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError as exc:
    imageio_ffmpeg = None
    _FFMPEG_AVAILABLE = False
    print(f"[STARTUP WARNING] imageio_ffmpeg not available ({exc}) — "
          f"audio will be sent to Gemini in its original format, which may be rejected.")

# ----------------------------------------------------------------------------
# Configuration (all via environment variables)
# ----------------------------------------------------------------------------
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")            # Google Apps Script /exec URL
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")          # Drive folder to store audio/photos
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")             # n8n workflow trigger URL (currently unused — n8n runs manually)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")               # Gemini API key (aistudio.google.com/apikey)
# "gemini-flash-latest" auto-updates to Google's current recommended Flash
# model, so it doesn't need manual bumping every time a pinned version gets
# deprecated. A second model is kept as a fallback in case the alias itself
# has a transient issue. Override via GEMINI_MODELS (comma-separated).
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-flash-latest,gemini-3.1-flash-lite").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
# How long to cache the product catalog in memory before re-fetching from
# the Sheet, in seconds. The catalog changes rarely, so there's no need to
# hit Apps Script on every single lead submission.
CATALOG_CACHE_TTL_SECONDS = int(os.getenv("CATALOG_CACHE_TTL_SECONDS", "900"))  # 15 min

if not GEMINI_API_KEY:
    print("[STARTUP WARNING] GEMINI_API_KEY is not set — summaries and product matching will be skipped entirely.")
else:
    print(f"[STARTUP] GEMINI_API_KEY loaded (starts with '{GEMINI_API_KEY[:6]}...', "
          f"length {len(GEMINI_API_KEY)})")

app = FastAPI(title="Event Lead Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# In-memory catalog cache
# ----------------------------------------------------------------------------
_catalog_cache = {"products": [], "fetched_at": 0.0}


def fetch_catalog(force_refresh: bool = False) -> list:
    """Fetches the product catalog from the Apps Script Web App (GET
    ?action=catalog) and caches it in memory for CATALOG_CACHE_TTL_SECONDS.

    Returns a list of dicts: [{product_id, name, category, price,
    description, keywords, image_url}, ...]. Returns an empty list (rather
    than raising) on any failure, so a catalog-fetch hiccup never blocks a
    lead from being saved — matching just gets skipped for that submission."""
    now = time.time()
    if not force_refresh and _catalog_cache["products"] and (now - _catalog_cache["fetched_at"] < CATALOG_CACHE_TTL_SECONDS):
        return _catalog_cache["products"]

    if not APPS_SCRIPT_URL:
        print("[Catalog fetch skipped] APPS_SCRIPT_URL is not set.")
        return _catalog_cache["products"]  # whatever's cached, even if stale/empty

    try:
        resp = requests.get(APPS_SCRIPT_URL, params={"action": "catalog"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        products = data.get("products", [])
        _catalog_cache["products"] = products
        _catalog_cache["fetched_at"] = now
        print(f"[Catalog fetch succeeded] {len(products)} products cached.")
        return products
    except (requests.RequestException, ValueError) as exc:
        print(f"[Catalog fetch failed] {exc}")
        return _catalog_cache["products"]  # fall back to stale cache rather than nothing


# ----------------------------------------------------------------------------
# Audio conversion
# ----------------------------------------------------------------------------
def _convert_to_wav(audio_bytes: bytes) -> Optional[bytes]:
    """Converts arbitrary audio bytes to WAV using a bundled static ffmpeg
    binary (via imageio-ffmpeg). Returns None (rather than raising) if
    ffmpeg isn't available or conversion fails."""
    if not _FFMPEG_AVAILABLE:
        return None

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    # Some hosts (Render included) can strip the execute bit off pip-installed
    # binaries during build/deploy. Re-asserting it here is cheap insurance
    # against a silent, hard-to-diagnose PermissionError on every request.
    try:
        os.chmod(ffmpeg_path, 0o755)
    except OSError:
        pass

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


# ----------------------------------------------------------------------------
# Gemini: summarize audio AND match it against the catalog in one call
# ----------------------------------------------------------------------------
def _build_catalog_listing(catalog: list) -> str:
    """Compact, token-cheap text listing of the catalog for the prompt.
    Full descriptions are truncated — Gemini only needs enough to judge
    relevance, not the full marketing copy."""
    lines = []
    for p in catalog:
        desc = (p.get("description") or "")[:120]
        lines.append(
            f"{p.get('product_id')} | {p.get('name')} | {p.get('category')} | "
            f"₹{p.get('price')} | keywords: {p.get('keywords')} | {desc}"
        )
    return "\n".join(lines)


def analyze_audio_with_gemini(audio_bytes: bytes, mime_type: str, catalog: list) -> dict:
    """Sends the raw audio bytes + a compact catalog listing to Gemini and
    asks for BOTH a short summary and a list of matched product_ids, as
    structured JSON in a single call.

    Returns {"summary": str, "matched_product_ids": list[str]}. Returns
    {"summary": "", "matched_product_ids": []} if every model fails, so a
    Gemini hiccup never blocks the lead from being saved."""
    empty_result = {"summary": "", "matched_product_ids": []}
    if not GEMINI_API_KEY:
        return empty_result

    wav_bytes = _convert_to_wav(audio_bytes)
    if wav_bytes is not None:
        send_bytes, send_mime = wav_bytes, "audio/wav"
    else:
        send_bytes, send_mime = audio_bytes, mime_type

    catalog_listing = _build_catalog_listing(catalog)
    catalog_block = (
        f"Here is our product catalog (format: product_id | name | category | price | keywords | description):\n\n{catalog_listing}\n\n"
        if catalog_listing else
        "No product catalog is available right now — leave matched_product_ids empty.\n\n"
    )

    prompt = (
        "Listen to this audio clip of a trade-show/event booth conversation.\n\n"
        + catalog_block +
        "Respond with ONLY a JSON object (no markdown fences, no extra text) in this exact shape:\n"
        '{"summary": "2-3 sentence summary of what the customer is interested in, their company/use '
        'case, and any specific needs, quantities, or timelines mentioned. No verbatim transcript.", '
        '"matched_product_ids": ["<product_id>", ...]}\n\n'
        "For matched_product_ids: only include product_ids from the catalog above that genuinely match "
        "what was discussed — infer from context and keywords, don't just string-match. Include at most "
        "5 products, ordered by relevance. If nothing in the catalog clearly matches, return an empty list."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": send_mime, "data": base64.b64encode(send_bytes).decode("utf-8")}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            # Newer Flash models "think" before answering, and those
            # thinking tokens are deducted from the SAME budget as the
            # actual output — with no cap set, the model can burn most of
            # it thinking and get cut off mid-JSON before finishing the
            # real response (exactly what produced the "Extra data" /
            # "Expecting ',' delimiter" JSON parse errors). Disabling
            # thinking avoids that outright; the explicit maxOutputTokens
            # is a second safety net in case thinking can't be fully
            # disabled for a given model.
            "thinkingConfig": {"thinkingBudget": 0},
            "maxOutputTokens": 2048,
        },
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
            finish_reason = data["candidates"][0].get("finishReason", "")
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            try:
                parsed = json.loads(raw_text)
            except ValueError as parse_exc:
                # Print the actual text that failed to parse (and why the
                # model stopped generating) instead of swallowing it —
                # this is what a truncated-JSON failure looks like.
                print(f"[Gemini analysis JSON parse failed, model={model}, finishReason={finish_reason}] "
                      f"{parse_exc} | raw text: {raw_text!r}")
                continue
            summary = str(parsed.get("summary", "")).strip()
            matched_ids = parsed.get("matched_product_ids", [])
            if not isinstance(matched_ids, list):
                matched_ids = []
            matched_ids = [str(pid).strip() for pid in matched_ids if str(pid).strip()]
            print(f"[Gemini analysis succeeded, model={model}, summary_len={len(summary)}, matches={matched_ids}]")
            return {"summary": summary, "matched_product_ids": matched_ids}
        except (requests.RequestException, KeyError, IndexError) as exc:
            error_body = getattr(getattr(exc, "response", None), "text", "")
            print(f"[Gemini analysis failed, model={model}] {exc} | response body: {error_body}")
            continue

    return empty_result


def resolve_matched_products(matched_product_ids: list, catalog: list) -> list:
    """Looks up full product details (name, price, image_url) for each
    matched product_id from the already-fetched catalog — never trusts
    Gemini to echo back accurate prices/URLs itself, only IDs."""
    catalog_by_id = {p.get("product_id"): p for p in catalog}
    resolved = []
    for pid in matched_product_ids:
        product = catalog_by_id.get(pid)
        if product:
            resolved.append({
                "product_id": product.get("product_id"),
                "name": product.get("name"),
                "price": product.get("price"),
                "image_url": product.get("image_url"),
            })
        else:
            print(f"[Matched product_id '{pid}' not found in catalog — skipping]")
    return resolved


# ----------------------------------------------------------------------------
# Apps Script + n8n
# ----------------------------------------------------------------------------
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
    matched_products: list,
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
        "transcript": transcript,
        "matched_products_json": json.dumps(matched_products),
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
        pass


@app.get("/")
def health_check():
    return {"status": "ok", "service": "event-lead-capture-api"}


@app.get("/api/diagnostics")
def diagnostics():
    return {
        "gemini_api_key_set": bool(GEMINI_API_KEY),
        "gemini_api_key_prefix": (GEMINI_API_KEY[:6] + "...") if GEMINI_API_KEY else None,
        "gemini_models": GEMINI_MODELS,
        "ffmpeg_available": _FFMPEG_AVAILABLE,
        "apps_script_url_set": bool(APPS_SCRIPT_URL),
        "gdrive_folder_id_set": bool(GDRIVE_FOLDER_ID),
        "n8n_webhook_url_set": bool(N8N_WEBHOOK_URL),
        "allowed_origins": ALLOWED_ORIGINS,
        "catalog_products_cached": len(_catalog_cache["products"]),
        "catalog_cache_age_seconds": round(time.time() - _catalog_cache["fetched_at"]) if _catalog_cache["fetched_at"] else None,
    }


@app.get("/api/catalog/refresh")
def refresh_catalog():
    """Manually force a catalog re-fetch — handy right after editing the
    Catalog sheet, instead of waiting up to CATALOG_CACHE_TTL_SECONDS."""
    products = fetch_catalog(force_refresh=True)
    return {"status": "ok", "products_cached": len(products)}


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

    catalog = fetch_catalog()

    analysis = analyze_audio_with_gemini(
        audio_bytes, audio.content_type or "audio/webm", catalog
    )
    transcript = analysis["summary"]
    matched_products = resolve_matched_products(analysis["matched_product_ids"], catalog)

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
        matched_products=matched_products,
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
            "matched_products": matched_products,
        }
    )

    return {
        "status": "success",
        "lead_id": lead_id,
        "audio_url": audio_link,
        "photo_url": photo_link,
        "transcript": transcript,
        "matched_products": matched_products,
    }
