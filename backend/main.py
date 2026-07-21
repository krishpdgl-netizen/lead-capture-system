"""
Test script — Batch transcribe audio from an Excel export
------------------------------------------------------------
Reads a column of Google Drive links from an .xlsx file (e.g. your Sheets
export from the lead capture system), downloads each audio file, runs it
through the same Groq Whisper transcribe + summarize pipeline as main.py,
and prints the results to the console. Also saves an output .xlsx with two
new columns (transcript, summary) alongside your original data.

Usage:
    export GROQ_API_KEY=gsk_...
    python test_transcription.py path/to/your_export.xlsx

Edit the two settings below to match your sheet's actual column name and
Drive link format if they differ.
"""

import os
import re
import sys

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Settings — adjust these to match your Excel file
# ----------------------------------------------------------------------------
AUDIO_URL_COLUMN = "audio_url"     # column header containing the Google Drive link
NAME_COLUMN = "name"               # column header used to label results (optional)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_WHISPER_MODELS = os.getenv("GROQ_WHISPER_MODELS", "whisper-large-v3-turbo,whisper-large-v3").split(",")
GROQ_SUMMARY_MODELS = os.getenv("GROQ_SUMMARY_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")

NO_SPEECH_MARKER = "(audio unclear — no reliable transcript)"
NO_SPEECH_PROB_THRESHOLD = 0.6

SUMMARY_PROMPT = (
    "You are given a verbatim transcript of an audio clip recorded at a "
    "trade-show/event booth. Summarize in 3-5 sentences what the customer "
    "is interested in, their company or use case, and any specific needs "
    "or timelines they mentioned. Only include details that are actually "
    "present in the transcript — never infer or assume anything that "
    "wasn't said. If the transcript is too short, garbled, or unrelated to "
    "a product/business conversation to summarize meaningfully, respond "
    "with exactly: (transcript too unclear to summarize)\n\n"
    "Respond with the summary only, as plain text with no headers or "
    "labels.\n\nTranscript:\n"
)


def drive_url_to_file_id(url: str) -> str:
    """Extracts the file ID from common Google Drive link formats:
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    Returns the input unchanged if no ID pattern is found (in case it's
    already a bare file ID)."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url.strip()


def download_drive_file(url: str) -> bytes:
    """Downloads a Google Drive file given its share URL, handling the
    'file too large to scan for viruses' confirmation page Drive shows for
    bigger files (like multi-minute audio recordings)."""
    file_id = drive_url_to_file_id(url)
    session = requests.Session()
    base = "https://drive.google.com/uc"

    resp = session.get(base, params={"id": file_id, "export": "download"}, stream=True, timeout=60)

    # Large files get an HTML confirmation page instead of the file itself,
    # with a confirm token embedded either in a cookie or the page body.
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token is None:
        match = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text or "")
        if match:
            token = match.group(1)

    if token:
        resp = session.get(base, params={"id": file_id, "export": "download", "confirm": token},
                            stream=True, timeout=120)

    resp.raise_for_status()
    return resp.content


def transcribe_audio_with_groq(audio_bytes: bytes, filename: str) -> str:
    """Same logic as main.py's transcribe_audio_with_groq — see that file
    for detailed comments on the no_speech_prob hallucination filtering."""
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
                files={"file": (filename, audio_bytes, "audio/webm")},
                data={"model": model, "response_format": "verbose_json"},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data.get("text", "").strip()
            segments = data.get("segments", [])

            if not text:
                print(f"  [Groq transcript — empty text, model={model}]")
                return NO_SPEECH_MARKER

            word_count = len(text.split())
            if segments and word_count <= 8:
                avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
                if avg_no_speech >= NO_SPEECH_PROB_THRESHOLD:
                    print(f"  [Groq transcript — high no_speech_prob ({avg_no_speech:.2f}) on "
                          f"{word_count}-word text, model={model}] discarding: {text!r}")
                    return NO_SPEECH_MARKER

            return text
        except (requests.RequestException, KeyError) as exc:
            error_body = getattr(getattr(exc, "response", None), "text", "")
            print(f"  [Groq transcript failed, model={model}] {exc} | response body: {error_body}")
            continue

    return "(Transcript generation failed)"


def summarize_transcript_with_groq(transcript: str) -> str:
    """Same logic as main.py's summarize_transcript_with_groq."""
    if not GROQ_API_KEY or not transcript or transcript == NO_SPEECH_MARKER:
        return transcript

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload_base = {
        "messages": [{"role": "user", "content": SUMMARY_PROMPT + transcript}],
        "temperature": 0.1,
    }

    for model in GROQ_SUMMARY_MODELS:
        model = model.strip()
        if not model:
            continue
        try:
            resp = requests.post(url, headers=headers, json={**payload_base, "model": model}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            summary = data["choices"][0]["message"]["content"].strip()
            if summary:
                return summary
        except (requests.RequestException, KeyError, IndexError) as exc:
            print(f"  [Groq summary failed, model={model}] {exc}")
            continue

    return transcript


def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python test_transcription.py path/to/your_export.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    df = pd.read_excel(excel_path)

    if AUDIO_URL_COLUMN not in df.columns:
        print(f"ERROR: column '{AUDIO_URL_COLUMN}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)

    transcripts = []
    summaries = []

    for i, row in df.iterrows():
        url = row.get(AUDIO_URL_COLUMN)
        label = row.get(NAME_COLUMN, f"row {i}") if NAME_COLUMN in df.columns else f"row {i}"

        if not isinstance(url, str) or not url.strip():
            print(f"[{label}] no audio URL — skipping")
            transcripts.append("")
            summaries.append("")
            continue

        print(f"[{label}] downloading...")
        try:
            audio_bytes = download_drive_file(url)
        except requests.RequestException as exc:
            print(f"[{label}] download failed: {exc}")
            transcripts.append("(download failed)")
            summaries.append("(download failed)")
            continue

        print(f"[{label}] transcribing ({len(audio_bytes) / 1024:.0f} KB)...")
        transcript = transcribe_audio_with_groq(audio_bytes, f"{label}.webm")
        print(f"[{label}] transcript: {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

        summary = summarize_transcript_with_groq(transcript)
        print(f"[{label}] summary: {summary}\n")

        transcripts.append(transcript)
        summaries.append(summary)

    df["transcript"] = transcripts
    df["summary"] = summaries

    out_path = excel_path.rsplit(".", 1)[0] + "_transcribed.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Done — results saved to {out_path}")


if __name__ == "__main__":
    main()
