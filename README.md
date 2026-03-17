# Voice AI Tutor

A local voice tutoring app with subject-routed specialist agents (English, History, Chemistry, Math).

## Setup

```powershell
pip install -e .
```

## Run Tutor

```powershell
python src/main.py
```

## Check Ingestion

```powershell
python -m vector
```

Fast status only (no embedding calls).

```powershell
python -m vector --ingest
```

Runs full ingestion using embeddings, then prints status.

If a PDF is image-only, create a sidecar OCR text file first.

Requirement: install Tesseract OCR on your machine, or provide `--tesseract-cmd`.

```powershell
python src/util/pdf_ocr.py --input assets/Algebra-2-Book.pdf
```

This creates `assets/Algebra-2-Book.ocr.txt`. Then run ingestion again:

```powershell
python -m vector --ingest
```

`vector.py` automatically uses `.ocr.txt` sidecar files when PDF text extraction is empty.

## Agent Voices

Per-agent voices are configured in `agent_voices.json`.

You can also set `VOICE_CONFIG_PATH` to load a different JSON file.
