# Voice AI Tutor

A local voice tutoring app with subject-routed specialist agents (English, History, Chemistry, Math).

## Setup

```powershell
pip install -e .
```

Install optional WaveGlow backend dependencies:

```powershell
pip install -e .[waveglow]
```

## Run Tutor

```powershell
python src/main.py
```

### Optional: Enable WaveGlow First

The TTS backend order is configurable with environment variable `TTS_BACKEND_ORDER`.

```powershell
$env:TTS_BACKEND_ORDER = "waveglow,kokoro,pyttsx3"
```

Optional WaveGlow runtime tuning:

```powershell
$env:WAVEGLOW_DEVICE = "cuda"
$env:WAVEGLOW_SIGMA = "0.8"
$env:WAVEGLOW_SAMPLE_RATE = "22050"
```

To keep responses low-latency, CPU WaveGlow is disabled by default.
If you explicitly want CPU WaveGlow, enable it:

```powershell
$env:WAVEGLOW_ALLOW_CPU = "1"
```

If WaveGlow cannot load models, the app automatically falls back to Kokoro and then pyttsx3.
If CUDA is not available, WaveGlow is skipped by default for lower latency.

Audio output follows your current Windows default output device automatically.
To pin a fixed output device index or name, set:

```powershell
$env:TTS_OUTPUT_DEVICE = "4"
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
