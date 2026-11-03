# Fraud Call Detection System

> **Research Disclaimer:** Predictions produced by this system are **probabilistic model estimates**, not legal or evidentiary proof of fraud. All flagged calls should be reviewed by a qualified human analyst before any action is taken.

---

## Overview

The Fraud Call Detection System is a research-grade, end-to-end machine learning pipeline that automatically identifies fraudulent telephone calls from raw audio recordings. It transcribes call audio using OpenAI Whisper, extracts semantic features from the resulting transcript, and classifies the call as **fraud** or **legitimate** using a Bidirectional LSTM (BiLSTM) neural network.

### Key Capabilities

| Capability | Detail |
|---|---|
| **Audio Ingestion** | WAV, MP3, FLAC, OGG, M4A — resampled to 16 kHz mono |
| **Automatic Speech Recognition** | OpenAI Whisper (tiny → large); language auto-detected |
| **Multilingual Support** | English, Hindi, Hinglish — Whisper handles multilingual natively |
| **Binary Classification** | Fraud / Legitimate via BiLSTM trained on transcript text |
| **Segment-level Analysis** | Long calls split into 30-second overlapping windows |
| **Probability Calibration** | Confidence tiers: High / Medium / Requires Review |
| **REST API** | FastAPI endpoint for real-time inference |
| **Reproducibility** | Fixed seed, group-aware splits, all hyperparameters in `config.yaml` |

### Research Validity Note

> This system was developed for **research and educational purposes**. Fraud probability scores are outputs of a statistical model trained on a specific robocall dataset. They reflect learned linguistic patterns — not verified malicious intent. **Do not use predictions as standalone evidence of fraud.**

---

## Architecture

The system is organized as a sequential pipeline:

```
Audio Input → Audio Preprocessing → Whisper ASR → Transcript Processing →
Feature Extraction → BiLSTM Classification → Probability Calibration →
Decision Engine → Prediction Report
```

| Stage | Module | Description |
|---|---|---|
| Audio Preprocessing | `src/audio/preprocessing.py` | Resample, normalize, silence check |
| Segmentation | `src/audio/segmentation.py` | Split long calls into overlapping windows |
| ASR | Whisper (via `openai-whisper`) | Speech-to-text transcription |
| Text Preprocessing | `src/features/text_preprocessing.py` | Tokenization, vocabulary building |
| Classification | `src/models/bilstm.py` | Bidirectional LSTM classifier |
| Inference & Calibration | `src/models/inference.py` | Probability thresholds, segment aggregation |
| API | `src/api/main.py` | FastAPI REST endpoint |

---

## Dataset

### NCSU Robocall Audio Dataset

The primary training corpus is the **[NCSU Robocall Audio Dataset](https://github.com/wspr-ncsu/robocall-audio-dataset)** published by North Carolina State University (NCSU) in collaboration with the FTC.

| Property | Value |
|---|---|
| **Source** | NCSU / FTC Enforcement Actions |
| **Content** | Real robocall audio recordings |
| **Language** | Primarily English |
| **Label** | `fraud` (all NCSU samples are verified fraud) |
| **Label Status** | `verified` — sourced from FTC enforcement actions |
| **Access** | Publicly available via GitHub |

### Label Strategy

```
Label Status Hierarchy
──────────────────────
verified   → NCSU / FTC calls (confirmed fraud from enforcement records)
weak       → Heuristically labeled from external sources
unknown    → Unlabeled or unverified external data
```

All NCSU recordings carry `label=fraud` and `label_status=verified` because they were collected as part of FTC enforcement proceedings — making them the highest-quality fraud signal available.

### Why Legitimate Calls Require External Data

The NCSU dataset contains **only fraud calls**. Training a binary classifier (fraud vs. legitimate) requires examples of both classes. The NCSU data alone can only train a one-class model, which is insufficient for reliable binary discrimination.

**To train a meaningful classifier you must supply legitimate call audio.** See [Adding Legitimate Call Data](#adding-legitimate-call-data) below.

---

## Directory Structure

```
Call_fraudulent_detection/
├── config.yaml                        # Central configuration (all hyperparameters)
├── requirements.txt                   # Python dependencies
├── main.py                            # CLI entry point
├── README.md
│
├── data/
│   ├── raw/                           # Downloaded NCSU audio (git-ignored)
│   ├── external/                      # User-supplied external data
│   │   └── hindi_hinglish/            # Optional Hindi/Hinglish data
│   ├── processed/
│   │   ├── audio/                     # Preprocessed 16 kHz WAV files
│   │   └── transcripts/               # Whisper transcript JSON files
│   ├── metadata/
│   │   └── manifest.csv               # Master record of all samples + labels
│   └── splits/                        # train.csv / val.csv / test.csv
│
├── models/                            # Saved model artefacts (git-ignored)
│   ├── model.keras                    # Trained BiLSTM weights
│   ├── vectorizer_config.json         # TextVectorization vocabulary
│   ├── label_encoder.pkl              # Fitted label encoder
│   └── model_config.json              # Hyperparameter snapshot
│
├── notebooks/
│   └── 01_data_exploration.md         # Data exploration guide
│
├── reports/
│   ├── evaluation/                    # Evaluation outputs and reports
│   └── figures/                       # Plots (ROC, PR curves, etc.)
│
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI application
│   ├── audio/
│   │   ├── preprocessing.py           # Resampling, normalization
│   │   └── segmentation.py            # Sliding-window segmentation
│   ├── data/
│   │   ├── ingestion.py               # NCSU dataset download
│   │   ├── dataset_loader.py          # Manifest and split loading
│   │   ├── dataset_validator.py       # Data integrity checks
│   │   ├── labeling.py                # Label assignment logic
│   │   └── splitting.py               # Group-aware train/val/test splits
│   ├── features/
│   │   └── text_preprocessing.py      # Tokenization, cleaning
│   ├── models/
│   │   ├── bilstm.py                  # Model definition + save/load
│   │   └── inference.py               # Prediction + calibration
│   ├── training/                      # Training loop
│   └── utils/
│       ├── config.py                  # YAML config loader
│       └── logger.py                  # Structured logging
│
└── tests/
    ├── fixtures/                      # Test audio and transcript fixtures
    └── test_text_preprocessing.py     # Unit tests
```

---

## Installation

### Requirements

- **Python 3.11+**
- **FFmpeg** — required for audio decoding (MP3, M4A, OGG, FLAC)
- **CUDA 11.8+** *(optional)* — for GPU-accelerated Whisper transcription and model training
- ~4 GB disk space for the base Whisper model and dataset

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Call_fraudulent_detection.git
cd Call_fraudulent_detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### FFmpeg Installation (Windows)

FFmpeg must be on your system `PATH` for audio decoding to work.

**Option A — winget (Windows 10/11):**
```powershell
winget install Gyan.FFmpeg
```

**Option B — Chocolatey:**
```powershell
choco install ffmpeg
```

**Option C — Manual:**
1. Download a static build from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your system `PATH`
4. Verify: `ffmpeg -version`

---

## Configuration

All behaviour is controlled by [`config.yaml`](config.yaml). Key sections:

| Section | Purpose |
|---|---|
| `project` | Name, version, `random_seed` (default 42) |
| `paths` | All data, model, and report directories |
| `dataset.ncsu` | NCSU repo URL, GitHub base, label metadata |
| `dataset.external` | External data sources (Hindi/Hinglish toggle) |
| `audio` | Sample rate, channels, duration limits, segment sizing |
| `transcription` | Whisper model size, device (`cpu`/`cuda`), language |
| `text_preprocessing` | Vocabulary size, sequence length, tokenization options |
| `model` | Embedding dim, LSTM units, dropout, learning rate, epochs |
| `training` | Split ratios, class weighting, checkpoint metric |
| `inference` | Confidence thresholds, segment aggregation method |
| `api` | Host, port, upload size limit |
| `logging` | Log level and output file |

**Key thresholds:**
```yaml
inference:
  confidence_threshold_high: 0.85    # High-confidence fraud flag
  confidence_threshold_medium: 0.65  # Medium-confidence, warrants review
  review_threshold: 0.65             # Below this → human review required
```

---

## Usage

### Step-by-Step Workflow

#### 1. Download Dataset

Downloads NCSU robocall audio from GitHub:

```bash
python main.py download-data

# Limit to N files (useful for quick testing):
python main.py download-data --max-files 20
```

#### 2. Validate Data

Checks audio file integrity, duration bounds, and metadata consistency:

```bash
python main.py validate-data
```

#### 3. Preprocess Audio

Resamples all audio to 16 kHz mono, normalizes amplitude, and removes files that are too short or silent:

```bash
python main.py preprocess
```

#### 4. Transcribe Audio

Runs Whisper ASR on preprocessed audio. Transcripts are cached in `data/processed/transcripts/`:

```bash
python main.py transcribe

# Use a larger Whisper model for better accuracy:
python main.py transcribe --model small
python main.py transcribe --model medium
```

Available Whisper model sizes: `tiny`, `base`, `small`, `medium`, `large`
*(Larger models are slower but more accurate, especially for accented or multilingual speech.)*

#### 5. Prepare Dataset

Builds `data/metadata/manifest.csv` and generates stratified train / val / test splits in `data/splits/`:

```bash
python main.py prepare-dataset
```

> **⚠️ Class Balance Warning**
>
> The NCSU dataset contains **only fraud calls**. A binary fraud/legitimate classifier cannot be trained on a single-class dataset — the model would have no negative examples to learn from. Before running `train`, you **must** supply legitimate call audio. See [Adding Legitimate Call Data](#adding-legitimate-call-data).

#### 6. Train Model

Trains the BiLSTM classifier with early stopping, learning rate reduction, and best-checkpoint saving:

```bash
python main.py train
```

Training checkpoints are saved to `models/`. The best model (by `val_recall`) is preserved automatically.

#### 7. Evaluate

Evaluates the best saved model and writes a metrics report to `reports/evaluation/`:

```bash
# Evaluate on the held-out test split (default):
python main.py evaluate

# Evaluate on the validation split:
python main.py evaluate --split val
```

#### 8. Predict

Run inference on a new audio file:

```bash
# Full-call prediction:
python main.py predict --audio path/to/call.wav

# Segment-level breakdown (for long calls):
python main.py predict --audio path/to/call.wav --segments
```

Output includes:
- Overall fraud probability
- Confidence tier (`HIGH` / `MEDIUM` / `REQUIRES_REVIEW`)
- Per-segment scores (with `--segments`)
- Detected language

#### 9. Launch API

Start the FastAPI REST server:

```bash
python main.py serve

# Or directly with uvicorn (supports hot-reload for development):
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### API Usage

Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict from audio file:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@path/to/call.wav"
```

**Predict with segment-level breakdown:**
```bash
curl -X POST "http://localhost:8000/predict?segments=true" \
     -H "accept: application/json" \
     -F "file=@path/to/call.wav"
```

**Example response:**
```json
{
  "fraud_probability": 0.923,
  "confidence_tier": "HIGH",
  "label": "fraud",
  "language": "en",
  "transcript_preview": "This is the IRS calling about your outstanding tax liability...",
  "segments": null
}
```

---

## Adding Legitimate Call Data

### The Class Imbalance Problem

Binary classification requires two classes. The NCSU dataset provides only fraud examples. Without legitimate (non-fraud) call audio, the model cannot learn to distinguish fraud from normal speech — it can only overfit to the fraud class.

### Recommended Sources of Legitimate Audio

| Source | Description | URL |
|---|---|---|
| **Mozilla Common Voice** | Crowdsourced read speech, many languages | https://commonvoice.mozilla.org |
| **VCTK Corpus** | 110 English speakers, various accents | https://datashare.ed.ac.uk/handle/10283/2950 |
| **LibriSpeech** | 1000 hours of English audiobook speech | https://www.openslr.org/12 |
| **VoxForge** | GPL-licensed speech for ASR training | http://www.voxforge.org |

### How to Add Legitimate Data

1. Place audio files in `data/external/<source_name>/`
2. Ensure files are WAV, MP3, FLAC, OGG, or M4A format
3. Extend `data/metadata/manifest.csv` with rows:

```csv
file_id,audio_path,label,label_status,source,language
leg_001,data/external/librispeech/file001.wav,legitimate,weak,librispeech,en
leg_002,data/external/librispeech/file002.wav,legitimate,weak,librispeech,en
```

4. Re-run `python main.py prepare-dataset` to rebuild splits with the new data.

> **Label Status:** Externally sourced legitimate calls carry `label_status=weak` because they have not been verified by enforcement proceedings — they are presumed legitimate, not confirmed.

---

## Adding Hindi/Hinglish Data

The `config.yaml` includes a dedicated extension point for Hindi and Hinglish call data:

```yaml
dataset:
  external:
    hindi_hinglish:
      enabled: false          # Set to true to include this source
      path: "data/external/hindi_hinglish"
      label_status: "unknown"
```

**Steps:**
1. Set `enabled: true` in `config.yaml`
2. Place Hindi/Hinglish audio in `data/external/hindi_hinglish/`
3. Extend the manifest with appropriate `language` codes (`hi`, `hi-en`)
4. Re-run `prepare-dataset` and `train`

> **Note:** Whisper detects and transcribes Hindi and Hinglish automatically — no language-specific configuration is needed for the ASR step. The `language: null` setting in `config.yaml` enables auto-detection.

---

## Model Architecture

The classifier is a **Bidirectional LSTM (BiLSTM)** operating on call transcripts. All hyperparameters are defined in `config.yaml` — nothing is hardcoded.

```
Input (raw text string)
       │
       ▼
TextVectorization  ──► integer token IDs, padded to max_sequence_length (256)
       │
       ▼
Embedding Layer    ──► dense vectors  [vocab_size=20000, embedding_dim=128]
       │
       ▼
Bidirectional LSTM ──► concatenated forward + backward hidden states [units=128]
       │
       ▼
Dropout (0.30)
       │
       ▼
Dense + ReLU       ──► [units=64]
       │
       ▼
Dropout (0.15)
       │
       ▼
Output Layer
  ├── Binary (num_classes=2): Dense(1, sigmoid)  → fraud probability ∈ [0,1]
  └── Ternary (num_classes=3): Dense(3, softmax) → [fraud, legitimate, uncertain]
```

**Saved artefacts** (`models/`):

| File | Contents |
|---|---|
| `model.keras` | Full Keras model with weights |
| `vectorizer_config.json` | TextVectorization vocabulary |
| `label_encoder.pkl` | Fitted sklearn LabelEncoder |
| `model_config.json` | Hyperparameter snapshot at training time |

---

## Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **Recall (Sensitivity)** | Primary metric — minimizes missed fraud (false negatives are costly) |
| **Precision** | Fraction of fraud predictions that are correct |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Overall discriminative power across all thresholds |
| **PR-AUC** | Performance under class imbalance (more informative than ROC when classes are skewed) |
| **Accuracy** | Overall correctness (secondary — misleading under imbalance) |
| **Confusion Matrix** | Absolute counts of TP, TN, FP, FN |

> **Why recall is prioritized:** In fraud detection, a missed fraud call (false negative) is typically more harmful than a false alarm (false positive). The model checkpoint is saved at peak `val_recall`. The `early_stopping_patience` and threshold tuning allow precision/recall trade-offs to be adjusted post-training.

---

## Limitations and Disclaimers

1. **Predictions are estimates, not proof.** The model assigns a fraud *probability* based on learned linguistic patterns. This is not legal evidence of fraud.

2. **Dataset scope.** The NCSU dataset consists of robocalls only. Generalization to other fraud types (live scam calls, vishing, SIM swap calls) is **unknown** and not validated.

3. **Legitimate call class required.** The NCSU dataset has no legitimate examples. A meaningful binary classifier **cannot** be trained without adding legitimate call data.

4. **Transcript quality matters.** Classification accuracy depends entirely on Whisper ASR quality. Poor audio (low SNR, heavy accent, heavy background noise) degrades transcription and therefore classification performance.

5. **Language coverage.** The model is trained on whatever transcripts are in the training set. If Hindi/Hinglish data is not added, the model may not generalize to those languages.

6. **Not validated for production.** This system has not undergone the rigorous validation required for deployment in a production fraud-enforcement context. Human review is mandatory.

---

## Windows Automated Execution Script (`run_system.bat`)

The system includes a dedicated Windows automation script (`run_system.bat`) designed for zero-friction setup, model management, and server execution.

### Features
- **Virtual Environment Automation:** Automatically checks for, creates, and activates `.venv`.
- **Dependency Management:** Automatically installs and verifies packages from `requirements.txt`.
- **Intelligent Training Prevention:** Does **not** retrain an existing model if valid production artifacts (`model.keras`, `vectorizer_config.json`, `label_encoder.pkl`, `model_config.json`) already exist in `models/`, unless explicitly forced with `--force-train`.
- **Automatic Browser Launch:** Polls `/api/v1/health` until the API is fully ready and automatically opens the Web UI dashboard in your default browser.
- **Support for Modes:** `all`, `train`, `serve`, `test`.

### Quick Windows Commands

```cmd
:: 1. Complete Workflow: Setup, Train (if needed), Launch Gateway & Web UI
run_system.bat

:: Or explicitly:
run_system.bat all

:: 2. Force Retraining of the Model (even if model.keras already exists):
run_system.bat all --force-train

:: 3. Train & Evaluate the BiLSTM Model only (does not start server):
run_system.bat train

:: 4. Force Retrain only:
run_system.bat train --force-train

:: 5. Launch FastAPI Gateway & Web UI immediately (serve only):
run_system.bat serve

:: 6. Run the 78-test Unit & Integration Test Suite:
run_system.bat test
```

---

## Responsive Web UI Dashboard

FastAPI serves a clean, responsive, dark-mode dashboard at `http://127.0.0.1:8000/` (or `/ui`):

- **Drag-and-Drop Audio Upload:** Accepts `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a` files with client-side audio player preview.
- **Analysis Modes:** Toggle between **Full-Call Classification** and **Segment-Level Analysis** (30-second windows with 15-second stride).
- **KPI Metrics:** Displays overall classification (`FRAUD`, `LEGITIMATE`, `UNCERTAIN`), color-coded probability gauge, confidence tier, and analyst review status.
- **Transcript Viewer:** Displays full call transcription, detected language code, and one-click copy to clipboard.
- **Linguistic Risk Flags:** Highlights detected threats, urgency, impersonation terms, financial payment demands, or OTP requests.
- **Segment Timeline Table:** Detailed table showing start, end, duration, exploratory provenance, and fraud score for each 30-second window.
- **API Error Inspection:** User-friendly alert banner with full error details and troubleshooting hints when API errors occur.

---

## Modular Backend & Gateway Architecture

The API backend is designed with strict enterprise separation of concerns so that future Android mobile applications and relational databases (PostgreSQL, SQLite) can be integrated without modifying the ML inference engine:

```text
src/
├── api/
│   ├── main.py              # Main FastAPI app, lifespan, CORS, and middleware
│   ├── middleware.py        # Correlation ID, telemetry timing, and API-key auth
│   ├── dependencies.py      # Dependency injection (DB session, repository, model service)
│   ├── v1/
│   │   ├── router.py        # Prefix /api/v1 aggregation
│   │   └── endpoints/
│   │       ├── health.py    # GET /api/v1/health
│   │       ├── model_info.py# GET /api/v1/model-info
│   │       ├── predict.py   # POST /api/v1/predict, POST /api/v1/predict/segments
│   │       └── transcribe.py# POST /api/v1/transcribe
│   └── web/
│       └── router.py        # Serves dashboard HTML template at / and /ui
├── schemas/                 # Pydantic v2 data transfer objects (API contracts)
│   ├── common.py            # HealthResponse, ErrorResponse
│   ├── prediction.py        # PredictionResponse, SegmentResult
│   ├── transcription.py     # TranscriptionResponse, TranscriptSegment
│   └── model_info.py        # ModelInfoResponse
├── services/                # Business & ML service layer
│   ├── model_service.py     # Dedicated ModelService (loads .keras; isolated from training code)
│   ├── audio_service.py     # Format validation, streaming temp storage, preprocessing
│   └── transcription_service.py # Whisper execution and filesystem caching
├── db/                      # Database connection and ORM models
│   ├── session.py           # Engine & SessionLocal (SQLite / PostgreSQL)
│   ├── base.py              # DeclarativeBase
│   └── models.py            # CallRecord and SegmentRecord tables
├── repositories/            # Repository pattern (data access layer)
│   ├── base.py              # IRepository generic interface
│   └── call_repository.py   # CallRepository CRUD & prediction log persistence
├── config/
│   └── settings.py          # Environment settings (.env) & config.yaml loader
└── web/                     # Frontend templates and static assets
    ├── templates/index.html # Dashboard HTML
    └── static/              # CSS styles and JavaScript logic
```

### Future Android App Integration
- **Identical REST Contract:** The Android app can call `POST /api/v1/predict` (multipart form audio upload) and receive the exact same structured JSON contract.
- **Authentication Ready:** Set `APP_API_KEY_ENABLED=true` in `.env` and pass `X-API-Key: your_key` or `Authorization: Bearer your_key` from the Android OkHttp/Retrofit client.
- **CORS Configured:** Add your Android app domain or local network IP in `APP_CORS_ORIGINS`.

### Database Configuration (PostgreSQL Ready)
Configure `DATABASE_URL` in `.env`:
```env
# Embedded default:
DATABASE_URL=sqlite:///./data/fraud_calls.db

# Production PostgreSQL:
# DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/fraud_db
```

---

## Testing

```bash
# Run the full test suite with verbose output:
python -m pytest tests/ -v

# Run a specific test module:
python -m pytest tests/test_text_preprocessing.py -v

# Run with coverage report:
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

Test fixtures (small audio samples, synthetic transcripts) are stored in `tests/fixtures/`.

---

## Reproducibility

| Mechanism | Detail |
|---|---|
| **Fixed random seed** | `project.random_seed: 42` in `config.yaml` — propagated to NumPy, TensorFlow, and Python `random` |
| **Group-aware splitting** | `training.group_split: true` prevents audio from the same call or source appearing in both train and test sets (prevents data leakage) |
| **Config snapshot** | `models/model_config.json` records the exact hyperparameters used for each training run |
| **Deterministic splits** | `data/splits/` CSVs are saved and versioned alongside the model artefacts |

---

## License and Ethics

### Dataset License

The NCSU Robocall Audio Dataset is publicly available from [NCSU / FTC](https://github.com/wspr-ncsu/robocall-audio-dataset). Refer to that repository for its specific license terms.

### Responsible Use

This system is provided for **research and educational purposes only**. Users are responsible for:

- Complying with applicable wiretapping, privacy, and data protection laws in their jurisdiction before recording or analyzing calls.
- Not using model predictions as sole grounds for any enforcement, legal, or disciplinary action.
- Maintaining human oversight in any operational deployment.
- Protecting the privacy of individuals whose calls may be analyzed.

> **Do not deploy this system in a production environment without independent validation, legal review, and appropriate human oversight mechanisms.**
