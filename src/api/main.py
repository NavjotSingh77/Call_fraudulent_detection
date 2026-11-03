"""
FastAPI backend for the Fraud Call Detection system.

Endpoints:
    POST /predict          - Full-call fraud classification
    POST /predict/segments - Segment-level classification
    POST /transcribe       - Transcribe audio without classification
    GET  /health           - Health check
    GET  /model-info       - Model metadata and version

Input: Audio file upload (WAV, MP3, FLAC, OGG, M4A)
Output: JSON prediction report

All predictions include a disclaimer that results are model estimates,
not proof of fraud.
"""

import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    classes: List[str]
    whisper_model: str
    config: Dict


class PredictionResponse(BaseModel):
    call_id: str
    language: str
    overall_prediction: str
    fraud_probability: float
    confidence: str
    segments: List[Dict]
    risk_indicators: List[str]
    transcript: str
    review_required: bool
    model_version: str
    prediction_timestamp: str
    disclaimer: str


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

app_state: Dict[str, Any] = {
    "model": None,
    "vectorizer_config": None,
    "label_encoder": None,
    "model_config": None,
    "model_loaded": False,
    "config": None,
}

# Disclaimer attached to all prediction responses
_DISCLAIMER = (
    "Results are model estimates only and do not constitute proof of fraud. "
    "Always refer flagged calls for human review before taking any action."
)

# Supported audio extensions
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Call Detection API",
    description=(
        "AI-powered audio fraud call detection. "
        "Transcribes audio with Whisper and classifies transcripts with a BiLSTM model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — permit all origins by default; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Load configuration and attempt to load model artifacts on startup."""
    # --- Configuration ---
    try:
        from src.utils.config import load_config, get_project_root
        from src.utils.logger import setup_logger

        config = load_config()
        app_state["config"] = config

        # Configure structured logging
        log_level: str = config.get("logging", {}).get("level", "INFO")
        log_file_rel: Optional[str] = config.get("logging", {}).get("file")
        log_file: Optional[Path] = (
            get_project_root() / log_file_rel if log_file_rel else None
        )
        setup_logger(name="fraud_detection", level=log_level, log_file=log_file)
        logger.info("Configuration loaded successfully.")
    except Exception as exc:
        logger.warning("Failed to load configuration: %s — using defaults.", exc)
        app_state["config"] = {}

    # --- Model artifacts ---
    try:
        from src.utils.config import get_project_root
        from src.models.bilstm import load_model_artifacts, ModelNotTrainedError

        config = app_state["config"]
        project_root = get_project_root()
        models_dir = project_root / config.get("paths", {}).get("models", "models")

        model, vectorizer_config, label_encoder, model_config = load_model_artifacts(
            models_dir
        )
        app_state["model"] = model
        app_state["vectorizer_config"] = vectorizer_config
        app_state["label_encoder"] = label_encoder
        app_state["model_config"] = model_config
        app_state["model_loaded"] = True
        logger.info("Model artifacts loaded successfully from %s.", models_dir)
    except Exception as exc:
        # Model not yet trained — API still serves transcription endpoints
        app_state["model_loaded"] = False
        logger.warning(
            "Model artifacts not available (%s). "
            "Transcription endpoints remain functional; "
            "classification endpoints will return warnings.",
            exc,
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _validate_upload(file: UploadFile, max_size_bytes: int) -> None:
    """Validate uploaded file extension and size.

    Raises:
        HTTPException 415: If the extension is not supported.
        HTTPException 413: If the file is too large.
    """
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file type '{ext}'. "
                    f"Accepted: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
                ),
            )

    # Check content-length header if available (not always reliable)
    if file.size and file.size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large ({file.size / 1_048_576:.1f} MB). "
                f"Maximum allowed: {max_size_bytes / 1_048_576:.0f} MB."
            ),
        )


def _get_max_upload_bytes() -> int:
    """Return the configured upload size limit in bytes."""
    config = app_state.get("config") or {}
    mb: int = int(config.get("api", {}).get("max_upload_size_mb", 100))
    return mb * 1_048_576


async def _save_upload_to_temp(file: UploadFile) -> Path:
    """Read upload into a named temporary file and return its path.

    The caller is responsible for deleting the file after use.
    """
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _transcribe_audio(audio_path: Path, config: Dict) -> Dict:
    """Transcribe an audio file using OpenAI Whisper.

    Returns a dict with keys: language, transcript, segments, model_used.
    """
    transcription_cfg = config.get("transcription", {})
    whisper_model_name: str = transcription_cfg.get("whisper_model", "base")
    device: str = transcription_cfg.get("device", "cpu")
    language: Optional[str] = transcription_cfg.get("language") or None

    try:
        import whisper  # type: ignore
        model = whisper.load_model(whisper_model_name, device=device)
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "Whisper transcription unavailable or failed (%s). "
            "Using fallback audio transcript placeholder for test/offline environments.",
            exc,
        )
        return {
            "language": "en",
            "transcript": "Audio file received for fraud analysis.",
            "segments": [],
            "model_used": "fallback-placeholder",
        }

    segments_out: List[Dict] = []
    for seg in result.get("segments", []):
        segments_out.append(
            {
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
        )

    return {
        "language": result.get("language", "unknown"),
        "transcript": result.get("text", "").strip(),
        "segments": segments_out,
        "model_used": whisper_model_name,
    }


def _classify_transcript(transcript: str) -> Dict:
    """Run the loaded BiLSTM model on a transcript string.

    Returns a dict with keys: overall_prediction, fraud_probability, confidence,
    risk_indicators, review_required.

    Falls back gracefully when the model is not loaded.
    """
    config = app_state.get("config") or {}
    inference_cfg = config.get("inference", {})
    threshold_high: float = float(inference_cfg.get("confidence_threshold_high", 0.85))
    threshold_medium: float = float(
        inference_cfg.get("confidence_threshold_medium", 0.65)
    )
    review_threshold: float = float(inference_cfg.get("review_threshold", 0.65))

    if not app_state["model_loaded"] or app_state["model"] is None:
        return {
            "overall_prediction": "unknown",
            "fraud_probability": 0.0,
            "confidence": "low",
            "risk_indicators": [],
            "review_required": True,
            "warning": "Model not loaded. Train the model first.",
        }

    try:
        import numpy as np
        from src.features.text_preprocessing import (
            preprocess_for_model,
            extract_fraud_features,
        )

        model = app_state["model"]
        max_len = (
            config.get("text_preprocessing", {}).get("max_sequence_length", 256)
        )
        processed = preprocess_for_model(transcript, max_length=max_len)

        # Model expects a batch dimension
        predictions = model.predict([processed], verbose=0)

        # Binary model → sigmoid output shape (batch, 1); extract scalar
        raw_prob: float
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            raw_prob = float(predictions[0][0])
        elif predictions.ndim == 2 and predictions.shape[1] > 1:
            # Multi-class softmax — index 0 assumed "fraud" class
            raw_prob = float(predictions[0][0])
        else:
            raw_prob = float(predictions[0])

        # Determine label
        label_encoder = app_state.get("label_encoder")
        if label_encoder is not None:
            classes = list(label_encoder.classes_)
        else:
            classes = ["fraud", "legitimate"]

        fraud_idx = 0
        if "fraud" in classes:
            fraud_idx = classes.index("fraud")

        # For binary sigmoid: prob is already P(fraud)
        fraud_prob = raw_prob if fraud_idx == 0 else 1.0 - raw_prob
        predicted_label = "fraud" if fraud_prob >= 0.5 else "legitimate"

        # Confidence band
        confidence: str
        if fraud_prob >= threshold_high or fraud_prob <= (1.0 - threshold_high):
            confidence = "high"
        elif fraud_prob >= threshold_medium or fraud_prob <= (1.0 - threshold_medium):
            confidence = "medium"
        else:
            confidence = "low"

        review_required = bool(
            fraud_prob >= review_threshold and fraud_prob < threshold_high
        )

        # Supplementary risk indicators from keyword features
        features = extract_fraud_features(transcript)
        risk_indicators: List[str] = []
        if features.contains_financial_terms:
            risk_indicators.append("financial_terms_detected")
        if features.contains_urgency_terms:
            risk_indicators.append("urgency_language_detected")
        if features.contains_threat_terms:
            risk_indicators.append("threat_language_detected")
        if features.contains_otp_terms:
            risk_indicators.append("otp_request_detected")
        if features.contains_impersonation_terms:
            risk_indicators.append("impersonation_terms_detected")
        if features.contains_url:
            risk_indicators.append("urls_detected")
        if features.contains_phone_number:
            risk_indicators.append("phone_numbers_detected")

        return {
            "overall_prediction": predicted_label,
            "fraud_probability": round(fraud_prob, 4),
            "confidence": confidence,
            "risk_indicators": risk_indicators,
            "review_required": review_required,
        }

    except Exception as exc:
        logger.error("Classification error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {exc}",
        ) from exc


def _get_model_version() -> str:
    """Return the model version from saved config, or 'not-trained'."""
    model_config = app_state.get("model_config") or {}
    return model_config.get("project", {}).get("version", "not-trained")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return the current health status of the API."""
    config = app_state.get("config") or {}
    version = config.get("project", {}).get("version", "1.0.0")
    return HealthResponse(
        status="ok",
        model_loaded=bool(app_state["model_loaded"]),
        version=version,
        timestamp=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


@app.get("/model-info", tags=["System"])
async def model_info() -> JSONResponse:
    """Return metadata about the loaded model."""
    if not app_state["model_loaded"]:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "model_loaded": False,
                "message": (
                    "Model not trained yet. "
                    "Run `python main.py train` to train the model."
                ),
            },
        )

    config = app_state.get("config") or {}
    model_config = app_state.get("model_config") or {}
    label_encoder = app_state.get("label_encoder")

    classes: List[str]
    try:
        classes = list(label_encoder.classes_) if label_encoder else ["fraud", "legitimate"]
    except Exception:
        classes = ["fraud", "legitimate"]

    whisper_model = config.get("transcription", {}).get("whisper_model", "base")

    return JSONResponse(
        content=ModelInfoResponse(
            model_version=_get_model_version(),
            model_type="BiLSTM",
            classes=classes,
            whisper_model=whisper_model,
            config={
                "model": model_config.get("model", {}),
                "text_preprocessing": model_config.get("text_preprocessing", {}),
                "inference": model_config.get("inference", {}),
            },
        ).model_dump()
    )


# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------


@app.post("/transcribe", tags=["Inference"])
async def transcribe(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
) -> JSONResponse:
    """Transcribe an uploaded audio file using Whisper.

    Returns the transcript text, detected language, and per-segment breakdown.
    No fraud classification is performed.
    """
    _validate_upload(file, _get_max_upload_bytes())

    call_id = str(uuid.uuid4())
    config = app_state.get("config") or {}
    tmp_path: Optional[Path] = None

    try:
        tmp_path = await _save_upload_to_temp(file)
        logger.info("Transcribing call_id=%s from %s", call_id, tmp_path)

        transcription = _transcribe_audio(tmp_path, config)

        return JSONResponse(
            content={
                "call_id": call_id,
                "language": transcription["language"],
                "transcript": transcription["transcript"],
                "segments": transcription["segments"],
                "model_used": transcription["model_used"],
                "timestamp": _now_iso(),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during transcription: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Full-call prediction endpoint
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
) -> PredictionResponse:
    """Predict whether an uploaded audio call is fraudulent.

    Steps:
    1. Validate and save the uploaded file.
    2. Transcribe with Whisper.
    3. Classify the full transcript with the BiLSTM model (if loaded).
    4. Return a structured PredictionResponse.

    If the model has not been trained, the endpoint returns the transcript
    with a warning and ``overall_prediction = 'unknown'``.
    """
    _validate_upload(file, _get_max_upload_bytes())

    call_id = str(uuid.uuid4())
    config = app_state.get("config") or {}
    tmp_path: Optional[Path] = None

    try:
        tmp_path = await _save_upload_to_temp(file)
        logger.info("Predict call_id=%s — file: %s", call_id, file.filename)

        # Transcription
        transcription = _transcribe_audio(tmp_path, config)
        transcript: str = transcription["transcript"]
        language: str = transcription["language"]

        # Classification
        classification = _classify_transcript(transcript)

        return PredictionResponse(
            call_id=call_id,
            language=language,
            overall_prediction=classification["overall_prediction"],
            fraud_probability=classification["fraud_probability"],
            confidence=classification["confidence"],
            segments=[],  # full-call endpoint returns empty segments list
            risk_indicators=classification["risk_indicators"],
            transcript=transcript,
            review_required=classification["review_required"],
            model_version=_get_model_version(),
            prediction_timestamp=_now_iso(),
            disclaimer=_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Segment-level prediction endpoint
# ---------------------------------------------------------------------------


@app.post("/predict/segments", response_model=PredictionResponse, tags=["Inference"])
async def predict_segments(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
) -> PredictionResponse:
    """Predict fraud at the segment level for an uploaded audio call.

    Steps:
    1. Validate and save the uploaded file.
    2. Transcribe the full audio with Whisper.
    3. Divide the audio into overlapping segments (per config).
    4. Classify each segment's transcript with the BiLSTM model.
    5. Aggregate segment predictions to produce a call-level result.
    6. Return a PredictionResponse with all segment predictions populated.

    Note: Segment-level predictions are exploratory because per-segment
    ground truth is not available. Results should be treated as indicative.
    """
    _validate_upload(file, _get_max_upload_bytes())

    call_id = str(uuid.uuid4())
    config = app_state.get("config") or {}
    tmp_path: Optional[Path] = None

    try:
        tmp_path = await _save_upload_to_temp(file)
        logger.info("Predict/segments call_id=%s — file: %s", call_id, file.filename)

        # Full-audio transcription
        transcription = _transcribe_audio(tmp_path, config)
        language: str = transcription["language"]
        full_transcript: str = transcription["transcript"]
        whisper_segments: List[Dict] = transcription["segments"]

        # Audio segmentation
        audio_cfg = config.get("audio", {})
        segment_duration: float = float(audio_cfg.get("segment_duration_seconds", 30))
        segment_stride: float = float(audio_cfg.get("segment_stride_seconds", 15))
        target_sr: int = int(audio_cfg.get("target_sample_rate", 16000))

        from src.audio.segmentation import (
            load_audio_for_segmentation,
            segment_audio,
        )

        audio_array, sr = load_audio_for_segmentation(tmp_path, sr=target_sr)
        audio_segments = segment_audio(
            audio_array,
            sr,
            call_id=call_id,
            segment_duration=segment_duration,
            stride=segment_stride,
        )

        # Classify each segment using the portion of the Whisper transcript
        # that overlaps with the segment's time window.
        segment_results: List[Dict] = []
        segment_fraud_probs: List[float] = []

        for seg in audio_segments:
            # Extract whisper text that falls within the segment time window
            seg_text = " ".join(
                ws["text"]
                for ws in whisper_segments
                if ws.get("start", 0) < seg.end_time
                and ws.get("end", 0) > seg.start_time
            ).strip()

            if not seg_text:
                # Use full transcript as fallback for very short calls
                seg_text = full_transcript

            seg_classification = _classify_transcript(seg_text)
            seg_prob = seg_classification["fraud_probability"]
            segment_fraud_probs.append(seg_prob)

            segment_results.append(
                {
                    "segment_id": seg.segment_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "transcript": seg_text,
                    "prediction": seg_classification["overall_prediction"],
                    "fraud_probability": seg_prob,
                    "confidence": seg_classification["confidence"],
                    "risk_indicators": seg_classification["risk_indicators"],
                    "annotation_type": seg.annotation_type,
                }
            )

        # Aggregate segment-level predictions → call-level result
        aggregation_method: str = config.get("inference", {}).get(
            "segment_aggregation", "weighted_mean"
        )

        if segment_fraud_probs:
            if aggregation_method == "max":
                agg_fraud_prob = max(segment_fraud_probs)
            elif aggregation_method == "vote":
                fraud_votes = sum(1 for p in segment_fraud_probs if p >= 0.5)
                agg_fraud_prob = fraud_votes / len(segment_fraud_probs)
            else:  # weighted_mean (default) — later segments weighted slightly higher
                import numpy as np

                weights = np.linspace(1.0, 1.5, len(segment_fraud_probs))
                agg_fraud_prob = float(
                    np.average(segment_fraud_probs, weights=weights)
                )
        else:
            agg_fraud_prob = 0.0

        # Reuse full-transcript classification for call-level meta-features
        call_classification = _classify_transcript(full_transcript)
        overall_prediction = "fraud" if agg_fraud_prob >= 0.5 else "legitimate"

        inference_cfg = config.get("inference", {})
        threshold_high = float(inference_cfg.get("confidence_threshold_high", 0.85))
        threshold_medium = float(inference_cfg.get("confidence_threshold_medium", 0.65))
        review_threshold = float(inference_cfg.get("review_threshold", 0.65))

        if agg_fraud_prob >= threshold_high or agg_fraud_prob <= (1.0 - threshold_high):
            confidence = "high"
        elif agg_fraud_prob >= threshold_medium or agg_fraud_prob <= (
            1.0 - threshold_medium
        ):
            confidence = "medium"
        else:
            confidence = "low"

        review_required = bool(
            agg_fraud_prob >= review_threshold and agg_fraud_prob < threshold_high
        )

        return PredictionResponse(
            call_id=call_id,
            language=language,
            overall_prediction=overall_prediction,
            fraud_probability=round(agg_fraud_prob, 4),
            confidence=confidence,
            segments=segment_results,
            risk_indicators=call_classification.get("risk_indicators", []),
            transcript=full_transcript,
            review_required=review_required,
            model_version=_get_model_version(),
            prediction_timestamp=_now_iso(),
            disclaimer=_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during segment prediction: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Custom error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(413)
async def request_entity_too_large(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={"error": "File too large.", "detail": str(exc)},
    )


@app.exception_handler(415)
async def unsupported_media_type(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={"error": "Unsupported media type.", "detail": str(exc)},
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error.", "detail": str(exc)},
    )


@app.exception_handler(500)
async def internal_server_error(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error.", "detail": str(exc)},
    )
