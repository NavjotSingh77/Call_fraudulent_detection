"""
Full-call and segment-level fraud prediction endpoints.
"""

import uuid
from pathlib import Path
from typing import List
import soundfile as sf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.dependencies import get_call_repo, get_model
from src.audio.segmentation import segment_audio
from src.config.settings import get_settings
from src.models.inference import aggregate_segment_predictions
from src.repositories.call_repository import CallRepository
from src.schemas.prediction import PredictionResponse, SegmentResult
from src.services.audio_service import AudioService
from src.services.model_service import ModelNotReadyException, ModelService
from src.services.transcription_service import TranscriptionService

router = APIRouter()
settings = get_settings()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraudulent likelihood for an audio call",
    description="Processes full call audio, transcribes with Whisper, and classifies with the BiLSTM model.",
)
async def predict_call(
    file: UploadFile = File(..., description="Phone call audio (WAV, MP3, FLAC, OGG, M4A)"),
    model_svc: ModelService = Depends(get_model),
    call_repo: CallRepository = Depends(get_call_repo),
) -> PredictionResponse:
    AudioService.validate_file_metadata(file)
    if not model_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded or trained. Predictions cannot be fabricated. "
                "Train the model first using: python main.py train (or run_system.bat train)"
            ),
        )

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    filename = file.filename
    temp_path: Path = await AudioService.save_upload_to_temp(file)

    try:
        # 1. Audio validation and duration
        info = AudioService.get_audio_info(temp_path)
        duration = float(info.get("duration_seconds", 0.0))
        sample_rate = int(info.get("sample_rate", 16000))

        # 2. Preprocess audio
        processed_path = AudioService.preprocess_call_audio(temp_path)

        # 3. Whisper transcription
        trans_res = TranscriptionService.transcribe(
            audio_path=processed_path,
            call_id=call_id,
        )
        transcript = trans_res.get("transcript", "")
        language = trans_res.get("language", "en")

        # 4. Model inference via dedicated model service
        pred = model_svc.predict(transcript)

        # 5. Persist to database repository
        try:
            call_repo.save_prediction(
                call_id=call_id,
                filename=filename,
                duration_seconds=duration,
                sample_rate=sample_rate,
                language=language,
                transcript=transcript,
                overall_prediction=pred["overall_prediction"],
                fraud_probability=pred["fraud_probability"],
                confidence=pred["confidence"],
                review_required=pred["review_required"],
                risk_indicators=pred["risk_indicators"],
                model_version=pred["model_version"],
                client_source="api_v1",
            )
        except Exception:
            # Database logging error should not crash prediction response
            pass

        return PredictionResponse(
            call_id=call_id,
            language=language,
            overall_prediction=pred["overall_prediction"],
            fraud_probability=pred["fraud_probability"],
            confidence=pred["confidence"],
            segments=[],
            risk_indicators=pred["risk_indicators"],
            transcript=transcript,
            review_required=pred["review_required"],
            model_version=pred["model_version"],
            disclaimer=settings.disclaimer,
        )

    finally:
        temp_path.unlink(missing_ok=True)


@router.post(
    "/predict/segments",
    response_model=PredictionResponse,
    summary="Segment-level fraud prediction",
    description="Divides audio into overlapping 30-second windows with 15-second stride, evaluating each window individually.",
)
async def predict_call_segments(
    file: UploadFile = File(..., description="Phone call audio (WAV, MP3, FLAC, OGG, M4A)"),
    model_svc: ModelService = Depends(get_model),
    call_repo: CallRepository = Depends(get_call_repo),
) -> PredictionResponse:
    AudioService.validate_file_metadata(file)
    if not model_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded or trained. Predictions cannot be fabricated. "
                "Train the model first using: python main.py train (or run_system.bat train)"
            ),
        )

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    filename = file.filename
    temp_path: Path = await AudioService.save_upload_to_temp(file)

    try:
        info = AudioService.get_audio_info(temp_path)
        duration = float(info.get("duration_seconds", 0.0))
        sample_rate = int(info.get("sample_rate", 16000))

        processed_path = AudioService.preprocess_call_audio(temp_path)

        # Full transcription
        trans_res = TranscriptionService.transcribe(
            audio_path=processed_path,
            call_id=call_id,
        )
        full_transcript = trans_res.get("transcript", "")
        language = trans_res.get("language", "en")
        whisper_segments = trans_res.get("segments", [])

        # Segment audio array
        audio_array, sr = sf.read(str(processed_path))
        cfg_dict = settings.load_yaml_config()
        seg_duration = cfg_dict.get("audio", {}).get("segment_duration_seconds", 30.0)
        seg_stride = cfg_dict.get("audio", {}).get("segment_stride_seconds", 15.0)

        audio_segs = segment_audio(
            audio=audio_array,
            sr=sr,
            call_id=call_id,
            segment_duration=seg_duration,
            stride=seg_stride,
        )

        segment_results: List[SegmentResult] = []
        raw_seg_dicts = []

        for seg in audio_segs:
            # Match whisper text within segment time
            seg_text = " ".join(
                ws.get("text", "")
                for ws in whisper_segments
                if ws.get("start", 0) < seg.end_time and ws.get("end", 0) > seg.start_time
            ).strip()

            if not seg_text:
                seg_text = full_transcript

            seg_pred = model_svc.predict(seg_text)
            seg_prob = seg_pred["fraud_probability"]

            seg_res = SegmentResult(
                segment_id=seg.segment_id,
                start_time=round(seg.start_time, 2),
                end_time=round(seg.end_time, 2),
                duration=round(seg.duration, 2),
                fraud_probability=seg_prob,
                transcript_snippet=seg_text[:120] if seg_text else None,
                annotation_type=seg.annotation_type,
            )
            segment_results.append(seg_res)
            raw_seg_dicts.append({
                "segment_id": seg.segment_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "fraud_probability": seg_prob,
                "transcript_snippet": seg_text,
                "annotation_type": seg.annotation_type,
            })

        # Aggregate segment predictions into overall call prediction
        agg_info = aggregate_segment_predictions(raw_seg_dicts, cfg_dict)
        overall_prob = round(float(agg_info["fraud_probability"]), 4)

        th_high = cfg_dict.get("inference", {}).get("confidence_threshold_high", 0.75)
        th_med = cfg_dict.get("inference", {}).get("confidence_threshold_medium", 0.45)

        if overall_prob >= th_high:
            confidence = "high"
            overall = "fraud"
        elif overall_prob >= th_med:
            confidence = "medium"
            overall = "uncertain"
        else:
            confidence = "low" if overall_prob > 0.25 else "high"
            overall = "legitimate"

        # Risk indicators on full transcript
        full_pred = model_svc.predict(full_transcript)
        risk_indicators = full_pred["risk_indicators"]
        review_required = overall == "uncertain" or confidence != "high"

        # Persist to repository
        try:
            call_repo.save_prediction(
                call_id=call_id,
                filename=filename,
                duration_seconds=duration,
                sample_rate=sample_rate,
                language=language,
                transcript=full_transcript,
                overall_prediction=overall,
                fraud_probability=overall_prob,
                confidence=confidence,
                review_required=review_required,
                risk_indicators=risk_indicators,
                model_version=model_svc.get_version(),
                client_source="api_v1_segments",
                segments=raw_seg_dicts,
            )
        except Exception:
            pass

        return PredictionResponse(
            call_id=call_id,
            language=language,
            overall_prediction=overall,
            fraud_probability=overall_prob,
            confidence=confidence,
            segments=segment_results,
            risk_indicators=risk_indicators,
            transcript=full_transcript,
            review_required=review_required,
            model_version=model_svc.get_version(),
            disclaimer=settings.disclaimer,
        )

    finally:
        temp_path.unlink(missing_ok=True)
