"""
Whisper audio transcription endpoint.
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.schemas.transcription import TranscriptSegment, TranscriptionResponse
from src.services.audio_service import AudioService
from src.services.transcription_service import TranscriptionService

router = APIRouter()


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe phone call audio",
    description="Transcribes audio using OpenAI Whisper without executing fraud classification.",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file in WAV, MP3, FLAC, OGG, or M4A format"),
) -> TranscriptionResponse:
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    temp_path: Path = await AudioService.save_upload_to_temp(file)

    try:
        # Preprocess and normalize audio
        processed_path = AudioService.preprocess_call_audio(temp_path)

        # Transcribe
        result = TranscriptionService.transcribe(
            audio_path=processed_path,
            call_id=call_id,
        )

        segments_data = [
            TranscriptSegment(
                id=s.get("id"),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
            )
            for s in result.get("segments", [])
        ]

        return TranscriptionResponse(
            call_id=call_id,
            language=result["language"],
            transcript=result["transcript"],
            segments=segments_data,
            model_used=result["model_used"],
        )

    finally:
        temp_path.unlink(missing_ok=True)
