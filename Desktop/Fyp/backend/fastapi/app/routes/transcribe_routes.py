from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.transcribe_service import process_transcription
import os
router = APIRouter(prefix="/transcribe", tags=["Transcription"])
@router.post("/audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Uploads audio file → Transcribes → Converts to Urdu → Translates to English
    """
    # Validate file format
    allowed_formats = {'.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_formats:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error", 
                "message": f"Unsupported file format: {file_ext}. Supported: {', '.join(allowed_formats)}"
            }
        )
    
    try:
        result = await process_transcription(file)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )