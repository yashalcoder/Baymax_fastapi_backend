from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.transcribe_service import process_transcription,process_medical_conversation,create_voice_embedding
from fastapi import Form
import tempfile
import json
import os
router = APIRouter(prefix="/transcribe", tags=["Transcription"])
@router.post("/audio")
async def transcribe_audio(file: UploadFile = File(...),doctorId: str = Form(...)):
    """
    Uploads audio file → Transcribes → Converts to Urdu → Translates to English
    """
    # Validate file format
    # file="../uploads/urdu_tes2_speaker.wav"
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
        print("Doctor ID received:", doctorId) 
        result = await process_transcription(file,doctorId)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    
@router.post("/enroll-doctor")
async def enroll_doctor(file: UploadFile = File(...)):
    """
    Create voice embedding for enrollment
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            print("\n🎤 Creating voice profile...")
            embedding = create_voice_embedding(tmp_path)
            
            return JSONResponse({
                "status": "success",
                "embedding": embedding,
                "metadata": {
                    "embedding_dimension": len(embedding),
                    "model": "facebook/wav2vec2-base",
                    "framework": "HuggingFace Transformers"
                }
            })
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/transcribe")
async def transcribe_conversation(
    file: UploadFile = File(...),
    doctor_embedding: str = Form(None),
    threshold: float = Form(0.70)
):
    """
    Transcribe conversation with voice verification
    """
    embedding = None
    if doctor_embedding:
        try:
            embedding = json.loads(doctor_embedding)
            print(f"\n✅ Received embedding: {len(embedding)}D")
        except:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid doctor_embedding format"}
            )
    
    result = await process_medical_conversation(file, embedding, threshold)
    return JSONResponse(content=result)


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "facebook/wav2vec2-base",
        "framework": "HuggingFace Transformers",
        "embedding_dim": 768
    }

