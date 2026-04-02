from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.symptom_extraction import extract_medical_information
router = APIRouter(prefix="/symptom")

@router.post("/extraction")
async def symptom_extract(transcript: str):
    try:
        result = extract_medical_information(transcript)
        return {"status": "success", "symptoms": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
