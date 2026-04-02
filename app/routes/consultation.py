
from fastapi import APIRouter, Form, UploadFile, File, Depends
from app.services.consultation_pipeline import run_consultation_pipeline

router = APIRouter(prefix="/consultation",
    tags=["Consultation"]  )

@router.post("/start")
async def start_consultation(
    file: UploadFile = File(...),
    doctor_id: str =Form(...),
    patient_id: str =Form(...),
):
    result = await run_consultation_pipeline(file, doctor_id, patient_id)
    return result