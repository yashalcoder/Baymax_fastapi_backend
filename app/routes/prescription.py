from fastapi import APIRouter
from app.schemas.patient import PatientInfo
from app.services.prescription import generatePrescription

router = APIRouter()

@router.post("/prescription")
async def generatePrescriptionAPI(patient_info: PatientInfo):
    return await generatePrescription(patient_info)

