from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ExtractedData(BaseModel):
    extracted_text: Optional[str] = None
    medical_terms: List[str] = []
    medicines: List[str] = []
    diagnoses: List[str] = []
    doctor_notes: Optional[str] = None   # ✅ NEW FIELD


class MedicalReport(BaseModel):
    patientId: str = Field(..., description="MongoDB ObjectId of patient")

    extractedData: ExtractedData

    filePath: Optional[str] = None

    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None