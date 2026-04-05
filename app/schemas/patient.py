from pydantic import BaseModel
from typing import Optional

class PatientInfo(BaseModel):
    name: str
    symptoms: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    vitals: Optional[str] = None
    medical_report: Optional[str] = None
    previous_diagnoses: str | None = None  # ✅ ADD THIS
    doctor_notes: str | None = None