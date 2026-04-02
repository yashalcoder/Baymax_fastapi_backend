from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# Nested Models
class Address(BaseModel):
    street: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""
    zipCode: Optional[str] = ""


class MedicalQualifications(BaseModel):
    degree: Optional[str] = ""
    university: Optional[str] = ""
    graduationYear: Optional[int] = None
    licenseNumber: Optional[str] = ""


class Professional(BaseModel):
    specialization: Optional[str] = ""
    subSpecialization: Optional[str] = ""
    experience: Optional[int] = None
    hospital: Optional[str] = ""


class VoiceFingerprint(BaseModel):
    data: Optional[dict] = {}


class Availability(BaseModel):
    day: Optional[str] = ""
    startTime: Optional[str] = ""
    endTime: Optional[str] = ""


# Base Schema (shared fields)
class DoctorBase(BaseModel):
    firstName: Optional[str] = ""
    lastName: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""
    alternatePhone: Optional[str] = ""
    gender: Optional[str] = ""
    dateOfBirth: Optional[str] = ""
    bio: Optional[str] = ""
    languages: Optional[List[str]] = []
    availability: Optional[List[Availability]] = []
    address: Optional[Address] = None
    medicalQualifications: Optional[MedicalQualifications] = None
    professional: Optional[Professional] = None


# Create Schema (for POST requests)
class DoctorCreate(DoctorBase):
    medicalDegree: str
    university: str
    graduationYear: int
    licenseNumber: str
    specialization: str
    subSpecialization: str
    password: str


# Update Schema (for PUT/PATCH requests)
class DoctorUpdate(DoctorBase):
    medicalDegree: Optional[str] = None
    university: Optional[str] = None
    graduationYear: Optional[int] = None
    licenseNumber: Optional[str] = None
    specialization: Optional[str] = None
    subSpecialization: Optional[str] = None
    password: Optional[str] = None


# Response Schema (for GET responses)
class DoctorResponse(DoctorBase):
    id: str = Field(alias="_id")
    userId: str
    doctorId: Optional[str] = ""
    medicalDegree: str
    university: str
    graduationYear: int
    licenseNumber: str
    specialization: str
    subSpecialization: str
    voiceFingerprint: Optional[VoiceFingerprint] = None
    voice_fingerprint: Optional[List] = []
    createdAt: datetime
    updatedAt: datetime

    class Config:
        populate_by_name = True  # allows _id alias to work
        from_attributes = True   # for ORM compatibility