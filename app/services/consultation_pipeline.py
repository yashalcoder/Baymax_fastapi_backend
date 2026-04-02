# app/services/consultation_pipeline.py

import json

from app.services.transcribe_service import process_transcription
from app.services.symptom_extraction import extract_medical_information, load_models
from app.services.prescription import generatePrescription
from app.schemas.patient import PatientInfo
from app.db import get_db
from bson import ObjectId
from fastapi import UploadFile

async def run_consultation_pipeline(
    file: UploadFile,
    doctor_id: str,
    patient_id: str,
    
):
    db = get_db()
    
    # ─────────────────────────────────────
    # STEP 1: Transcription + Speaker ID
    # ─────────────────────────────────────
    print("🎙️ Step 1: Transcribing audio...")
    transcription_response = await process_transcription(file, doctor_id)
    
    if hasattr(transcription_response, 'body'):
        transcription_result = json.loads(transcription_response.body)
    else:
        transcription_result = transcription_response
    # ← Yeh add karo
    if transcription_result.get("code") == "DOCTOR_VERIFICATION_FAILED":
        return {
            "status": "error",
            "message": transcription_result["message"],
            "similarity": transcription_result.get("avg_similarity")
        }
    if transcription_result.get("status") == "error":
        return transcription_result
    # ─────────────────────────────────────
    # STEP 2: Patient ke dialogue nikalo
    # ─────────────────────────────────────
    print("📝 Step 2: Extracting patient dialogue...")
    conversation = transcription_result.get("conversation", [])
    
    patient_text = " ".join([
        seg["english"]
        for seg in conversation
        if "patient" in seg.get("speaker", "").lower()
    ])
    
    if not patient_text:
        # fallback - full transcript use karo
        patient_text = transcription_result.get("full_transcript", "")

    # ─────────────────────────────────────
    # STEP 3: NER - symptoms, duration, severity
    # ─────────────────────────────────────
    print("🔬 Step 3: Extracting medical entities...")
    extracted = extract_medical_information(patient_text)
        
    # consultation_pipeline.py mein symptoms_str banate waqt

    # Full transcript bhi include karo sirf diseases ke liye
    symptoms_str = ", ".join(extracted["diseases"]) if extracted["diseases"] else ""

    # Manually common symptoms bhi check karo English transcript mein
    full_english = " ".join([
        seg["english"] for seg in conversation
        if "patient" in seg.get("speaker", "").lower()
    ])

    extra_symptoms = []
    keyword_map = {
        "fever": "fever",
        "headache": "headache", 
        "weakness": "weakness",
        "throat": "sore throat",
        "tired": "fatigue",
        "sleep": "insomnia",
        "hungry": "loss of appetite"
    }

    for keyword, symptom in keyword_map.items():
        if keyword in full_english.lower():
            extra_symptoms.append(symptom)

    # Combine karo
    all_symptoms = list(set(extracted["diseases"] + extra_symptoms))
    symptoms_str = ", ".join(all_symptoms) if all_symptoms else full_english[:300]
        # ─────────────────────────────────────
    # STEP 4: Patient DB se allergies/history lo
    # ─────────────────────────────────────
    print("👤 Step 4: Fetching patient data from DB...")
    patient_data = await db["users"].find_one(
        {"userId": ObjectId(patient_id)}
    )
    print("Patient Data:", patient_data)  # Debugging line
    medical_report = None
    if patient_data:
        medical_report = (
            f"Blood Group: {patient_data.get('bloodGroup', 'N/A')}, "
            f"Allergies: {patient_data.get('allergies', 'None')}, "
            f"Major Disease: {patient_data.get('majorDisease', 'None')}"
        )
    
    # Latest vitals lo
    vitals_str = None
    vitals_list = patient_data.get("vitals", []) if patient_data else []
    if vitals_list:
        latest = vitals_list[-1]
        vitals_str = (
            f"BP: {latest.get('bloodPressure', 'N/A')}, "
            f"HR: {latest.get('heartRate', 'N/A')}, "
            f"Temp: {latest.get('temperature', 'N/A')}"
        )

    # ─────────────────────────────────────
    # STEP 5: PatientInfo schema banao
    # ─────────────────────────────────────
    patient_info = PatientInfo(
        name=patient_data.get("fullName", "Patient") if patient_data else "Patient",
        symptoms=symptoms_str,
        severity=extracted.get("severity"),
        duration=extracted.get("duration"),
        vitals=vitals_str,
        medical_report=medical_report
    )

    # ─────────────────────────────────────
    # STEP 6: Prescription generate karo
    # ─────────────────────────────────────
    print("💊 Step 6: Generating prescription...")
    prescription = generatePrescription(patient_info)

    # ─────────────────────────────────────
    # STEP 7: DB mein save karo
    # ─────────────────────────────────────
    print("💾 Step 7: Saving to DB...")
    consultation_doc = {
        "doctorId": ObjectId(doctor_id),
        "patientId": ObjectId(patient_id),
        "transcript": transcription_result.get("full_transcript"),
        "conversation": conversation,
        "extractedEntities": extracted,
        "prescription": prescription.get("result"),
        "createdAt": __import__("datetime").datetime.utcnow()
    }
    
    result = await db["consultations"].insert_one(consultation_doc)

    # ─────────────────────────────────────
    # FINAL RESPONSE
    # ─────────────────────────────────────
    return {
        "status": "success",
        "consultationId": str(result.inserted_id),
        "transcription": {
            "full": transcription_result.get("full_transcript"),
            "conversation": conversation
        },
        "extracted": extracted,
        "prescription": prescription.get("result"),
        "patientInfo": {
            "name": patient_info.name,
            "vitals": vitals_str,
            "allergies": patient_data.get("allergies") if patient_data else None
        }
    }