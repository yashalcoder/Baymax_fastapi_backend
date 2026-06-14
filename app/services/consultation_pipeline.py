# app/services/consultation_pipeline.py

import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.services.transcribe_service import process_transcription
from app.services.symptom_extraction import extract_medical_information, load_models
from app.services.prescription import generatePrescription
from app.schemas.patient import PatientInfo
from app.db import get_db
from bson import ObjectId
from fastapi import UploadFile

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings in batch using text-embedding-3-small."""
    try:
        processed_texts = [text.replace("\n", " ") for text in texts]
        response = await openai_client.embeddings.create(
            input=processed_texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"⚠️ Error generating OpenAI embeddings: {e}")
        raise e

def format_report_for_embedding(report) -> str:
    extracted = report.get("extractedData", {})
    diagnoses = extracted.get("diagnoses", [])
    notes = extracted.get("doctor_notes", [])
    
    parts = []
    if diagnoses:
        if isinstance(diagnoses, list):
            parts.append(f"Diagnoses: {', '.join(diagnoses)}")
        elif isinstance(diagnoses, str):
            parts.append(f"Diagnosis: {diagnoses}")
    if notes:
        if isinstance(notes, list):
            parts.append(f"Doctor Notes: {' '.join(notes) if isinstance(notes, list) else notes}")
            
    # Fallback to general fields if no diagnoses/notes found
    if not parts:
        parts.append(str(extracted))
        
    return " | ".join(parts)

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
    # STEP 4: Patient DB se allergies/history lo + RAG Retrieval
    # ─────────────────────────────────────
    patient_data = await db["patients"].find_one({"_id": ObjectId(patient_id)})

    print("📄 Fetching medical reports for RAG...")
    reports_cursor = db["medicalreports"].find(
        {"patientId": ObjectId(patient_id)}
    ).sort("createdAt", -1)
    
    all_reports = await reports_cursor.to_list(length=100)
    print("Total medical reports found in DB:", len(all_reports))
    
    selected_reports = []
    rag_metadata = []
    
    if all_reports:
        # Check if RAG is useful and we have patient text/reports
        if patient_text and len(all_reports) > 0:
            try:
                # Format report contents for semantic search
                report_texts = [format_report_for_embedding(r) for r in all_reports]
                
                # Get embeddings for query and reports
                query_emb = (await get_embeddings([patient_text]))[0]
                report_embs = await get_embeddings(report_texts)
                
                # Calculate cosine similarities using dot product (vectors are normalized)
                query_vector = np.array(query_emb)
                similarities = []
                for emb in report_embs:
                    sim = np.dot(query_vector, np.array(emb))
                    similarities.append(float(sim))
                    
                # Zip and sort by similarity descending
                ranked = sorted(
                    zip(all_reports, similarities, report_texts),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Select top 3 relevant reports
                top_k = ranked[:3]
                selected_reports = [item[0] for item in top_k]
                
                # Populate RAG metadata
                for report, score, text in top_k:
                    rag_metadata.append({
                        "reportId": str(report.get("_id")),
                        "similarity_score": round(score, 4),
                        "preview": text[:150] + "..." if len(text) > 150 else text,
                        "createdAt": str(report.get("createdAt"))
                    })
                print(f"✅ RAG successfully retrieved top {len(selected_reports)} reports.")
            except Exception as e:
                print(f"⚠️ RAG semantic search failed: {e}. Falling back to chronological reports.")
                selected_reports = all_reports[:3]
                rag_metadata = [{"status": "fallback", "reason": str(e)}]
        else:
            # Fallback if no patient text to query
            selected_reports = all_reports[:3]
            rag_metadata = []
    else:
        selected_reports = []
        rag_metadata = []

    # Print premium console statistics dashboard for RAG
    print("\n" + "="*80)
    print("📂  RAG MEDICAL REPORTS RETRIEVAL & ACCURACY REPORT")
    print("="*80)
    print(f"🔹 Patient ID                  : {patient_id}")
    print(f"🔹 Total Reports Found in DB   : {len(all_reports)}")
    print(f"🔹 Reports Selected for Context: {len(selected_reports)}")
    print("-"*80)
    print(f"📋 SELECTED REPORTS DETAILS:")
    if selected_reports and any(item.get("status") != "fallback" for item in rag_metadata):
        for idx, item in enumerate(rag_metadata, 1):
            print(f"   {idx}. Report ID       : {item.get('reportId')}")
            print(f"      Similarity Score: {item.get('similarity_score')}")
            print(f"      Created At      : {item.get('createdAt')}")
            print(f"      Content Preview : {item.get('preview')}")
            print(f"      " + "-"*40)
    elif selected_reports:
        print(f"   ⚠️ Fallback chronological reports used (total: {len(selected_reports)})")
        for idx, report in enumerate(selected_reports, 1):
            print(f"   {idx}. Report ID       : {str(report.get('_id'))}")
            print(f"      Created At      : {str(report.get('createdAt'))}")
            print(f"      " + "-"*40)
    else:
        print("   ❌ No reports selected (or none available).")
    print("="*80 + "\n")

    all_diagnoses = []
    all_notes = []

    for report in selected_reports:
        extracted_data = report.get("extractedData", {})

        # diagnoses (list or str)
        diagnoses = extracted_data.get("diagnoses", [])
        if isinstance(diagnoses, list):
            all_diagnoses.extend(diagnoses)
        elif isinstance(diagnoses, str):
            all_diagnoses.append(diagnoses)

        # doctor_notes (list OR string)
        notes = extracted_data.get("doctor_notes", [])
        
        if isinstance(notes, list):
            all_notes.extend(notes)
        elif isinstance(notes, str):
            all_notes.append(notes)
            
    medical_report = None

    if patient_data:
        medical_report = (
            f"Blood Group: {patient_data.get('bloodGroup', 'N/A')}, "
            f"Allergies: {patient_data.get('allergies', 'None')}, "
            f"Major Disease: {patient_data.get('majorDisease', 'None')}"
        )
    previous_diagnoses_str = ", ".join(set(all_diagnoses)) if all_diagnoses else None
    doctor_notes_str = " | ".join(set(all_notes)) if all_notes else None
            
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
        medical_report=medical_report,
        previous_diagnoses=previous_diagnoses_str,
        doctor_notes=doctor_notes_str, 
    ) 

    # ─────────────────────────────────────
    # STEP 6: Prescription generate karo
    # ─────────────────────────────────────
    print("💊 Step 6: Generating prescription...")
    prescription = await generatePrescription(patient_info)

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
        "ragMetadata": rag_metadata,
        "ragStats": {
            "total_reports_available": len(all_reports),
            "reports_selected": len(selected_reports)
        },
        "voiceVerification": transcription_result.get("metadata", {}).get("verification"),
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
            "conversation": conversation,
            "summary": transcription_result.get("summary"),
        },
        "extracted": extracted,
        "prescription": prescription.get("result"),
        "rag_metadata": rag_metadata,
        "rag_stats": {
            "total_reports_available": len(all_reports),
            "reports_selected": len(selected_reports)
        },
        "voice_verification": transcription_result.get("metadata", {}).get("verification"),
        "patientInfo": {
            "name": patient_info.name,
            "vitals": vitals_str,
            "allergies": patient_data.get("allergies") if patient_data else None
        }
    }
