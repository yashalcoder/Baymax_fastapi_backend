import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from app.schemas.patient import PatientInfo
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generatePrescription(patient_info: PatientInfo):
    prompt = f"""
You are a professional medical doctor.

Patient Information:
- Name: {patient_info.name}
- Symptoms: {patient_info.symptoms}
- Severity: {patient_info.severity or "Not specified"}
- Duration: {patient_info.duration or "Not specified"}
"""
    if patient_info.vitals:
        prompt += f"- Vitals: {patient_info.vitals}\n"

    if patient_info.medical_report:
        prompt += f"- Medical History: {patient_info.medical_report}\n"

    # ✅ NEW — Previous reports se context
    if patient_info.previous_diagnoses:
        prompt += f"- Previous Diagnoses (from past reports): {patient_info.previous_diagnoses}\n"

    if patient_info.doctor_notes:
        prompt += f"- Previous Doctor Notes: {patient_info.doctor_notes}\n"

    prompt += """
Based on current symptoms AND previous medical history above, generate prescription.

TASK:
Generate a JSON object with:
- diagnosis: string
- prescription: list of medicines, each with:
    - medicine: string
    - type: string
    - dosage: string (e.g., "100 mg")
    - duration: string (e.g., "once daily for 7 days")
    - precautions: string
- advice: list of strings
- disclaimer: string

IMPORTANT:
- Consider previous diagnoses to avoid contradicting medicines
- If patient has recurring condition, mention it in diagnosis
- Return ONLY valid JSON, no markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an experienced licensed physician."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=600
    )

    result_text = response.choices[0].message.content.strip()

    # Strip ```json and ``` if present
    result_text = re.sub(r"^```json\s*", "", result_text, flags=re.IGNORECASE)
    result_text = re.sub(r"```$", "", result_text, flags=re.IGNORECASE).strip()

    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        result_json = {"error": "Failed to parse JSON from model", "raw_output": result_text}

    return {
        "patient_name": patient_info.name,
        "result": result_json
    }
