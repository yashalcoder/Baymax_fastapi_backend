from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import transcribe_routes, symptom_routes,prescription,consultation
from app.services.symptom_extraction import load_models
from app.db import connect_to_mongo, close_mongo_connection, get_doctor_collection

import subprocess
import os

app = FastAPI(title="BayMax Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- FFmpeg Test ----------------
def test_ffmpeg():
    ffmpeg_path = r"C:\Users\yasha\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    if not os.path.exists(ffmpeg_path):
        print(f"❌ FFmpeg NOT FOUND: {ffmpeg_path}")
        return False

    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("✓ FFmpeg working")
        return True
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False


# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_app():
    print("🚀 Starting BayMax Backend...")

    test_ffmpeg()           # safe place
    load_models()           # load ONCE
    await connect_to_mongo()

    print("✅ Startup completed")


# ---------------- Shutdown ----------------
@app.on_event("shutdown")
async def shutdown_app():
    await close_mongo_connection()
    print("🛑 MongoDB connection closed")


# ---------------- Routes ----------------
@app.get("/")
async def root():
    return {"message": "Welcome to BayMax API!"}


@app.get("/test-db")
async def test_db():
    try:
        collection = get_doctor_collection()
        count = await collection.count_documents({})
        return {"status": "connected", "doctors_count": count}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


app.include_router(transcribe_routes.router)
app.include_router(symptom_routes.router)
app.include_router(prescription.router)
app.include_router(consultation.router)
# Debug - sabse last line
print("Consultation routes:", consultation.router.routes)