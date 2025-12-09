# app/main.py
from fastapi import FastAPI
# from app.routes import users, diagnosis, auth
from app.routes import transcribe_routes
# from app.routes import symptom_routes
from fastapi.middleware.cors import CORSMiddleware

from app.db import connect_to_mongo, close_mongo_connection,get_db,get_doctor_collection


app = FastAPI(title="BayMax Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import subprocess
import os

# Test karo FFmpeg chal raha hai ya nahi
def test_ffmpeg():
    ffmpeg_path = r"C:\Users\Yashal Rafique\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    
    # Check file exists
    if not os.path.exists(ffmpeg_path):
        print(f"❌ FFmpeg NOT FOUND at: {ffmpeg_path}")
        return False
    
    print(f"✓ FFmpeg file exists: {ffmpeg_path}")
    
    # Try running it
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✓ FFmpeg works!")
        print(f"   Version: {result.stdout.split()[2]}")
        return True
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

# Test karo
test_ffmpeg()
# Startup / shutdown events
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

@app.get("/")
async def root():
    return {"message": "Welcome to Baymax API!"}

@app.get("/test-db")
async def test_db():
    try:
        doctor_collection = get_doctor_collection()
        count = await doctor_collection.count_documents({})
        return {"status": "connected", "doctors_count": count}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# Register routes
app.include_router(transcribe_routes.router)
# app.include_router(symptom_routes)
# app.include_router(users.router, prefix="/api/users", tags=["Users"])
# app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
# app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["Diagnosis"])
