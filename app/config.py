# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file into environment

MONGO_URI = os.getenv("DB_URL")
FFMPEG_DIR = os.getenv("FFMPEG_DIR", r"C:\Users\yasha\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# SECRET_KEY = os.getenv("SECRET_KEY", "defaultsecretkey")