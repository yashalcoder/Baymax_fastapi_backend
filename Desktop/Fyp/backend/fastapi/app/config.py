# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file into environment

MONGO_URI = os.getenv("DB_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# SECRET_KEY = os.getenv("SECRET_KEY", "defaultsecretkey")
# 