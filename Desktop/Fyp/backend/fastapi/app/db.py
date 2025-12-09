from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI

client = None
db = None
doctor_collection = None
async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["Baymax_DB"]  # ✅ match exactly database name

    print("✅ Connected to MongoDB")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("✅ MongoDB connection closed")

# Dependency for FastAPI endpoints
def get_db():
    return db
def get_doctor_collection():
    return db.get_collection("doctors")