# app/main.py
from fastapi import FastAPI
# from app.routes import users, diagnosis, auth
from app.routes import transcribe_routes
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="BayMax Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routes
app.include_router(transcribe_routes.router)
# app.include_router(users.router, prefix="/api/users", tags=["Users"])
# app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
# app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["Diagnosis"])
