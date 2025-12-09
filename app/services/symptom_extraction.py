# from fastapi import FastAPI
# from pyngrok import ngrok
# import spacy
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Load SciSpacy model
# nlp = spacy.load("en_ner_bc5cdr_md")

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "✅ SciSpaCy API is running successfully!"}

# @app.post("/extract_symptoms")
# async def extract_symptoms(data: dict):
#     text = data.get("text", "")
#     doc = nlp(text)
#     symptoms = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]
#     return {"symptoms": symptoms}

# print(extract_symptoms("i hav pain in jeart legpain"))

import spacy
from fastapi import FastAPI

app = FastAPI()

# Load model once at startup, not on every request
@app.on_event("startup")
async def load_model():
    global nlp
    try:
        # If you trained custom model
        nlp = spacy.load("en_ner_bc5cdr_md")  # your folder name
        # OR for standard model
        # nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/extract_symptoms")
async def extract_symptoms(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"entities": entities}