import os
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

async def process_transcription(file):
    # Read the uploaded file into memory
    file_content = await file.read()
    
    # Create a BytesIO object with a filename attribute
    file_bytes = BytesIO(file_content)
    file_bytes.name = file.filename  # CRITICAL: Preserve the original filename
    
    # Transcribe directly from memory
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Correct model name
        file=file_bytes
    )

    raw_text = transcript.text

    # Convert Hindi → Urdu script
    script_fix = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a transliteration assistant. Convert Hindi/Devanagari script into Urdu script (Arabic). Do not translate."},
            {"role": "user", "content": raw_text}
        ]
    )
    urdu_text = script_fix.choices[0].message.content.strip()

    # Translate Urdu → English
    translation = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Translate the following Urdu text into fluent English."},
            {"role": "user", "content": urdu_text}
        ]
    )
    english_text = translation.choices[0].message.content.strip()

    return {
        "status": "success",
        "urdu_or_punjabi_text": urdu_text,
        "english_translation": english_text
    }