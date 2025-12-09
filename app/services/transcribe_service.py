# import os
# from io import BytesIO
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_api_key)

# async def process_transcription(file):
#     # Read the uploaded file into memory
#     file_content = await file.read()
    
#     # Create a BytesIO object with a filename attribute
#     file_bytes = BytesIO(file_content)
#     file_bytes.name = file.filename  # CRITICAL: Preserve the original filename
    
#     # Transcribe directly from memory
#     transcript = client.audio.transcriptions.create(
#         model="whisper-1",  # Correct model name
#         file=file_bytes
#     )

#     raw_text = transcript.text

#     # Convert Hindi → Urdu script
#     script_fix = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a transliteration assistant. Convert Hindi/Devanagari script into Urdu script (Arabic). Do not translate."},
#             {"role": "user", "content": raw_text}
#         ]
#     )
#     urdu_text = script_fix.choices[0].message.content.strip()

#     # Translate Urdu → English
#     translation = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Translate the following Urdu text into fluent English."},
#             {"role": "user", "content": urdu_text}
#         ]
#     )
#     english_text = translation.choices[0].message.content.strip()

#     return {
#         "status": "success",
#         "urdu_or_punjabi_text": urdu_text,
#         "english_translation": english_text
#    }



import os
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import assemblyai as aai

load_dotenv()

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")  # Add to .env file

client = OpenAI(api_key=openai_api_key)

# async def process_medical_conversation(file):
#     """
#     Process audio file from frontend:
#     1. Transcribe with speaker diarization (AssemblyAI)
#     2. Convert to Urdu script
#     3. Translate to English
#     """
    
#     # Read uploaded file
#     file_content = await file.read()
#     file_bytes = BytesIO(file_content)
#     file_bytes.name = file.filename
    
#     # ============================================
#     # STEP 1: Transcribe with Speaker Diarization   
#     # ============================================
#     config = aai.TranscriptionConfig(
#         speaker_labels=True,
#         format_text=True,
#         punctuate=True,
#         speech_model=aai.SpeechModel.universal,
#         language_detection=True,
#         speakers_expected=2  # Doctor + Patient
    
        
#     )
    
#     transcriber = aai.Transcriber(config=config)
    
#     # Transcribe from memory (not file path)
#     transcript = transcriber.transcribe(file_bytes)
    
#     if transcript.status == aai.TranscriptStatus.error:
#         return {
#             "status": "error",
#             "message": transcript.error
#         }
#      # ============================================
#     # STEP 2: Merge short-gap fake speaker switches
#     # ============================================
#     merged_utterances = []
#     prev_speaker = None
#     prev_end = None

#     for u in transcript.utterances:
#         # gap in seconds
#         gap = ((u.start - prev_end) / 1000) if prev_end is not None else None

#         # Only merge if **same speaker and short gap**
#         if prev_speaker is not None and gap is not None:
#             if u.speaker == prev_speaker and gap < 1.5:
#                 # Merge text with previous utterance
#                 merged_utterances[-1].text += " " + u.text
#                 merged_utterances[-1].end = u.end
#                 continue  # skip adding as new utterance

#         # Otherwise, treat as new utterance
#         merged_utterances.append(u)
#         prev_speaker = u.speaker
#         prev_end = u.end

#     # ============================================
#     # STEP 2: Process each speaker's utterances
#     # ============================================
#     conversation = []
    
#     for utterance in merged_utterances:
#         # speaker = f"Speaker {utterance.speaker}"  # "Speaker A", "Speaker B"
#         speaker_roles = {"A": "Doctor", "B": "Patient"}
#         speaker = speaker_roles.get(utterance.speaker, f"Speaker {utterance.speaker}")
#         text = utterance.text
        
#         # Convert to Urdu script
#         script_fix = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": "Convert Hindi/Devanagari script to Urdu script (Arabic). Do not translate, only change script."
#                 },
#                 {"role": "user", "content": text}
#             ]
#         )
#         urdu_text = script_fix.choices[0].message.content.strip()
        
#         # Translate to English
#         translation = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": "Translate the following Urdu text into fluent English."
#                 },
#                 {"role": "user", "content": urdu_text}
#             ]
#         )
#         english_text = translation.choices[0].message.content.strip()
        
#         conversation.append({
#             "speaker": speaker,
#             "timestamp": f"{utterance.start / 1000:.2f}s - {utterance.end / 1000:.2f}s",
#             "original": text,
#             "urdu": urdu_text,
#             "english": english_text
#         })
    
#     # ============================================
#     # STEP 3: Return formatted results
#     # ============================================
#     return {
#         "status": "success",
#         "full_transcript": transcript.text,
#         "conversation": conversation,
#         "metadata": {
#             "duration": f"{transcript.audio_duration}s",
#             "speakers_detected": len(set(u.speaker for u in transcript.utterances)),
#             "language_detected": getattr(transcript, 'language_code', 'unknown')
#         }
#     }


# ============================================
# FastAPI Endpoint Example
# ============================================
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from app.db import get_db,get_doctor_collection
from fastapi import Depends
from bson import ObjectId

# FFmpeg path set karo - YE ZARURI HAI
FFMPEG_PATH = r"C:\Users\Yashal Rafique\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

print(f"✓ FFmpeg path set: {AudioSegment.converter}")
# app = FastAPI()
async def get_doctor_embedding(doctor_id: str):
    doctor_collection = get_doctor_collection()  # collection object
 

    if not ObjectId.is_valid(doctor_id):
        raise Exception(f"Invalid doctor_id: {doctor_id}")
    doctor = await doctor_collection.find_one({"userId": ObjectId(doctor_id)})
    # doctor = await doctor_collection.find_one({"email":"yashalrafique@gmail.com"})
    if not doctor:
        raise Exception(f"Doctor not found: {doctor_id}")
    
    embeddings = doctor.get("voice_fingerprint")
    if not embeddings:
        raise Exception(f"No embeddings found for doctor {doctor_id}")
    
    return embeddings
# @app.post("/transcribe")
async def process_transcription(file: UploadFile = File(...), doctorId: str = None):
    print("\n" + "="*60)
    print("DEBUG: process_transcription called")
    print("="*60)
    print(f"file type: {type(file)}")
    print(f"file: {file}")
    print(f"doctorId: {doctorId}")
    
    try:
        # Test file reading
        print("\nReading file...")
        content = await file.read()
        print(f"✓ File read successfully: {len(content)} bytes")
        
        # Test embeddings
        print(f"\nGetting embeddings for doctor: {doctorId}")
        embeddings = await get_doctor_embedding(doctorId)
        print(f"✓ Embeddings loaded: {len(embeddings)} values")
        
        # Create BytesIO
        print("\nCreating BytesIO object...")
        file_obj = BytesIO(content)
        file_obj.name = file.filename
        print(f"✓ BytesIO created: {file_obj.name}")
        
        # Process
        print("\nCalling process_medical_conversation...")
        result = await process_medical_conversation(
            file_obj,
            doctor_voice_embedding=embeddings
        )
        
        print("✓ Processing complete!")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
# # ============================================
# # Example Response Format
# # ============================================
# """
# {
#   "status": "success",
#   "full_transcript": "Original full text...",
#   "conversation": [
#     {
#       "speaker": "Speaker A",
#       "timestamp": "0.50s - 3.20s",
#       "original": "Original transcribed text",
#       "urdu": "اردو میں متن",
#       "english": "Text in English"
#     },
#     {
#       "speaker": "Speaker B",
#       "timestamp": "3.50s - 6.80s",
#       "original": "Patient's response",
#       "urdu": "مریض کا جواب",
#       "english": "Patient's answer"
#     }
#   ],
#   "metadata": {
#     "duration": "45.2s",
#     "speakers_detected": 2,
#     "language_detected": "ur"
#   }
# }
# """
# """
# Simple Voice Embedding using HuggingFace Transformers
# 100% Windows compatible - No Pyannote, No SpeechBrain
# Uses Wav2Vec2 for voice features
# """
import wave  # Add this import
import os
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import assemblyai as aai
import numpy as np
from scipy.spatial.distance import cosine
from pydub import AudioSegment
import tempfile
import json

# HuggingFace imports (Windows compatible)
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa

load_dotenv()
AudioSegment.converter = r"C:\Users\Yashal Rafique\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# ============================================
# Load HuggingFace Wav2Vec2 Model (One Time)
# ============================================
print("🔄 Loading HuggingFace Wav2Vec2 model...")
print("   Model: facebook/wav2vec2-base (360M parameters)")
print("   First download: ~1.5GB (one-time only)")

# Use smaller base model (good for voice recognition)
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

# Set to evaluation mode
model.eval()

print("✅ Model loaded successfully!\n")


import noisereduce as nr
import os
import tempfile
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects

def preprocess_audio(audio_path: str, trim_silence: bool = True) -> str:
    """
    Preprocess audio:
    - Convert to mono, 16kHz
    - Reduce background noise
    - Normalize audio
    - Trim silence (optional)
    Returns path to processed audio (WAV)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Noise reduction
        y = nr.reduce_noise(y=y, sr=sr)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Save temporarily as WAV
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        # Convert to 16-bit PCM WAV
        import wave
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            audio_int = np.int16(y * 32767)
            wav_file.writeframes(audio_int.tobytes())
        
        # Optional: Trim silence
        if trim_silence:
            sound = AudioSegment.from_wav(temp_path)
            # Auto trim silence
            sound = effects.strip_silence(sound, silence_thresh=-40, padding=100)
            sound.export(temp_path, format="wav")
        
        return temp_path
    
    except Exception as e:
        print("Error in preprocessing:", e)
        return audio_path

def convert_audio_to_wav(input_path: str) -> str:
    """
    Convert any audio format to WAV using FFmpeg
    Handles WebM, Opus, MP3, M4A, etc.
    """
    try:
        print("convertiung extesnion")
        # Check if already WAV
        if input_path.lower().endswith('.wav'):
            return input_path
        
        # Create output path
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        
        print(f"   → Converting {os.path.basename(input_path)} to WAV...")
        
        # Use pydub for conversion (uses FFmpeg internally)
        audio = AudioSegment.from_file(input_path)
        
        # Export as WAV (16kHz, mono)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format='wav')
        
        print(f"   → Converted to WAV successfully")
        return output_path
        
    except Exception as e:
        print(f"   ⚠️ Conversion failed: {e}")
        # Return original if conversion fails
        return input_path

# ============================================
# Create Voice Embedding
# ============================================
def create_voice_embedding(audio_path: str) -> list:
    """
    Generate voice embedding from audio file
    Uses HuggingFace Wav2Vec2
    
    Args:
        audio_path: Path to audio file
    
    Returns: 768-dimensional embedding vector
    """
    try:
        # audio_path = convert_audio_to_wav(audio_path)
        # Load audio with librosa (Windows compatible)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Process with HuggingFace processor
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Get last hidden state and average over time dimension
            embeddings = outputs.last_hidden_state
            # Average pooling over time
            embedding = torch.mean(embeddings, dim=1).squeeze()
        
        # Convert to list
        embedding_list = embedding.numpy().tolist()
        
        return embedding_list
    
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        raise e


# ============================================
# Extract Audio Segment
# ============================================
def extract_audio_segment(full_audio_path: str, start_ms: int, end_ms: int) -> str:
    """
    Extract audio segment using librosa + wave (No FFmpeg issues)
    """
    try:
        # Load audio
        y, sr = librosa.load(full_audio_path, sr=16000, mono=True)
        
        # Calculate sample indices
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Create temp file path
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)  # Close file descriptor immediately
        
        # Save using wave module (built-in Python)
        import wave
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sr)  # 16000 Hz
            
            # Convert float32 to int16
            segment_int = np.int16(segment * 32767)
            wav_file.writeframes(segment_int.tobytes())
        
        return temp_path
        
    except Exception as e:
        print(f"❌ Segment extraction failed: {e}")
        raise e

# ============================================
# Compute Similarity
# ============================================
def compute_similarity(embedding1: list, embedding2: list) -> float:
    """
    Calculate cosine similarity between embeddings
    Returns: 0.0 to 1.0
    """
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    similarity = 1 - cosine(emb1, emb2)
    return float(max(0.0, min(1.0, similarity)))


# ============================================
# Verify Speaker
# ============================================
def verify_speaker(
    segment_audio_path: str,
    enrolled_embedding: list,
    threshold: float = 0.70
) -> dict:
    """
    Verify if segment matches enrolled voice
    """
    
    try:
        # Generate segment embedding
        segment_embedding = create_voice_embedding(segment_audio_path)
        print("Segment embedding",segment_embedding)
        # Compute similarity
        similarity = compute_similarity(enrolled_embedding, segment_embedding)
        
        is_doctor = similarity >= threshold
        
        return {
            "is_doctor": is_doctor,
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "confidence": "High" if similarity > 0.80 else "Medium" if similarity > 0.65 else "Low"
        }
    
    except Exception as e:
        return {
            "is_doctor": None,
            "similarity": 0.0,
            "threshold": threshold,
            "confidence": "Error",
            "error": str(e)
        }

async def process_medical_conversation(
    file,
    doctor_voice_embedding: list = None,
    verification_threshold: float = 0.70
):
    """
    Complete processing pipeline with HuggingFace embeddings
    """
    
    file_content = file.read()
    file_bytes = BytesIO(file_content)
    filename = getattr(file, 'name', 'audio.wav')
    file_bytes.name = filename
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_content)
        temp_audio_path = tmp.name
     # ADD THIS: Preprocess audio
     # ADD THIS: Preprocess audio
    print("\n[0/4] 🔄 Preprocessing audio...")
    cleaned_audio_path = preprocess_audio(temp_audio_path)
    print("✅ Audio cleaned")   
    try:
        print("=" * 60)
        print("🎙️ PROCESSING CONVERSATION")
        print("=" * 60)
        
        # ============================================
        # STEP 1: AssemblyAI Diarization
        # ============================================
        print("\n[1/4] 🔄 Running speaker diarization...")
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            format_text=True,
            punctuate=True,
            speakers_expected=2,
            language_detection=True
        )
        
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(file_bytes)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {"status": "error", "message": transcript.error}
        
        print(f"✅ Found {len(transcript.utterances)} segments")
        
        # ============================================
        # STEP 2: Merge Utterances
        # ============================================
        print("\n[2/4] 🔄 Merging short utterances...")
        
        merged_utterances = []
        prev_speaker = None
        prev_end = None

        for u in transcript.utterances:
            gap = ((u.start - prev_end) / 1000) if prev_end else None
            
            if prev_speaker and gap and u.speaker == prev_speaker and gap < 0.5:
                merged_utterances[-1].text += " " + u.text
                merged_utterances[-1].end = u.end
                continue
            
            merged_utterances.append(u)
            prev_speaker = u.speaker
            prev_end = u.end
        
        print(f"✅ Merged to {len(merged_utterances)} segments")
        
        # ============================================
        # STEP 3: Voice Verification & Translation
        # ============================================
        print(f"\n[3/4] 🔄 Voice verification & translation...")
        print(f"   Threshold: {verification_threshold}")
        print(f"   Status: {'ENABLED' if doctor_voice_embedding else 'DISABLED'}")
        
        conversation = []
        stats = {
            "verified_as_doctor": 0,
            "verified_as_patient": 0,
            "errors": 0
        }
        
        # IMPORTANT: Loop through ALL segments
        for idx, utterance in enumerate(merged_utterances, 1):
            print(f"\n   Segment {idx}/{len(merged_utterances)}: {utterance.start/1000:.1f}s - {utterance.end/1000:.1f}s")
            
            text = utterance.text
            speaker_label = f"Speaker {utterance.speaker}"
            verification_result = None
            
            # ============================================
            # Voice Verification (if enabled)
            # ============================================
            if doctor_voice_embedding:
                segment_path = None
                try:
                    segment_path = extract_audio_segment(
                        cleaned_audio_path,
                        utterance.start,
                        utterance.end
                    )
                    print(f"      → Extracted segment: {segment_path}")
                    
                    verification_result = verify_speaker(
                        segment_path,
                        doctor_voice_embedding,
                        verification_threshold
                    )
                    print(f"      → Verification: {verification_result}")
                    
                    if verification_result.get("skipped"):
                        # Segment too short, use AssemblyAI label
                        speaker_label = "Doctor" if utterance.speaker == "A" else "Patient"
                        stats["errors"] += 1
                        print(f"      → {speaker_label} (skipped - too short)")
                        
                    elif verification_result["is_doctor"] is not None:
                        is_doctor = verification_result["is_doctor"]
                        speaker_label = "Doctor ✓" if is_doctor else "Patient"
                        
                        if is_doctor:
                            stats["verified_as_doctor"] += 1
                        else:
                            stats["verified_as_patient"] += 1
                        
                        print(f"      → {speaker_label} (sim: {verification_result['similarity']:.3f})")
                    else:
                        stats["errors"] += 1
                        speaker_label = "Unknown"
                
                except Exception as e:
                    print(f"      → ⚠️ Verification error: {e}")
                    stats["errors"] += 1
                    speaker_label = "Unknown"
                
                finally:
                    if segment_path and os.path.exists(segment_path):
                        try:
                            os.unlink(segment_path)
                        except:
                            pass
            else:
                # No verification, use AssemblyAI labels
                speaker_label = "Doctor" if utterance.speaker == "A" else "Patient"
            
            # ============================================
            # Translation (for ALL segments)
            # ============================================
            try:
                print(f"      → Translating...")
                
                # Urdu script conversion
                script_fix = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Convert to Urdu script. No translation."},
                        {"role": "user", "content": text}
                    ]
                )
                urdu_text = script_fix.choices[0].message.content.strip()
                
                # English translation
                translation = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate Urdu to English."},
                        {"role": "user", "content": urdu_text}
                    ]
                )
                english_text = translation.choices[0].message.content.strip()
                
                print(f"      ✓ Translation complete")
                
            except Exception as e:
                print(f"      → ⚠️ Translation error: {e}")
                urdu_text = text
                english_text = text
            
            # ============================================
            # Build conversation entry
            # ============================================
            entry = {
                "speaker": speaker_label,
                "timestamp": f"{utterance.start/1000:.2f}s - {utterance.end/1000:.2f}s",
                "original": text,
                "urdu": urdu_text,
                "english": english_text
            }
            
            if verification_result:
                entry["verification"] = verification_result
            
            conversation.append(entry)
            print(f"      ✓ Segment {idx} processed")
        
        print(f"\n✅ Processing complete!")
        print(f"   Total segments processed: {len(conversation)}")
        print("=" * 60 + "\n")
        
        return {
            "status": "success",
            "full_transcript": transcript.text,
            "conversation": conversation,
            "metadata": {
                "duration": f"{transcript.audio_duration}s",
                "total_segments": len(conversation),
                "speakers_detected": len(set(u.speaker for u in transcript.utterances)),
                "verification": {
                    "enabled": doctor_voice_embedding is not None,
                    "threshold": verification_threshold,
                    "stats": stats
                }
            }
        }
    
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass