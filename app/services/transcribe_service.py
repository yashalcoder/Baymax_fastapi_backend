import os
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import assemblyai as aai

load_dotenv()

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# ============================================
# FastAPI Endpoint Example
# ============================================
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from app.db import get_db,get_doctor_collection
from fastapi import Depends
from bson import ObjectId

# FFmpeg path set karo
FFMPEG_PATH = r"C:\Users\Yashal Rafique\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

print(f"✓ FFmpeg path set: {AudioSegment.converter}")


async def get_doctor_embedding(doctor_id: str):
    doctor_collection = get_doctor_collection()
 
    doctor_collection = get_doctor_collection()
 
    if not ObjectId.is_valid(doctor_id):
        raise Exception(f"Invalid doctor_id: {doctor_id}")
    doctor = await doctor_collection.find_one({"userId": ObjectId(doctor_id)})
    
    
    if not doctor:
        raise Exception(f"Doctor not found: {doctor_id}")
    
    embeddings = doctor.get("voice_fingerprint")
    if not embeddings:
        raise Exception(f"No embeddings found for doctor {doctor_id}")
    
    return embeddings


async def process_transcription(file: UploadFile = File(...), doctorId: str = None):
    print("\n" + "="*60)
    print("DEBUG: process_transcription called")
    print("="*60)
    print(f"file type: {type(file)}")
    print(f"file: {file}")
    print(f"doctorId: {doctorId}")
    
    try:
        print("\nReading file...")
        content = await file.read()
        print(f"✓ File read successfully: {len(content)} bytes")
        
        print(f"\nGetting embeddings for doctor: {doctorId}")
        embeddings = await get_doctor_embedding(doctorId)
        print(f"✓ Embeddings loaded: {len(embeddings)} values")
        
        print("\nCreating BytesIO object...")
        file_obj = BytesIO(content)
        file_obj.name = file.filename
        print(f"✓ BytesIO created: {file_obj.name}")
        
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

import wave
import numpy as np
from scipy.spatial.distance import cosine
import tempfile
import json

# HuggingFace imports
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects

load_dotenv()

# Load HuggingFace Wav2Vec2 Model
print("🔄 Loading HuggingFace Wav2Vec2 model...")
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()
print("✅ Model loaded successfully!\n")


def preprocess_audio(audio_path: str, trim_silence: bool = True) -> str:
    """Preprocess audio: mono, 16kHz, noise reduction, normalization"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y = nr.reduce_noise(y=y, sr=sr)
        y = librosa.util.normalize(y)
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            audio_int = np.int16(y * 32767)
            wav_file.writeframes(audio_int.tobytes())
        
        if trim_silence:
            sound = AudioSegment.from_wav(temp_path)
            sound = effects.strip_silence(sound, silence_thresh=-40, padding=100)
            sound.export(temp_path, format="wav")
        
        return temp_path
    except Exception as e:
        print("Error in preprocessing:", e)
        return audio_path


def create_voice_embedding(audio_path: str) -> list:
    """Generate 768-dimensional voice embedding using Wav2Vec2"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = librosa.util.normalize(audio)
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            embedding = torch.mean(embeddings, dim=1).squeeze()
        
        return embedding.numpy().tolist()
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        raise e


def extract_audio_segment(full_audio_path: str, start_ms: int, end_ms: int) -> str:
    """Extract audio segment using librosa"""
    try:
        y, sr = librosa.load(full_audio_path, sr=16000, mono=True)
        
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        segment = y[start_sample:end_sample]
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            segment_int = np.int16(segment * 32767)
            wav_file.writeframes(segment_int.tobytes())
        
        return temp_path
    except Exception as e:
        print(f"❌ Segment extraction failed: {e}")
        raise e


def compute_similarity(embedding1: list, embedding2: list) -> float:
    """Calculate cosine similarity (0.0 to 1.0)"""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    similarity = 1 - cosine(emb1, emb2)
    return float(max(0.0, min(1.0, similarity)))


def verify_speaker(segment_audio_path: str, enrolled_embedding: list, threshold: float = 0.70) -> dict:
    """Verify if segment matches enrolled voice"""
    try:
        segment_embedding = create_voice_embedding(segment_audio_path)
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
    FIXED VERSION - Option 3:
    - Only verify Speaker A against doctor's voice
    - All other speakers automatically labeled as Patient
    """
    file_content = file.read()
    file_bytes = BytesIO(file_content)
    filename = getattr(file, 'name', 'audio.wav')
    file_bytes.name = filename

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_content)
        temp_audio_path = tmp.name

    try:
        print("=" * 60)
        print("🎙️ PROCESSING CONVERSATION (FIXED - Option 3)")
        print("=" * 60)

        # ============================================
        # STEP 1: AssemblyAI Diarization
        # ============================================
        print("\n[1/5] 🔄 Running speaker diarization...")
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
        print("\n[2/5] 🔄 Merging short utterances...")
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
        # STEP 3: Extract Doctor's Voice Embedding (if not provided)
        # ============================================
        if doctor_voice_embedding is None:
            print("\n[3/5] 🔄 Extracting doctor's voice embedding from Speaker A...")
            
            doctor_segments = [u for u in merged_utterances if u.speaker == "A"]
            
            if len(doctor_segments) == 0:
                print("⚠️  No Speaker A found! Skipping voice verification.")
            else:
                doctor_embeddings = []
                segments_used = 0
                
                for utterance in doctor_segments[:5]:
                    segment_path = None
                    try:
                        segment_path = extract_audio_segment(
                            temp_audio_path,
                            utterance.start,
                            utterance.end
                        )
                        
                        embedding = create_voice_embedding(segment_path)
                        if embedding is not None:
                            doctor_embeddings.append(embedding)
                            segments_used += 1
                            print(f"     ✓ Extracted embedding from segment {segments_used}")
                    
                    except Exception as e:
                        print(f"     ⚠️  Segment skipped: {e}")
                    
                    finally:
                        if segment_path and os.path.exists(segment_path):
                            try:
                                os.unlink(segment_path)
                            except:
                                pass
                
                if doctor_embeddings:
                    doctor_voice_embedding = np.mean(doctor_embeddings, axis=0).tolist()
                    print(f"✅ Doctor's voice fingerprint created from {segments_used} segments")
                else:
                    print("⚠️  Could not extract doctor's voice embedding")
        else:
            print("\n[3/5] ✅ Using provided doctor's voice embedding")

        # ============================================
        # STEP 4: Voice Verification - ONLY FOR SPEAKER A
        # ============================================
        print(f"\n[4/5] 🔄 Verifying ONLY Speaker A against doctor's voice...")
        print(f"     Threshold: {verification_threshold}")
        print(f"     Rule: Speaker A → verify, Others → auto Patient")
        
        conversation = []
        stats = {
            "verified_as_doctor": 0,
            "verified_as_patient": 0,
            "failed_verification": 0,
            "auto_patient": 0
        }
        
        for idx, utterance in enumerate(merged_utterances, 1):
            print(f"\n  Segment {idx}/{len(merged_utterances)}: {utterance.start/1000:.1f}s - {utterance.end/1000:.1f}s")
            print(f"  AssemblyAI Label: Speaker {utterance.speaker}")
            
            text = utterance.text
            speaker_label = f"Speaker {utterance.speaker}"
            verification_result = None
            
            # ============================================
            # FIXED LOGIC - Option 3
            # ============================================
            if doctor_voice_embedding and utterance.speaker == "A":
                # Only verify Speaker A
                print(f"  → Verifying Speaker A...")
                segment_path = None
                try:
                    segment_path = extract_audio_segment(
                        temp_audio_path,
                        utterance.start,
                        utterance.end
                    )
                    
                    verification_result = verify_speaker(
                        segment_path,
                        doctor_voice_embedding,
                        verification_threshold
                    )
                    
                    if verification_result["is_doctor"]:
                        speaker_label = "Doctor ✓"
                        stats["verified_as_doctor"] += 1
                        print(f"  → Doctor ✓ (similarity: {verification_result['similarity']:.3f})")
                    else:
                        speaker_label = "Unknown (failed verification)"
                        stats["failed_verification"] += 1
                        print(f"  → Failed verification (similarity: {verification_result['similarity']:.3f})")
                
                except Exception as e:
                    print(f"  → ⚠️  Verification error: {e}")
                    speaker_label = "Unknown"
                    stats["failed_verification"] += 1
                
                finally:
                    if segment_path and os.path.exists(segment_path):
                        try:
                            os.unlink(segment_path)
                        except:
                            pass
            
            elif doctor_voice_embedding and utterance.speaker != "A":
                # Automatically label as Patient (no verification needed)
                speaker_label = "Patient"
                stats["auto_patient"] += 1
                print(f"  → Patient (auto-labeled, no verification)")
            
            else:
                # No doctor embedding available
                speaker_label = "Doctor" if utterance.speaker == "A" else "Patient"
                print(f"  → {speaker_label} (no verification available)")
            
            # ============================================
            # Translation
            # ============================================
            try:
                print(f"  → Translating...")
                
                script_fix = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Convert to Urdu script. No translation."},
                        {"role": "user", "content": text}
                    ]
                )
                urdu_text = script_fix.choices[0].message.content.strip()
                
                translation = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate Urdu to English."},
                        {"role": "user", "content": urdu_text}
                    ]
                )
                english_text = translation.choices[0].message.content.strip()
                
                print(f"  ✓ Translation complete")
            
            except Exception as e:
                print(f"  → ⚠️  Translation error: {e}")
                urdu_text = text
                english_text = text
            
            # Build conversation entry
            entry = {
                "speaker": speaker_label,
                "assemblyai_label": f"Speaker {utterance.speaker}",
                "timestamp": f"{utterance.start/1000:.2f}s - {utterance.end/1000:.2f}s",
                "original": text,
                "urdu": urdu_text,
                "english": english_text
            }
            
            if verification_result:
                entry["verification"] = verification_result
            
            conversation.append(entry)
            print(f"  ✓ Segment {idx} processed")
        
        print(f"\n✅ Processing complete!")
        print(f"   Total segments: {len(conversation)}")
        print(f"   Verified as Doctor (Speaker A): {stats['verified_as_doctor']}")
        print(f"   Failed verification (Speaker A): {stats['failed_verification']}")
        print(f"   Auto-labeled as Patient (Speaker B/C): {stats['auto_patient']}")
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
                    "strategy": "Only Speaker A verified, others auto-labeled as Patient",
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