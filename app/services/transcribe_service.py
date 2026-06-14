import os
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import assemblyai as aai
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from app.db import get_db,get_doctor_collection
from fastapi import Depends
from bson import ObjectId
import wave
import numpy as np
from scipy.spatial.distance import cosine
import tempfile
import json
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import asyncio
from app.config import FFMPEG_DIR

load_dotenv()

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.http_timeout = 600.0

client = OpenAI(api_key=openai_api_key)
async_client = AsyncOpenAI(api_key=openai_api_key)

import sys

if sys.platform == "win32":
    AudioSegment.converter = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    AudioSegment.ffmpeg = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")
else:
    AudioSegment.converter = "ffmpeg"
    AudioSegment.ffmpeg = "ffmpeg"
    AudioSegment.ffprobe = "ffprobe"

print(f"[OK] FFmpeg path set: {AudioSegment.converter} (Platform: {sys.platform})")

processor = None
model = None

def load_audio_as_numpy_fast(audio_path: str, target_sr: int = 16000) -> tuple:
    """Fast audio loading using AudioSegment and NumPy (alternative to slow librosa.load)"""
    try:
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1.0, 1.0] depending on sample width
        if sound.sample_width == 2:
            samples = samples / 32768.0
        elif sound.sample_width == 4:
            samples = samples / 2147483648.0
        else:
            samples = samples / 128.0
            
        return samples, target_sr
    except Exception as e:
        print(f"[WARN] load_audio_as_numpy_fast failed: {e}. Falling back to librosa.")
        import librosa
        return librosa.load(audio_path, sr=target_sr, mono=True)

def load_transcribe_models():
    global processor, model
    if processor is None or model is None:
        print("[LOAD] Loading HuggingFace Wav2Vec2 model...")
        model_name = "facebook/wav2vec2-base"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()
        print("[OK] Model loaded successfully!\n")

async def get_doctor_embedding(doctor_id: str):
    doctor_collection = get_doctor_collection()
 
 
    if not ObjectId.is_valid(doctor_id):
        raise Exception(f"Invalid doctor_id: {doctor_id}")
    # doctor = await doctor_collection.find_one({"userId": ObjectId(doctor_id)}) #this is for wehn i will run full project
    doctor = await doctor_collection.find_one({"_id": ObjectId(doctor_id)}) # this is for testing only when i will run this file independently fastapi alone
    
    print(f"Doctor data retrieved for ID {doctor_id}: {doctor}")
    
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
        print(f"[OK] File read successfully: {len(content)} bytes")
        
        print(f"\nGetting embeddings for doctor: {doctorId}")
        embeddings = await get_doctor_embedding(doctorId)
        print(f"[OK] Embeddings loaded: {len(embeddings)} values")
        
        print("\nCreating BytesIO object...")
        file_obj = BytesIO(content)
        file_obj.name = file.filename
        print(f"[OK] BytesIO created: {file_obj.name}")
        
        print("\nCalling process_medical_conversation...")
        result = await process_medical_conversation(
            file_obj,
            doctor_voice_embedding=embeddings
        )
        
        print("[OK] Processing complete!")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )







def preprocess_audio(audio_path: str, trim_silence: bool = True) -> str:
    """Preprocess audio: mono, 16kHz, noise reduction, normalization"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        # y = nr.reduce_noise(y=y, sr=sr)
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
        load_transcribe_models()
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


def create_voice_embedding_from_array(audio_array: np.ndarray) -> list:
    """Generate 768-dimensional voice embedding directly from a numpy array"""
    try:
        load_transcribe_models()
        audio = librosa.util.normalize(audio_array)
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            embedding = torch.mean(embeddings, dim=1).squeeze()
        
        return embedding.numpy().tolist()
    except Exception as e:
        print(f"❌ Embedding creation from array failed: {e}")
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


def verify_speaker(segment_audio_path: str, enrolled_embedding: list, threshold: float = 0.40) -> dict:
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


def verify_speaker_from_array(audio_array: np.ndarray, enrolled_embedding: list, threshold: float = 0.40) -> dict:
    """Verify if numpy audio slice matches enrolled voice"""
    try:
        segment_embedding = create_voice_embedding_from_array(audio_array)
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


def get_audio_slice(audio_data: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> np.ndarray:
    """Extract slice from numpy audio data in-memory"""
    start_sample = int((start_ms / 1000.0) * sample_rate)
    end_sample = int((end_ms / 1000.0) * sample_rate)
    return audio_data[start_sample:end_sample]


async def process_medical_conversation(
    file,
    doctor_voice_embedding: list = None,
    verification_threshold: float = 0.40
):
    """
    FIXED VERSION - Option 3:
    - Only verify Speaker A against doctor's voice
    - All other speakers automatically labeled as Patient
    """
    file_content = file.read()
    if asyncio.iscoroutine(file_content):
        file_content = await file_content

    file_bytes = BytesIO(file_content)
    filename = getattr(file, 'name', 'audio.wav')
    file_bytes.name = filename

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_content)
        temp_audio_path = tmp.name

    try:
        print("=" * 60)
        print("[AUDIO] PROCESSING CONVERSATION (FIXED - Option 3)")
        print("=" * 60)

        # ============================================
        # STEP 1: AssemblyAI Diarization
        # ============================================
        print("\n[1/5] [LOAD] Running speaker diarization...")
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            format_text=True,
            punctuate=True,
            speakers_expected=2,
            language_detection=True
        )
        
        transcriber = aai.Transcriber(config=config)
        # Use asyncio.to_thread to run blocking AssemblyAI transcribe call in a thread pool
        transcript = await asyncio.to_thread(transcriber.transcribe, temp_audio_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {"status": "error", "message": transcript.error}
        
        print(f"[OK] Found {len(transcript.utterances)} segments")

        # ============================================
        # STEP 2: Merge Utterances
        # ============================================
        print("\n[2/5] [LOAD] Merging short utterances...")
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
        
        print(f"[OK] Merged to {len(merged_utterances)} segments")

        # Load audio into memory ONCE for extraction and verification (runs in background thread)
        audio_data = None
        sample_rate = 16000
        if doctor_voice_embedding is not None or any(u.speaker == "A" for u in merged_utterances):
            print("\n[Memory] Loading audio into memory...")
            try:
                # Convert to wav first if not already wav (webm/mp4 etc won't load in librosa)
                if not temp_audio_path.endswith(".wav"):
                    converted_path = temp_audio_path + "_converted.wav"
                    sound = AudioSegment.from_file(temp_audio_path)
                    sound = sound.set_frame_rate(16000).set_channels(1)
                    sound.export(converted_path, format="wav")
                    load_path = converted_path
                else:
                    load_path = temp_audio_path

                audio_data, sample_rate = await asyncio.to_thread(
                    librosa.load, load_path, sr=16000, mono=True
                )
                print("[OK] Audio successfully loaded into memory")

                # Cleanup converted file
                if load_path != temp_audio_path and os.path.exists(load_path):
                    os.unlink(load_path)

            except Exception as e:
                print(f"[WARN] Failed to load audio into memory: {e}")
        # ============================================
        # STEP 3: Extract Doctor's Voice Embedding (if not provided)
        # ============================================
        if doctor_voice_embedding is None:
            print("\n[3/5] [LOAD] Extracting doctor's voice embedding from Speaker A...")
            doctor_segments = [u for u in merged_utterances if u.speaker == "A"]
            
            if len(doctor_segments) == 0:
                print("[WARN] No Speaker A found! Skipping voice verification.")
            elif audio_data is None:
                print("[WARN] Audio data not loaded. Skipping voice verification.")
            else:
                doctor_embeddings = []
                segments_used = 0
                
                for utterance in doctor_segments[:5]:
                    try:
                        slice_data = get_audio_slice(audio_data, sample_rate, utterance.start, utterance.end)
                        if len(slice_data) > 0:
                            embedding = await asyncio.to_thread(create_voice_embedding_from_array, slice_data)
                            if embedding is not None:
                                doctor_embeddings.append(embedding)
                                segments_used += 1
                                print(f"     [OK] Extracted embedding from segment {segments_used}")
                    except Exception as e:
                        print(f"     [WARN] Segment skipped: {e}")
                
                if doctor_embeddings:
                    doctor_voice_embedding = np.mean(doctor_embeddings, axis=0).tolist()
                    print(f"[OK] Doctor's voice fingerprint created from {segments_used} segments")
                else:
                    print("[WARN] Could not extract doctor's voice embedding")
        else:
            print("\n[3/5] [OK] Using provided doctor's voice embedding")

        # ============================================
        # STEP 4: Voice Verification & Concurrently Processing Utterances
        # ============================================
        print(f"\n[4/5] [LOAD] Verifying ONLY Speaker A against doctor's voice...")
        print(f"     Threshold: {verification_threshold}")
        print(f"     Rule: Speaker A → verify, Others → auto Patient")


        async def verify_utterance_speaker(idx, utterance):
            """Verify Speaker A's voice against doctor's embedding (local CPU-bound)"""
            speaker_label = f"Speaker {utterance.speaker}"
            verification_result = None
            
            if doctor_voice_embedding and utterance.speaker == "A" and audio_data is not None:
                try:
                    slice_data = get_audio_slice(audio_data, sample_rate, utterance.start, utterance.end)
                    if len(slice_data) > 0:
                        verification_result = await asyncio.to_thread(
                            verify_speaker_from_array,
                            slice_data,
                            doctor_voice_embedding,
                            verification_threshold
                        )
                        if verification_result["is_doctor"]:
                            speaker_label = "Doctor"
                        else:
                            speaker_label = "Unknown (failed verification)"
                    else:
                        speaker_label = "Unknown"
                except Exception as e:
                    print(f"  -> [WARN] Verification error on segment {idx}: {e}")
                    speaker_label = "Unknown"
            elif doctor_voice_embedding and utterance.speaker != "A":
                speaker_label = "Patient"
            else:
                speaker_label = "Doctor" if utterance.speaker == "A" else "Patient"
                
            return idx, utterance, speaker_label, verification_result

        # Step 4.1: Run speaker verification concurrently for all segments
        verification_tasks = [verify_utterance_speaker(idx, u) for idx, u in enumerate(merged_utterances, 1)]
        verified_results = await asyncio.gather(*verification_tasks)

        # Step 4.2: Perform a single batch translation using GPT-4o-mini
        segments_to_translate = [
            {"idx": idx, "speaker": speaker_label, "text": u.text}
            for idx, u, speaker_label, _ in verified_results
        ]
        
        processed_translations = {}
        conversation_summary = {
        "urdu": "خلاصہ دستیاب نہیں",
        "english": "Summary not available"
        }  # ✅ Default defined BEFORE try block
        if segments_to_translate:
            try:
                print(f"\n[OpenAI] Batch translating/refining {len(segments_to_translate)} segments...")
                system_prompt = (
                    "You are a medical scribe assistant. You are given a chronological sequence of transcribed dialogue segments "
                    "from a doctor-patient consultation. The transcription is in Urdu (or Roman Urdu/English mix).\n\n"
                    "Your tasks:\n"
                    "1. Correct and refine the Urdu transcription for each segment. Convert any phonetically transcribed Urdu to proper Urdu script. "
                    "Do NOT add any medical advice, responses, or extra conversation. Keep the meaning exactly as what was spoken in the segment.\n"
                    "2. Translate each segment to English.\n"
                    "3. After processing all segments, generate a concise summary of the entire conversation in both Urdu and English. "
                    "The summary should capture the key points discussed, complaints mentioned, and any conclusions or advice given during the consultation.\n\n"
                    "Return the output strictly as a JSON object with the following keys:\n"
                    "- 'processed_segments': a list of objects, each with:\n"
                    "    - 'idx': the index of the segment (must match the input idx)\n"
                    "    - 'urdu': the corrected Urdu script\n"
                    "    - 'english': the English translation\n"
                    "- 'summary': an object with:\n"
                    "    - 'urdu': a concise summary of the full conversation in Urdu script\n"
                    "    - 'english': a concise summary of the full conversation in English\n\n"
                    "Match the exact 'idx' for each segment."
                )
                
                response = await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(segments_to_translate, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"}
                )
                
                response_data = json.loads(response.choices[0].message.content)
                for item in response_data.get("processed_segments", []):
                    processed_translations[item["idx"]] = {
                        "urdu": item["urdu"],
                        "english": item["english"]
                    }
                # Extract summary from response
                conversation_summary = response_data.get("summary", conversation_summary)  

                print("[OK] Batch translation complete!")
            except Exception as e:
                print(f"[WARN] OpenAI Batch translation failed: {e}")

        # Step 4.3: Build conversation list
        conversation = []
        for idx, u, speaker_label, verification_result in verified_results:
            translation_data = processed_translations.get(idx, {"urdu": u.text, "english": u.text})
            
            entry = {
                "speaker": speaker_label,
                "assemblyai_label": f"Speaker {u.speaker}",
                "timestamp": f"{u.start/1000:.2f}s - {u.end/1000:.2f}s",
                "original": u.text,
                "urdu": translation_data["urdu"],
                "english": translation_data["english"]
            }
            
            if verification_result:
                entry["verification"] = verification_result
                
            conversation.append(entry)

        # Compute stats
        stats = {
            "verified_as_doctor": 0,
            "verified_as_patient": 0,
            "failed_verification": 0,
            "auto_patient": 0
        }
        similarities = []
        for entry in conversation:
            lbl = entry["speaker"]
            if lbl == "Doctor":
                stats["verified_as_doctor"] += 1
            elif lbl == "Unknown (failed verification)":
                stats["failed_verification"] += 1
            elif lbl == "Patient":
                stats["auto_patient"] += 1

            # If there was a verification result, collect the similarity score
            ver = entry.get("verification")
            if ver and "similarity" in ver:
                similarities.append(ver["similarity"])

        avg_similarity = round(float(np.mean(similarities)), 4) if similarities else 0.0

        # Print premium console statistics dashboard for voice verification
        print("\n" + "="*80)
        print("🎙️  VOICE VERIFICATION ACCURACY & STATISTICS REPORT")
        print("="*80)
        print(f"🔹 Voice Verification Enabled: {doctor_voice_embedding is not None}")
        print(f"🔹 Verification Strategy     : Only Speaker A verified, others auto-labeled as Patient")
        print(f"🔹 Verification Threshold    : {verification_threshold}")
        print(f"🔹 Average Speaker Similarity: {avg_similarity}")
        print(f"🔹 Total Dialog Segments     : {len(conversation)}")
        print("-"*80)
        print(f"📈 DETAILED SEGMENT STATS:")
        print(f"   - Verified as Doctor (Speaker A)         : {stats['verified_as_doctor']}")
        print(f"   - Failed Voice Verification (Speaker A)  : {stats['failed_verification']}")
        print(f"   - Auto-labeled as Patient (Speaker B/C)  : {stats['auto_patient']}")
        print(f"   - Total Verified Speaker A Segments      : {len(similarities)}")
        if similarities:
            print(f"   - Max Similarity Score Recorded          : {max(similarities)}")
            print(f"   - Min Similarity Score Recorded          : {min(similarities)}")
        print("="*80 + "\n")
        
        return {
            "status": "success",
            "full_transcript": transcript.text,
            "conversation": conversation,
            "summary":conversation_summary,
            "metadata": {
                "duration": f"{transcript.audio_duration}s",
                "total_segments": len(conversation),
                "speakers_detected": len(set(u.speaker for u in transcript.utterances)),
                "verification": {
                    "enabled": doctor_voice_embedding is not None,
                    "threshold": verification_threshold,
                    "strategy": "Only Speaker A verified, others auto-labeled as Patient",
                    "stats": stats,
                    "average_similarity": avg_similarity
                }
            }
        }
    
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

