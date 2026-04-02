# ClinicalBERT is a base model - It's designed for embeddings, not direct NER or classification

# ClinicalBERT output: 768-dimensional vectors
# We need: Labeled entities (DISEASE, DURATION, SEVERITY)


# Task-specific fine-tuning is required:

# BC5CDR model = ClinicalBERT fine-tuned on disease/chemical NER
# BERT-NER = BERT fine-tuned on temporal entities (CoNLL-2003)
# BART-MNLI = BART fine-tuned for natural language inference
import re
import os
import spacy
from transformers import AutoTokenizer, AutoModel
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nlp = None
tokenizer = None
model = None
import os
print("Current working dir:", os.getcwd())

def load_models():
    global nlp, tokenizer, model

    print("🔄 Loading SciSpacy...")
    model_path="./models/en_ner_bc5cdr_md-0.5.4"
    nlp = spacy.load(model_path)
    print("✅ SciSpacy loaded")


def extract_severity(text):
    severities = ["mild", "moderate", "severe", "chronic", "acute", "weak", "heavy", "intense"]
    for s in severities:
        if re.search(rf"\b{s}\b", text.lower()):
            return s
    return None


def extract_duration(text):
    # Numeric patterns
    pattern = r"(\d+\s?(days?|weeks?|months?|years?))"
    match = re.search(pattern, text.lower())
    if match:
        return match.group(1)
    
    # Word-based patterns (three, four, five, etc.)
    word_pattern = r"(three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?)"
    word_match = re.search(word_pattern, text.lower())
    if word_match:
        return word_match.group(0)
    
    return None


def extract_medical_entities(text):
    doc = nlp(text)

    diseases = set()
    chemicals = set()

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            diseases.add(ent.text)
        elif ent.label_ == "CHEMICAL":
            chemicals.add(ent.text)

    return list(diseases), list(chemicals)
def extract_medical_information(text):
    diseases, chemicals = extract_medical_entities(text)
    severity = extract_severity(text)
    duration = extract_duration(text)

    return {
        "input_text": text,
        "diseases": diseases,
        "chemicals_drugs": chemicals,
        "severity": severity,
        "duration": duration
    }


# import re
# import os
# import spacy
# from transformers import AutoTokenizer, AutoModel, pipeline

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# nlp = None
# tokenizer = None
# model = None

# # NEW: ML models for duration and severity
# temporal_ner = None
# severity_classifier = None

# print("Current working dir:", os.getcwd())

# def load_models():
#     global nlp, tokenizer, model, temporal_ner, severity_classifier

#     print("🔄 Loading ClinicalBERT...")
#     tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#     model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#     print("✅ ClinicalBERT loaded")

#     print("🔄 Loading SciSpacy...")
#     model_path = "./models/en_ner_bc5cdr_md-0.5.4"
#     nlp = spacy.load(model_path)
#     print("✅ SciSpacy loaded")

#     # NEW: Load duration extraction model
#     print("🔄 Loading Duration Extractor (BERT-NER)...")
#     temporal_ner = pipeline(
#         "ner",
#         model="./models/bert-base-NER",
#         tokenizer="./models/bert-base-NER",
#         aggregation_strategy="simple",
#         device=-1
#     )
#     print("✅ Duration Extractor loaded")

#     # NEW: Load severity classifier (LOCAL)
#     print("🔄 Loading Severity Classifier (BART)...")
#     severity_classifier = pipeline(
#         "zero-shot-classification",
#         model="./models/bart-large-mnli",
#         tokenizer="./models/bart-large-mnli",
#         device=-1
#     )
#     print("✅ Severity Classifier loaded")


# def extract_severity(text):
#     """ML-based severity extraction using zero-shot classification"""
    
#     # Check if text contains medical symptoms
#     symptom_keywords = ['pain', 'ache', 'fever', 'symptom', 'feel', 'sick', 'headache', 'weak']
#     if not any(kw in text.lower() for kw in symptom_keywords):
#         return None
    
#     # Severity labels
#     severity_labels = ["mild", "moderate", "severe", "chronic", "acute"]
    
#     # Use zero-shot classification
#     result = severity_classifier(
#         text,
#         candidate_labels=severity_labels,
#         hypothesis_template="The patient's condition is {}.",
#         multi_label=False
#     )
    
#     # Return highest confidence label if confidence > 0.4
#     if result['scores'][0] > 0.4:
#         return result['labels'][0]
    
#     return None


# def extract_duration(text):
#     """ML-based duration extraction using BERT-NER"""
    
#     # Use NER model to detect temporal entities
#     entities = temporal_ner(text)
    
#     # Keywords that indicate duration
#     duration_keywords = ['day', 'week', 'month', 'year', 'hour']
    
#     # Look for duration-related entities
#     for entity in entities:
#         entity_text = entity['word'].lower()
#         # Check if entity contains duration keywords
#         if any(kw in entity_text for kw in duration_keywords):
#             return entity['word']
    
#     # Fallback: Enhanced regex patterns
#     # Numeric patterns: "3 days", "5 weeks"
#     pattern = r"(\d+\s?(days?|weeks?|months?|years?))"
#     match = re.search(pattern, text.lower())
#     if match:
#         return match.group(1)
    
#     # Word-based patterns: "three days", "four weeks"
#     word_pattern = r"(one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?)"
#     word_match = re.search(word_pattern, text.lower())
#     if word_match:
#         return word_match.group(0)
    
#     return None


# def extract_medical_entities(text):
#     """Extract diseases and chemicals using SciSpacy"""
#     doc = nlp(text)

#     diseases = set()
#     chemicals = set()

#     for ent in doc.ents:
#         if ent.label_ == "DISEASE":
#             diseases.add(ent.text)
#         elif ent.label_ == "CHEMICAL":
#             chemicals.add(ent.text)

#     return list(diseases), list(chemicals)


# def extract_medical_information(text):
#     """Complete medical information extraction with ML models"""
#     diseases, chemicals = extract_medical_entities(text)
#     severity = extract_severity(text)  # ML-based
#     duration = extract_duration(text)  # ML-based

#     return {
#         "input_text": text,
#         "diseases": diseases,
#         "chemicals_drugs": chemicals,
#         "severity": severity,
#         "duration": duration
#     }