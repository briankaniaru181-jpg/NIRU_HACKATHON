from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import requests
from typing import List, Dict, Optional
import google.generativeai as genai

app = FastAPI(title="CBC Service")

# ── API Keys ──────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ── Try Gemini models ─────────────────────────────────────
print("🔄 Initializing Gemini for CBC...")
try:
    gemini_client = genai.GenerativeModel("gemini-3.1-flash-lite")
    test_response = gemini_client.generate_content("Sema jambo")
    print(f"✅ Gemini 3.1 Flash-Lite connected successfully!")
    selected_model = "gemini-3.1-flash-lite"
except Exception as e:
    print(f"⚠️ Gemini 3.1 Flash-Lite failed: {e}")
    try:
        gemini_client = genai.GenerativeModel("gemini-2.0-flash-lite")
        test_response = gemini_client.generate_content("Sema jambo")
        print(f"✅ Gemini 2.0 Flash-Lite connected successfully!")
        selected_model = "gemini-2.0-flash-lite"
    except Exception as e2:
        print(f"⚠️ Gemini 2.0 Flash-Lite failed: {e2}")
        try:
            gemini_client = genai.GenerativeModel("gemini-1.5-flash")
            test_response = gemini_client.generate_content("Sema jambo")
            print(f"✅ Gemini 1.5 Flash connected successfully!")
            selected_model = "gemini-1.5-flash"
        except Exception as e3:
            print(f"❌ All Gemini models failed! Error: {e3}")
            gemini_client = None
            selected_model = "None"

# ── Request / Response models ─────────────────────────────
class CBCRequest(BaseModel):
    query: str
    subject: str
    chapter: Optional[str] = None

class CBCResponse(BaseModel):
    answer: str
    subject: str
    source: str = "Opiq CBC"
    confidence: float = 0.0

# ── Opiq subject URLs ─────────────────────────────────────
CBC_REGISTRY = {
    "kilimo_f1": {
        "name": "Kilimo",
        "url": "https://opiq.co.ke/books/kilimo-f1",
        "grade": "Form 1"
    },
    "biolojia_f1": {
        "name": "Biolojia",
        "url": "https://opiq.co.ke/books/biolojia-f1",
        "grade": "Form 1"
    },
    "kemia_f1": {
        "name": "Kemia",
        "url": "https://opiq.co.ke/books/kemia-f1",
        "grade": "Form 1"
    },
    "fizikia_f1": {
        "name": "Fizikia",
        "url": "https://opiq.co.ke/books/fizikia-f1",
        "grade": "Form 1"
    },
    "hisabati_f1": {
        "name": "Hisabati",
        "url": "https://opiq.co.ke/books/hisabati-f1",
        "grade": "Form 1"
    },
}

# ── Opiq scraper ──────────────────────────────────────────
def fetch_opiq_content(subject_id: str) -> str:
    """Fetch chapter content from Opiq"""
    if subject_id not in CBC_REGISTRY:
        return ""

    subject = CBC_REGISTRY[subject_id]
    url = subject["url"]

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', ' ', response.text)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Return first 4000 chars as context
            return text[:4000]
    except Exception as e:
        print(f"Opiq fetch error: {e}")

    return ""

# ── Gemini generation ─────────────────────────────────────
def generate_answer(
    query: str,
    context: str,
    subject_name: str
) -> str:
    """Generate CBC answer using Gemini"""
    if not gemini_client:
        return "Samahani, mfumo wa Gemini haufanyi kazi. Tafadhali jaribu tena baadaye."
    
    try:
        prompt = f"""Wewe ni msaidizi wa CBC (Competency-Based Curriculum) wa Kenya.
Unafundisha somo la {subject_name} kwa Kiswahili.

KANUNI:
1. Jibu kwa Kiswahili sanifu PEKEE
2. Tumia taarifa zilizotolewa kwenye MUKTADHA
3. Jibu liwe na maneno 80-150
4. Ikiwa jibu halipo kwenye muktadha, sema hivyo kwa uaminifu
5. Tumia lugha rahisi inayofaa wanafunzi wa sekondari

MUKTADHA WA CBC:
{context[:3000]}

SWALI LA MWANAFUNZI: {query}

JIBU:"""

        response = gemini_client.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"Gemini error: {e}")
        return "Samahani, kuna hitilafu. Tafadhali jaribu tena."

# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cbc",
        "gemini_model": selected_model,
        "subjects_available": list(CBC_REGISTRY.keys())
    }

@app.get("/subjects")
def list_subjects():
    return {
        "subjects": [
            {
                "id": k,
                "name": v["name"],
                "grade": v["grade"],
                "url": v["url"]
            }
            for k, v in CBC_REGISTRY.items()
        ]
    }

@app.post("/query", response_model=CBCResponse)
def query_cbc(request: CBCRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    if request.subject not in CBC_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Subject '{request.subject}' not found. "
                   f"Available: {list(CBC_REGISTRY.keys())}"
        )

    subject = CBC_REGISTRY[request.subject]
    subject_name = subject["name"]

    # Fetch content from Opiq
    print(f"📚 Fetching Opiq content for {request.subject}...")
    context = fetch_opiq_content(request.subject)

    if not context:
        print("⚠️ Opiq fetch failed, using Gemini general knowledge")
        context = (
            f"Somo la {subject_name} katika mtaala wa CBC Kenya. "
            f"Jibu swali kwa ujuzi wako wa somo hili."
        )
        confidence = 0.5
    else:
        print(f"✅ Opiq content fetched successfully ({len(context)} chars)")
        confidence = 0.85

    # Generate answer
    answer = generate_answer(request.query, context, subject_name)

    return CBCResponse(
        answer=answer,
        subject=subject_name,
        source="Opiq CBC",
        confidence=confidence
    )

@app.get("/search")
def search_subject(query: str = ""):
    """Search for subjects by name"""
    if not query:
        return {"results": []}
    
    results = []
    query_lower = query.lower()
    for subject_id, subject in CBC_REGISTRY.items():
        if query_lower in subject["name"].lower() or query_lower in subject_id.lower():
            results.append({
                "id": subject_id,
                "name": subject["name"],
                "grade": subject["grade"]
            })
    
    return {"results": results}