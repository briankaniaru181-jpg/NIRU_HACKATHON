from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import requests
from typing import List, Dict, Optional
import google.generativeai as genai
from bs4 import BeautifulSoup

app = FastAPI(title="CBC Service")

# ── API Keys ──────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ── Initialize Gemini ─────────────────────────────────────
print("🔄 Initializing Gemini for CBC...")
print(f"API Key set: {bool(GOOGLE_API_KEY)}")

gemini_client = None
selected_model = "None"

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
            print(f"❌ All Gemini models failed!")
            gemini_client = None
            selected_model = "None"

print(f"Final status - Model: {selected_model}, Client: {bool(gemini_client)}")

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

# ── Opiq subject URLs (ONLY 3 SUBJECTS) ──────────────────
CBC_REGISTRY = {
    "kilimo_f1": {
        "name": "Kilimo",
        "url": "https://opiq.co.ke/kit/78/chapter/3980",
        "grade": "Form 1"
    },
    "biolojia_f1": {
        "name": "Biolojia",
        "url": "https://opiq.co.ke/kit/36/chapter/1579",
        "grade": "Form 1"
    },
    "kemia_f1": {
        "name": "Kemia",
        "url": "https://opiq.co.ke/kit/37/chapter/1599",
        "grade": "Form 1"
    }
}

# ── Opiq noise filter (from your Kaggle code) ────────────
OPIQ_NOISE = {
    "more like this", "more options", "copy link", "report mistake",
    "find more content on the subject", "chapter contents",
    "go to main content", "main menu", "context menu",
    "opiq score", "tasks", "chapter is studied",
    "text assistance trial version", "add content", "add file", "add text",
    "files to be added", "report error", "opiq uses cookies",
    "allow mandatory", "please wait", "log in", "join", "search",
    "about", "library", "accessibility", "eula", "privacy notice",
    "use of cookies", "terms and conditions", "service provided by",
    "in english", "loading", "success", "cancel", "back", "send",
    "remove", "close menu", "error", "close", "nbs!", "nb!", "order",
}

# ── Opiq scraper with BeautifulSoup ──────────────────────
def fetch_opiq_content(subject_id: str) -> str:
    """Fetch chapter content from Opiq using the correct chapter URLs"""
    if subject_id not in CBC_REGISTRY:
        return ""

    subject = CBC_REGISTRY[subject_id]
    url = subject["url"]

    try:
        print(f"🌐 Fetching: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", 
                        "header", "button", "form", "iframe", "noscript"]):
            tag.decompose()
        
        # Get text and clean it
        raw_text = soup.get_text(separator="\n", strip=True)
        raw_text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", raw_text)
        raw_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text)
        raw_text = raw_text.replace("–", " – ")
        
        # Filter lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        lines = [re.sub(r"[ \t]+", " ", line) for line in lines]
        
        content_lines = []
        for line in lines:
            line_lower = line.lower()
            if len(line) < 4:
                continue
            if line_lower in OPIQ_NOISE:
                continue
            if line.startswith("http") or line.startswith("P.O") or line.startswith("+"):
                continue
            if "@" in line and "." in line:
                continue
            content_lines.append(line)
        
        cleaned = "\n".join(content_lines)
        print(f"✅ Fetched {len(cleaned)} chars from Opiq")
        return cleaned[:4000]
        
    except Exception as e:
        print(f"❌ Opiq fetch error: {e}")
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
        return f"Samahani, kuna hitilafu: {str(e)[:50]}"

# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cbc",
        "gemini_model": selected_model,
        "gemini_working": bool(gemini_client),
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