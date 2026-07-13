from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import requests
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
import wikipediaapi
import wikipedia

app = FastAPI(title="General Chat Service")

# ── API Keys ─────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_client = genai.GenerativeModel("gemini-1.5-flash")

# ── Request / Response models ─────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict] = []

# ── Entity mappings ───────────────────────────────────────
ENTITY_MAPPINGS = {
    "gavana wa mombasa": "Abdulswamad Nassir",
    "mombasa governor": "Abdulswamad Nassir",
    "barack obama": "Barack Obama",
    "raila odinga": "Raila Odinga",
    "william ruto": "William Ruto",
    "rais wa kenya": "William Ruto",
    "president of kenya": "William Ruto",
}

CREATIVE_PREFIXES = [
    "andika", "tunga", "badilisha", "fasiri", "tafsiri",
    "sahihisha", "unda", "fupisha", "panua", "translate",
    "eleza maana", "fafanua"
]

IDENTITY_TRIGGERS = [
    "wewe ni nani", "jina lako", "unaitwa nani",
    "wewe ni nini", "uliundwa na nani", "sauti ni nini"
]

# ── Helpers ───────────────────────────────────────────────
def is_creative(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in CREATIVE_PREFIXES)

def is_identity(query: str) -> bool:
    q = re.sub(r'[?!.,;:]', '', query.lower().strip())
    return any(t in q for t in IDENTITY_TRIGGERS)

def get_search_term(query: str) -> str:
    q = query.lower()
    for key, value in ENTITY_MAPPINGS.items():
        if key in q:
            return value
    return query

# ── Tavily ────────────────────────────────────────────────
def search_tavily(query: str) -> Optional[Dict]:
    if not TAVILY_API_KEY:
        return None
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,
                "exclude_domains": [
                    "facebook.com", "instagram.com",
                    "twitter.com", "tiktok.com"
                ],
            },
            timeout=12
        )
        data = r.json()
        if data.get("answer"):
            results = data.get("results", [])
            url = results[0].get("url", "") if results else ""
            return {
                "content": data["answer"],
                "source": "Tavily",
                "url": url
            }
        results = data.get("results", [])
        if results:
            txt = "\n\n".join(
                f"[{r.get('title','')}] {r.get('content','')}"
                for r in results
            )
            return {
                "content": txt,
                "source": "Tavily",
                "url": results[0].get("url", "")
            }
    except Exception as e:
        print(f"Tavily error: {e}")
    return None

# ── Wikipedia ─────────────────────────────────────────────
def search_wikipedia(query: str) -> Tuple[str, List[Dict]]:
    wiki_en = wikipediaapi.Wikipedia(
        language='en',
        user_agent='SautiAI/1.0'
    )
    wiki_sw = wikipediaapi.Wikipedia(
        language='sw',
        user_agent='SautiAI/1.0'
    )
    term = get_search_term(query)
    context = ""
    sources = []

    try:
        # Search English Wikipedia first
        wikipedia.set_lang("en")
        en_results = wikipedia.search(term, results=3)
        for title in en_results:
            page = wiki_en.page(title)
            if page.exists():
                context += f"[Wikipedia EN - {page.title}]\n{page.summary}\n\n"
                sources.append({
                    'title': page.title,
                    'url': page.fullurl,
                    'language': 'en'
                })
                break

        # Search Swahili Wikipedia
        if not context:
            wikipedia.set_lang("sw")
            sw_results = wikipedia.search(term, results=3)
            for title in sw_results:
                page = wiki_sw.page(title)
                if page.exists():
                    context += f"[Wikipedia SW - {page.title}]\n{page.summary}\n\n"
                    sources.append({
                        'title': page.title,
                        'url': page.fullurl,
                        'language': 'sw'
                    })
                    break
    except Exception as e:
        print(f"Wikipedia error: {e}")

    return context, sources

# ── Gemini generation ─────────────────────────────────────
def generate_with_gemini(query: str, context: str) -> str:
    try:
        prompt = (
            "Wewe ni msaidizi wa Kiswahili. "
            "Tumia TAARIFA zilizotolewa PEKEE. "
            "Jibu kwa Kiswahili. Maneno 50-150 tu.\n\n"
            f"TAARIFA:\n{context[:2500]}\n\n"
            f"SWALI: {query}\nJIBU:"
        )
        response = gemini_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Samahani, sikupata jibu. Tafadhali jaribu tena."

# ── Main endpoint ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.message.strip()

    if not query:
        return ChatResponse(
            answer="Tafadhali andika swali lako.",
            sources=[]
        )

    # Identity questions
    if is_identity(query):
        return ChatResponse(
            answer=(
                "Mimi ni Sauti, msaidizi wa AI "
                "aliyeundwa kufikiria kwa njia ya Kiafrika. "
                "Ninaweza kukusaidia na maswali ya jumla, "
                "elimu, na mazungumzo kwa Kiswahili."
            ),
            sources=[]
        )

    # Creative/grammar — straight to Gemini no RAG
    if is_creative(query):
        try:
            response = gemini_client.generate_content(
                f"Jibu kwa Kiswahili:\n{query}"
            )
            return ChatResponse(
                answer=response.text.strip(),
                sources=[]
            )
        except Exception as e:
            return ChatResponse(
                answer="Samahani, jaribu tena.",
                sources=[]
            )

    # Try Tavily first
    context = ""
    sources = []

    tavily_result = search_tavily(query)
    if tavily_result:
        context = tavily_result["content"]
        sources = [{
            "title": tavily_result["source"],
            "url": tavily_result["url"],
            "language": "en"
        }]

    # Fallback to Wikipedia
    if not context:
        context, sources = search_wikipedia(query)

    if not context:
        return ChatResponse(
            answer="Samahani, sikupata taarifa za kutosha kujibu swali hili.",
            sources=[]
        )

    answer = generate_with_gemini(query, context)
    return ChatResponse(answer=answer, sources=sources)

@app.get("/health")
def health():
    return {"status": "ok", "service": "general-chat"}