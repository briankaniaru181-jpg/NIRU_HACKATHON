import sys
import subprocess
import importlib
import gradio as gr
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
import os
import time
import requests
import tempfile
from pathlib import Path
from datetime import datetime
import gc

import torch
import torchaudio
import wikipediaapi
import wikipedia
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from pydub import AudioSegment
from tqdm import tqdm
import yt_dlp
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from kaggle_secrets import UserSecretsClient
from groq import Groq
import base64

# =============================
# API KEYS
# =============================
secrets = UserSecretsClient()
os.environ["TAVILY_API_KEY"] = secrets.get_secret("TAVILY_API_KEY")
groq_client = Groq(api_key=secrets.get_secret("GROQ_API_KEY"))

print("All imports successful")

# =============================
# DEVICE & MODEL LOADING
# =============================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

if device == "cuda":
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"📊 Starting with {free_mem:.2f} MiB free VRAM")

print("\n📥 Loading fine-tuned Swahili-Gemma model...")
start = time.time()
MODEL_PATH = "/kaggle/input/notebooks/briangreenheart/finetuninggood/swahili-gemma-finetuned/merged_model"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    local_files_only=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"✅ Model loaded in {time.time() - start:.2f} seconds")

print("\n📥 Loading embedding model...")
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
print("✅ Embedding model loaded")

# =============================
# GENERATION SETTINGS (improved defaults)
# =============================
generation_config = {
    "max_new_tokens": 220,
    "do_sample": True,
    "temperature": 0.30, # default / no-context path
    "top_p": 0.95,
    "top_k": 64,
    "repetition_penalty": 1.18, # lowered from 1.3 — critical fix
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "use_cache": True,
}
print("\n⚙️ Generation Settings:")
for k, v in generation_config.items():
    if k not in ['pad_token_id', 'eos_token_id', 'use_cache']:
        print(f" • {k}: {v}")

# =============================
# KISWAHILI LITERATURE RAG (unchanged)
# =============================
class SwahiliLiteratureRAG:
    def __init__(self, knowledge_base_path: str = None, model_path: str = None):
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.model = self.load_model(model_path)
        self.similarity_threshold = 0.15
        self.swahili_stop_words = {
            'ni', 'nini', 'na', 'ya', 'za', 'wa', 'la', 'kwa', 'katika', 'au', 'je',
            'hii', 'ile', 'hilo', 'hiyo', 'hayo', 'hao', 'yule', 'huyu', 'huu',
            'kuhusu', 'juu', 'chini', 'mbele', 'nyuma', 'ndani', 'nje', 'karibu',
            'mbali', 'hapa', 'pale', 'kule', 'sasa', 'jana', 'kesho', 'leo'
        }

    def load_knowledge_base(self, path: str = None) -> List[Dict]:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
        return [
            {"instruction": "Eleza kuhusu Fasihi Simulizi.", "input": "", "output": "Fasihi simulizi ni sanaa inayotumia lugha kuwasilisha ujumbe unaomhusu binadamu..."},
            {"instruction": "Tofautisha Fasihi na Sanaa Nyingine.", "input": "", "output": "Fasihi hutumia lugha na wahusika kuwasilisha maudhui..."},
            {"instruction": "Tofautisha Fasihi Simulizi na Fasihi Andishi.", "input": "", "output": "Fasihi simulizi huwasilishwa kwa mdomo... Fasihi andishi huwasilishwa kwa maandishi..."},
            {"instruction": "Eleza vipengele vya Fasihi Simulizi.", "input": "", "output": "Vipengele vya fasihi simulizi ni pamoja na lugha, mandhari, wahusika, maudhui, na mtindo."}
        ]

    def load_model(self, path: str = None):
        if not path or not os.path.exists(path):
            print("Literature RAG: no model path found. Using knowledge base only.")
            return None
        try:
            print(f"Loading literature RAG model from: {path}")
            model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True, use_cache=True, attn_implementation="eager"
            )
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float32)
            print("Literature RAG model loaded!")
            return pipe
        except Exception as e:
            print(f"Error loading literature model: {e}")
            return None

    def normalize_text(self, text: str) -> str:
        return re.sub(r'[.,!?;:()]', '', text.lower().strip())

    def extract_key_terms(self, text: str) -> List[str]:
        words = self.normalize_text(text).split()
        return [w for w in words if len(w) > 2 and w not in self.swahili_stop_words]

    def preprocess_query(self, query: str) -> str:
        processed = self.normalize_text(query)
        patterns = [
            (r'^(je,?\s*)?(.*?)\s*ni\s*nini\??$', r'\2'),
            (r'^(je,?\s*)?nini\s*(ni|kuhusu)\s*(.*?)\??$', r'\3'),
            (r'^eleza\s*(kuhusu\s*)?(.*?)$', r'\2'),
            (r'^tofautisha\s*(.*?)$', r'\1'),
            (r'^(je,?\s*)?una\s*maelezo\s*(ya|kuhusu)\s*(.*?)\??$', r'\3'),
        ]
        for pattern, replacement in patterns:
            if re.match(pattern, processed, re.IGNORECASE):
                processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE).strip()
                break
        return processed

    def calculate_similarity(self, query: str, target: str) -> float:
        query_terms = self.extract_key_terms(query)
        target_terms = self.extract_key_terms(target)
        if not query_terms or not target_terms:
            return 0.0
        exact_matches = len(set(query_terms) & set(target_terms))
        partial_score = 0
        for q_term in query_terms:
            for t_term in target_terms:
                if len(q_term) > 3 and len(t_term) > 3:
                    if q_term in t_term or t_term in q_term:
                        partial_score += 0.7
                    elif q_term[:4] == t_term[:4]:
                        partial_score += 0.5
        return min((exact_matches + partial_score) / max(len(query_terms), len(target_terms)), 1.0)

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        processed_query = self.preprocess_query(query)
        scored_docs = []
        for i, doc in enumerate(self.knowledge_base):
            inst_sim = self.calculate_similarity(processed_query, doc['instruction'])
            out_sim = self.calculate_similarity(processed_query, doc['output'])
            bonus = 0.3 if ('fasihi simulizi' in self.normalize_text(query) and
                            'fasihi simulizi' in self.normalize_text(doc['instruction'] + ' ' + doc['output'])) else 0
            max_sim = min(max(inst_sim, out_sim) + bonus, 1.0)
            if max_sim > self.similarity_threshold:
                scored_docs.append({'doc': doc, 'similarity': max_sim, 'index': i})
        scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_docs[:top_k]

    def answer_query(self, query: str) -> Tuple[str, str]:
        if not query.strip():
            return "Tafadhali andika swali lako.", ""
        retrieved_docs = self.retrieve_documents(query)
        if not retrieved_docs:
            return """Samahani, sijapata maelezo yanayolingana na swali lako katika hifadhidata yangu.

Unaweza kuuliza maswali kuhusu:
- Fasihi simulizi na fasihi andishi
- Vipengele vya fasihi
- Sarufi ya Kiswahili
- Utamaduni wa Kiafrika

Au jaribu kuongeza maneno muhimu zaidi katika swali lako.""", "Hakuna data inayolingana"
        best_doc = retrieved_docs[0]
        confidence = best_doc['similarity']
        if confidence > 0.75:
            return best_doc['doc']['output'], f"High confidence: {confidence:.1%}"
        elif confidence > 0.4:
            combined = "\n\n".join(f"*Sehemu {i+1}:* {d['doc']['output']}" for i, d in enumerate(retrieved_docs[:2]))
            return combined, f"Combined knowledge: {confidence:.1%}"
        else:
            return (f"*Maelezo yanayokaribiana:* {best_doc['doc']['output']}\n\n"
                    "Kumbuka: Jibu hili linaweza lisilingane kabisa na swali lako."), f"Low confidence: {confidence:.1%}"


# =============================
# CREATIVE / GRAMMAR DETECTION
# =============================
CREATIVE_PREFIXES = [
    "andika", "tunga", "badilisha", "fasiri", "tafsiri",
    "sahihisha", "unda", "ongeza", "punguza", "fupisha", "panua",
    "sema kwa", "translate", "eleza maana", "fafanua"
    # deliberately removed: "unadhani", "unafikiri" → now treated more carefully
]
def is_creative_or_grammar_query(query: str) -> bool:
    q = query.lower().strip()
    for prefix in CREATIVE_PREFIXES:
        if prefix in q:
            print(f" 🎨 Creative/grammar detected ('{prefix}') — skipping RAG")
            return True
    return False

# =============================
# WIKIPEDIA & ENTITY MAPPINGS
# =============================
class WikipediaKnowledgeBase:
    def __init__(self, user_agent='SwahiliRAG/1.0'):
        self.wiki_en = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
        self.wiki_sw = wikipediaapi.Wikipedia(language='sw', user_agent=user_agent)
        self.documents = []
        self.index = None
        self.page_cache = {}
        self.entity_mappings = {
            "gavana wa mombasa": "Abdulswamad Nassir",
            "mombasa governor": "Abdulswamad Nassir",
            "barack obama": "Barack Obama",
            "raila odinga": "Raila Odinga",
            "hassan joho": "Hassan Joho",
            "william ruto": "William Ruto",
            "rais wa kenya": "William Ruto",
            "president of kenya": "William Ruto",
        }

    def get_best_search_term(self, query: str) -> str:
        query_lower = query.lower()
        for key, value in self.entity_mappings.items():
            if key in query_lower:
                print(f" 🔍 Mapped → '{value}'")
                return value
        return query

    def search_and_fetch(self, query: str, max_results: int = 3) -> List[Dict]:
        if query in self.page_cache:
            return self.page_cache[query]
        articles = []
        term = self.get_best_search_term(query)
        print(f" 🔎 Searching: '{term}'")
        try:
            wikipedia.set_lang("en")
            for title in wikipedia.search(term, results=max_results):
                page = self.wiki_en.page(title)
                if page.exists():
                    articles.append({'title': page.title, 'content': page.summary, 'url': page.fullurl, 'language': 'en'})
                    break
            wikipedia.set_lang("sw")
            for title in wikipedia.search(term, results=2):
                page = self.wiki_sw.page(title)
                if page.exists():
                    articles.append({'title': page.title, 'content': page.summary, 'url': page.fullurl, 'language': 'sw'})
                    break
        except Exception as e:
            print(f" ⚠️ Wikipedia error: {e}")
        self.page_cache[query] = articles
        return articles

    def create_vector_store(self, articles: List[Dict]) -> int:
        self.documents = []
        for article in articles:
            chunks = [article['content'][i:i+500] for i in range(0, len(article['content']), 450)]
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'title': article['title'],
                    'content': chunk,
                    'url': article['url'],
                    'language': article.get('language', 'en'),
                    'chunk_id': i
                })
        if self.documents:
            texts = [f"passage: {d['content']}" for d in self.documents]
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings.astype('float32'))
        return len(self.documents)

    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[Dict]:
        if not self.index or not self.documents:
            return []
        q_emb = embedding_model.encode([f"query: {query}"])
        _, indices = self.index.search(q_emb.astype('float32'), k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

# =============================
# TAVILY + DDG (simplified)
# =============================
class TavilyRetriever:
    def __init__(self):
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        self.endpoint = "https://api.tavily.com/search"
    def search(self, query: str, max_results: int = 3) -> Optional[Dict]:
        if not self.api_key:
            print(" ⚠️ No TAVILY_API_KEY")
            return None
        try:
            r = requests.post(self.endpoint, json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True,
            
            
            }, timeout=12)
            data = r.json()
            if data.get("answer"):
                results = data.get("results", [])
                url = results[0].get("url", "") if results else ""
                return {"content": data["answer"], "source": "Tavily", "url": url}
            results = data.get("results", [])
            if results:
                txt = "\n\n".join(f"[{r.get('title','')}] {r.get('content','')}" for r in results)
                return {"content": txt, "source": "Tavily", "url": results[0].get("url","")}
        except Exception as e:
            print(f" Tavily error: {e}")
        return None
class DuckDuckGoRetriever:
    def search(self, query: str) -> Optional[Dict]:
        try:
            r = requests.get("https://api.duckduckgo.com/", params={
                "q": query, "format": "json", "no_html": 1
            }, timeout=6)
            data = r.json()
            if data.get("AbstractText"):
                return {"content": data["AbstractText"], "source": "DDG", "url": data.get("AbstractURL","")}
            if data.get("Answer"):
                return {"content": data["Answer"], "source": "DDG", "url": data.get("AnswerURL", "")}
        except:
            pass
        return None

# =============================
# POST-PROCESSING (improved)
# =============================
def extract_model_response(full_response: str) -> str:
    if "\nmodel\n" in full_response:
        resp = full_response.split("\nmodel\n")[-1].strip()
    else:
        resp = full_response.strip()
    # Remove junk
    resp = re.sub(r'\[[\w\s\/\-]+\]', '', resp)
    resp = re.sub(r'^(Answer:|Jibu:|Jibu sahihi:)\s*', '', resp, flags=re.I)
    # Better sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', resp)
    clean = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if len(s) < 4: continue
        norm = re.sub(r'\s+', ' ', s.lower().strip('.,!?'))
        if norm in seen: continue
        seen.add(norm)
        clean.append(s)
    clean = clean[:5]
    final = ' '.join(s.strip() for s in clean if s.strip()).strip()
    # Catch obvious garbage
    if re.search(r'(\d+\.){10,}', final) or len(final.split()) > 120 and '.' in final * 10:
        return "Samahani, nimepata hitilafu ya uundaji. Tafadhali jaribu tena."
    return final if final else "Samahani, sikupata jibu sahihi. Unaweza kuuliza tena?"

# =============================
# CONFIDENCE PROBE
# =============================
def should_use_rag(query: str, confidence_threshold: float = 0.68) -> Tuple[bool, str, float]:
    q_lower = query.lower()
    factual_kw = ["nani", "wapi", "lini", "idadi", "mwaka", "tarehe", "sasa", "karne", "202", "milioni"]
    is_factual = any(kw in q_lower for kw in factual_kw)
    is_short = len(query.split()) < 3
    if is_short and not is_factual:
        print(" 💬 Short casual query — skipping RAG")
        return False, "", 1.0
    formatted = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = extract_model_response(txt)
    has_placeholder = any(x in answer.lower() for x in ['???', 'unknown', '[jina', '[tarehe'])
    too_short = len(answer.split()) < 4
    top_conf = 0.95
    use_rag = is_factual or has_placeholder or too_short
    print(f" 📊 Probe confidence rough: {top_conf:.2f} | placeholder: {has_placeholder} | short: {too_short}")
    return use_rag, answer, top_conf

# =============================
# MAIN FUNCTION
# =============================

def hybrid_generate(prompt: str, context: str, query: str) -> str:
    if context.strip():
        print(" 🌐 Context available — using Groq directly...")
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Wewe ni msaidizi wa Kiswahili. Tumia TAARIFA zilizotolewa PEKEE. Jibu kwa Kiswahili. Maneno 50-200 tu."},
                    {"role": "user", "content": f"TAARIFA:\n{context[:2500]}\n\nSWALI: {query}\nJIBU:"}
                ],
                model="llama-3.3-70b-versatile",
            )
            result = response.choices[0].message.content
            if result and result.strip():
                print(" ✅ Groq successful")
                return result.strip()
        except Exception as e:
            print(f" ⚠️ Groq failed: {e}")

    # Fallback to local model if Groq fails or no context
    print(" 🔄 Falling back to local model...")
    formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    cfg = {**generation_config, "temperature": 0.15}
    with torch.no_grad():
        out = model.generate(**inputs, **cfg)
    local_response = extract_model_response(tokenizer.decode(out[0], skip_special_tokens=True))
    return local_response or "Samahani, sikupata jibu. Tafadhali jaribu tena."
def generate_with_rag(
    query: str,
    kb: WikipediaKnowledgeBase,
    tavily: Optional[TavilyRetriever] = None,
    ddg: Optional[DuckDuckGoRetriever] = None,
    show_context: bool = False,
    confidence_threshold: float = 0.68
) -> Tuple[str, List[Dict]]:
    print(f"\n🔍 Query: {query}")

    # Identity response
    IDENTITY_TRIGGERS = ["wewe ni nani", "kazi yako ni nini", "unaweza kufanya nini", "wewe ni robot", "wewe ni binadamu", "sauti ni nini", "unaitwa nani", "jina lako", "una jina", "mambo, unaitwa", "unaitwa", "uliundwa na nani", "nani alikuunda", "wewe ni nini", "unatoka wapi", "ulitoka wapi", "ulitokea wapi", "ni nani wewe"]
    query_clean = re.sub(r'[?!.,;:]', '', query.lower().strip())
    if any(t in query_clean for t in IDENTITY_TRIGGERS):
        return "Mimi ni Sauti, msaidizi wa AI aliyeundwa kufikiria kwa njia ya Kiafrika. Ninaweza kukufunza somo lolote kwa lugha ya Kiswahili, maswali ya jumla, na kubadili mazungumzo ya lugha za Kiafrika kuwa maandishi.", []

    if is_creative_or_grammar_query(query):
        cfg = {**generation_config, "temperature": 0.40}
        formatted = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **cfg)
        return extract_model_response(tokenizer.decode(out[0], skip_special_tokens=True)), []

    use_rag, initial, conf = should_use_rag(query, confidence_threshold)
    if not use_rag:
        print(f" ✅ Confident direct — still checking Tavily...")
        pass

    print(f" 🔍 Retrieving (conf was {conf:.2f})...")
    context = ""
    sources = []

    # Try Tavily first
    print(" 🌐 Trying Tavily...")
    res = tavily.search(query) if tavily else None
    if res:
        context = res['content']
        sources = [{'title': res['source'], 'url': res.get('url', ''), 'language': 'en'}]

    # Fall back to Wikipedia if Tavily returns nothing
    if not context:
        print(" 📖 Tavily empty — trying Wikipedia...")
        articles = kb.search_and_fetch(query)
        if articles:
            kb.create_vector_store(articles)
            docs = kb.retrieve_relevant_context(query, k=3)
            for doc in docs:
                lang_note = " (Kiswahili)" if doc['language'] == 'sw' else ""
                context += f"[Wikipedia{lang_note} – {doc['title']}]\n{doc['content']}\n\n"
                sources.append({'title': doc['title'], 'url': doc['url'], 'language': doc['language']})

    # Fall back to DDG if both Tavily and Wikipedia return nothing
    if not context:
        print(" 🦆 Trying DuckDuckGo...")
        res = ddg.search(query) if ddg else None
        if res:
            context = res['content']
            sources = [{'title': res['source'], 'url': res.get('url', ''), 'language': 'en'}]

    if not context:
        return "Samahani, sikupata taarifa za kutosha kujibu swali hili.", []

    if show_context:
        print(f"\n📚 Context (truncated):\n{context[:600]}...\n")

    prompt = f"""Wewe ni msaidizi wa Kiswahili sahihi na wa kuaminika.
KANUNI (LAZIMA UFuate):
1. Answer in standard Kiswahili ONLY.
2. Use ONLY information provided in the CONTEXT. DO NOT CHANGE STATISTICS or create false ones.
3. FOR STATISTICS AND QUANTITIES: Use ACTUAL numbers from the DATA without changing even one.
4. After answering, ask ONE relevant follow-up question to keep the conversation going.
5. DO NOT provide personal opinions unless the question clearly asks you to.
6. DO NOT REPEAT sentences or lists without reason. DO NOT LINK unrelated ideas.
7. Ikiwa jibu halipo kwenye TAARIFA, sema: "Samahani, maelezo yaliyotolewa hayatoshi."
8. Jibu kwa Kiswahili sanifu pekee — USITUMIE Kiingereza.
9. FOR NUMBERS AND STATISTICS: Use the EXACT numbers from the PROVIDED INFORMATION without changing even a single digit.
10. Answer with 50–120 words then ask a short follow-up question.
11. Do NOT repeat sentences or lists without reason. Do NOT combine unrelated ideas.
TAARIFA:
{context[:2000]}
SWALI: {query}
JIBU:"""

    cleaned = hybrid_generate(prompt, context, query)
    return cleaned, sources

class GeneralChatbot:
    def __init__(self):
        self.kb = WikipediaKnowledgeBase()
        self.tavily = TavilyRetriever()
        self.ddg = DuckDuckGoRetriever()
    def generate_response(self, message: str):
        return generate_with_rag(message, self.kb, self.tavily, self.ddg, show_context=True)
# =============================
# KIKUYU TRANSCRIBER
# =============================
class KikuyuTranscriber:
    def __init__(self, output_dir="/kaggle/working/kikuyu_transcriptions", model_size="300M", prefer_ctc=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        variants = [
            
            f"omniASR_LLM_{model_size}",
            f"omniASR_LLM_{model_size}_v2",
        ]
        if model_size == "300M" and not prefer_ctc:
            variants.insert(0, "omniASR_CTC_300M_v2")

        self.pipeline = None
        self.model_card = None

        print(f"\nTrying to load model ({model_size})...")
        for name in variants:
            try:
                print(f"  → {name}")
                self.pipeline = ASRInferencePipeline(model_card=name)
                self.model_card = name
                print(f"✅ Loaded: {name}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:80]}...")

        if self.pipeline is None:
            raise RuntimeError("No model could be loaded. Try smaller size or check internet/storage.")

        self.is_llm = "LLM" in self.model_card
        print(f"Model type: {'LLM (supports lang)' if self.is_llm else 'CTC (zero-shot, no lang)'}")

    def download_youtube_audio(self, url):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.audio_dir / f"audio_{timestamp}.wav"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'outtmpl': str(output_path.with_suffix('')),
            'quiet': False,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36',
            },
            'retries': 3,
            'fragment_retries': 3,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
        print(f"Downloaded: {title} ({duration}s)")
        return output_path, title, duration

    def split_audio_into_chunks(self, audio_path, max_sec=30):
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        duration_ms = len(audio)
        chunk_ms = max_sec * 1000
        chunks = []
        temp_files = []
        for i in range(0, duration_ms, chunk_ms):
            chunk = audio[i:i + chunk_ms]
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                chunk.export(tmp.name, format="wav")
                temp_files.append(tmp.name)
                chunks.append(tmp.name)
        print(f"Split into {len(chunks)} chunks")
        return chunks, temp_files

    def transcribe_chunk(self, chunk_path, lang="kik_Latn"):
        duration = len(AudioSegment.from_wav(chunk_path)) / 1000
        if duration > 40:
            return "[Chunk too long]"
        try:
            kwargs = {}
            if self.is_llm:
                kwargs["lang"] = [lang]
            trans = self.pipeline.transcribe([chunk_path], batch_size=1, **kwargs)
            return trans[0].strip() if trans else "[Empty]"
        except Exception as e:
            return f"[Error: {str(e)[:60]}]"

    def process_video(self, url, lang="kik_Latn"):
        audio_file, title, dur = self.download_youtube_audio(url)
        if not audio_file.exists():
            return None
        chunks, temps = self.split_audio_into_chunks(audio_file)
        transcriptions = []
        for chunk in tqdm(chunks, desc="Transcribing"):
            text = self.transcribe_chunk(chunk, lang)
            transcriptions.append(text)
        for t in temps:
            try: os.unlink(t)
            except: pass
        full_text = " ".join(transcriptions)
        result = {
            "title": title,
            "url": url,
            "duration": dur,
            "transcription": full_text,
            "model": self.model_card,
            "lang": lang if self.is_llm else "zero-shot (CTC)"
        }
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:40]
        out_file = self.output_dir / f"{safe_title}_{datetime.now():%Y%m%d_%H%M}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_file}")
        return result, full_text, str(out_file)

# =============================
# INITIALIZE ALL SYSTEMS
# =============================
def initialize_systems():
    knowledge_base_paths = [
        "/kaggle/input/datasets/briangreenheart/combined/combined (1).json"
    ]
    kb_path = next((p for p in knowledge_base_paths if os.path.exists(p)), None)
    if not kb_path:
        print("Warning: Knowledge base not found. Using default data.")
    lit_model_paths = [
        "/kaggle/input/gemma/swahili-gemma-finetuned",
        "/kaggle/input/swahili-gemma-finetuned",
        "swahili-gemma-finetuned"
    ]
    lit_model_path = next((p for p in lit_model_paths if os.path.exists(p)), None)
    if not lit_model_path:
        print("Warning: Literature model not found. Using knowledge base only.")
    rag_system = SwahiliLiteratureRAG(kb_path, lit_model_path)
    general_chat = GeneralChatbot()
    kikuyu_transcriber = KikuyuTranscriber(model_size="1B", prefer_ctc=False)
    return rag_system, general_chat, kikuyu_transcriber


print("Initializing all systems...")
rag_system, general_chat, kikuyu_transcriber = initialize_systems()
print("All systems initialized!")

# =============================
# LIBRARY LOADER
# =============================
LIBRARY_PATH = "/kaggle/input/datasets/briangreenheart/literary-works/LITERARY WORKS"

LIBRARY_METADATA = [
    {"id": "kasiri", "title": "Kasiri ya Mwinyi Fuad", "author": "Adam Shafi Adam", "category": "Fasihi ya Kiswahili", "filename": "Kasiri ya Mwinyi Fuad.txt"},
    {"id": "kusadikika", "title": "Kusadikika", "author": "Shaaban Robert", "category": "Fasihi ya Kiswahili", "filename": "Kusadikika.txt"},
    {"id": "utengano", "title": "Utengano", "author": "Said Ahmed Mohamed", "category": "Fasihi ya Kiswahili", "filename": "Utengano.txt"},
    {"id": "walenisi", "title": "Walenisi", "author": "Katama Mkangi", "category": "Fasihi ya Kiswahili", "filename": "Walenisi.txt"},
    {"id": "mpambano", "title": "Mpambano (The Duel)", "author": "Anton Chekhov", "category": "Tafsiri", "filename": "Mpambano.txt"},
    {"id": "kifo", "title": "Kifo cha Ivan Ilyich", "author": "Leo Tolstoy", "category": "Tafsiri", "filename": "Kifo Cha Ivann Ilyich.txt"},
    {"id": "jekyll", "title": "Kisa cha Ajabu cha Dkt. Jekyll na Mr. Hyde", "author": "R.L. Stevenson", "category": "Tafsiri", "filename": "Kisa cha Ajabu cha Dkt. jekyll na Mr. Hyde.txt"},
    {"id": "manifesto", "title": "Manifesto ya Kikomunisti", "author": "Marx & Engels", "category": "Tafsiri", "filename": "Manifesto ya Kikomunisti.txt"},
]

def load_book_text(filename: str) -> str:
    path = os.path.join(LIBRARY_PATH, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Hitilafu ya kusoma faili: {str(e)}"

def get_page(text: str, page: int, words_per_page: int = 500) -> Tuple[str, int, int]:
    words = text.split()
    total_pages = max(1, (len(words) + words_per_page - 1) // words_per_page)
    page = max(0, min(page, total_pages - 1))
    start = page * words_per_page
    end = start + words_per_page
    return " ".join(words[start:end]), page, total_pages

# =============================
# GRADIO APP
# =============================
def create_app():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

    .gradio-container { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3, .header-text h1 { font-family: 'Playfair Display', serif; font-weight: 700; line-height: 1.2; }

    .header {
        background: linear-gradient(135deg, #B8651B 0%, #C77A2E 100%);
        padding: 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .header .block { background: none !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    .logo-container { display: flex; align-items: center; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; }
    .header-text { text-align: left; }
    .header-text h1 { margin: 0; font-size: 3rem; font-weight: 700; color: #fff; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }

    .main-container {
        display: flex !important; flex-direction: row !important; justify-content: space-between !important;
        align-items: flex-start !important; flex-wrap: nowrap !important; gap: 1.5rem !important; width: 100%;
    }
    .sidebar {
        flex: 0 0 22%; max-width: 260px; min-width: 200px; background: white;
        border-radius: 16px; padding: 1.2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e9ecef;
    }
    .chat-main {
        flex: 1 1 78%; min-width: 0; background: white; border-radius: 16px;
        padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid #e9ecef;
    }
    .sidebar-title {
        font-family: 'Playfair Display', serif; font-weight: 600; font-size: 1.2rem;
        margin-bottom: 1rem; color: #2c3e50; text-align: center;
    }
    .sidebar-buttons { display: flex; flex-direction: column; gap: 0.8rem; }
    .sidebar-btn {
        width: 100% !important; text-align: left !important; justify-content: flex-start !important;
        padding: 12px 16px !important; border-radius: 12px !important; font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    .chat-container {
        background: white; border-radius: 20px; padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); height: fit-content; border: 1px solid #e9ecef;
    }
    .kiswahili-active { border-top: 6px solid #B8651B; }
    .general-active { border-top: 6px solid #1F5D4F; }
    .transcriber-active { border-top: 6px solid #6B35A8; }
    .library-active { border-top: 6px solid #C9A84C; }
    .chatbot-container { border: 1px solid #e9ecef; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }

    .toggle-btn {
        margin: 0 0.25rem; border-radius: 12px !important; padding: 10px 16px !important;
        font-size: 14px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important; border: 2px solid transparent !important;
    }
    .kiswahili-active-mode { background: linear-gradient(135deg, #B8651B 0%, #C77A2E 100%) !important; color: white !important; border-color: #9a5216 !important; }
    .general-active-mode { background: linear-gradient(135deg, #1F5D4F 0%, #2A7A68 100%) !important; color: white !important; border-color: #16453a !important; }
    .transcriber-active-mode { background: linear-gradient(135deg, #6B35A8 0%, #8B55C8 100%) !important; color: white !important; border-color: #4a2278 !important; }
    .library-active-mode { background: linear-gradient(135deg, #C9A84C 0%, #E0B96A 100%) !important; color: white !important; border-color: #a8873a !important; }

    .brown-primary-btn, .general-send-btn, .transcriber-btn {
        border-radius: 12px !important; padding: 12px 30px !important;
        font-size: 16px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important; border: 2px solid transparent !important;
    }
    .brown-primary-btn { background: linear-gradient(135deg, #B8651B 0%, #C77A2E 100%) !important; color: white !important; border-color: #9a5216 !important; }
    .general-send-btn { background: linear-gradient(135deg, #1F5D4F 0%, #2A7A68 100%) !important; color: white !important; border-color: #16453a !important; }
    .transcriber-btn { background: linear-gradient(135deg, #6B35A8 0%, #8B55C8 100%) !important; color: white !important; border-color: #4a2278 !important; }
    .library-btn { background: linear-gradient(135deg, #C9A84C 0%, #E0B96A 100%) !important; color: white !important; border-radius: 12px !important; padding: 12px 30px !important; font-size: 16px !important; font-weight: 600 !important; border: 2px solid #a8873a !important; }

    .textbox { border-radius: 12px !important; border: 2px solid #e1e5e9 !important; transition: all 0.3s ease !important; }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    .sources-display { font-size: 13px; color: #555; padding: 8px 12px; background: #f8f9fa; border-radius: 8px; margin-top: 8px; }

    .transcriber-panel {
        background: #faf8ff; border-radius: 16px; padding: 1.5rem;
        border: 1px solid #e0d5f5; margin-bottom: 1rem;
    }
    .transcriber-info {
        background: #f0ebff; border-radius: 10px; padding: 12px 16px;
        font-size: 13px; color: #5a3a8a; margin-bottom: 1rem;
        border-left: 4px solid #6B35A8;
    }
    .book-card { background: #fffdf5; border: 1px solid #e8d9a0; border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem; cursor: pointer; transition: all 0.2s ease; }
    .book-card:hover { background: #fff8e1; border-color: #C9A84C; }
    .book-category-swahili { color: #B8651B; font-weight: 600; font-size: 12px; }
    .book-category-tafsiri { color: #1F5D4F; font-weight: 600; font-size: 12px; }
    .library-info { background: #fffdf5; border-radius: 10px; padding: 12px 16px; font-size: 13px; color: #7a6a2a; margin-bottom: 1rem; border-left: 4px solid #C9A84C; }
    """

    logo_path = "/kaggle/input/datasets/briangreenheart/bestttt/Sauti logo.png"
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()

    with gr.Blocks(title="Sauti - AI Chat Assistant", theme=gr.themes.Soft(), css=custom_css) as app:

        # ── Header ──────────────────────────────────────────────
        with gr.Column(elem_classes="header"):
            gr.HTML(f"""
            <div class="logo-container">
                <div style="display:flex;align-items:center;justify-content:center;gap:2rem;">
                    <img src="data:image/png;base64,{logo_base64}"
                         style="width:90px;height:90px;object-fit:contain;border-radius:50%;
                                background:white;padding:10px;box-shadow:0 4px 20px rgba(0,0,0,0.15);"
                         alt="Sauti Logo">
                    <div class="header-text">
                        <h1>Sauti</h1>
                        <p style="font-size:1.2rem;font-weight:400;color:#fff;margin-top:0.3rem;opacity:0.9;">
                            AI that thinks the African way
                        </p>
                    </div>
                </div>
            </div>
            """)

        # ── Main layout ─────────────────────────────────────────
        with gr.Row(elem_classes="main-container"):

            # ── Sidebar ─────────────────────────────────────────
            with gr.Column(elem_classes="sidebar"):
                with gr.Column():
                    kiswahili_toggle = gr.Button(
                        "📚 Msaidizi wa Kiswahili", variant="primary", size="lg",
                        elem_classes="toggle-btn kiswahili-active-mode"
                    )
                    general_toggle = gr.Button(
                        "🌍 Msaidizi wa Jumla", variant="secondary", size="lg",
                        elem_classes="toggle-btn"
                    )
                    transcriber_toggle = gr.Button(
                        "🎙️ African Language Transcriber", variant="secondary", size="lg",
                        elem_classes="toggle-btn"
                    )
                    library_toggle = gr.Button(
                        "📖 Maktaba", variant="secondary", size="lg",
                        elem_classes="toggle-btn"
                    )

                gr.HTML("<div style='margin:1rem 0;border-top:1px solid #e1e5e9;'></div>")

                with gr.Column(visible=True) as kiswahili_sidebar:
                    gr.HTML('<div class="sidebar-title">Quick Questions</div>')
                    with gr.Column(elem_classes="sidebar-buttons"):
                        kiswahili_btn1 = gr.Button("Eleza kuhusu Nomino katika Kiswahili", elem_classes="sidebar-btn")
                        kiswahili_btn2 = gr.Button("Taja aina za Nomino na utoe mifano", elem_classes="sidebar-btn")
                        kiswahili_btn3 = gr.Button("Fafanua maana ya Vitenzi katika Kiswahili", elem_classes="sidebar-btn")
                        kiswahili_btn4 = gr.Button("Eleza aina za Vitenzi na utoe mifano", elem_classes="sidebar-btn")

                with gr.Column(visible=False) as general_sidebar:
                    gr.HTML('<div class="sidebar-title">Quick Questions</div>')
                    with gr.Column(elem_classes="sidebar-buttons"):
                        general_btn1 = gr.Button("Nisimulie hadithi kuhusu mwalimu", elem_classes="sidebar-btn")
                        general_btn2 = gr.Button("Eleza kuhusu AI kwa kifupi", elem_classes="sidebar-btn")
                        general_btn3 = gr.Button("Andika shairi kuhusu elimu", elem_classes="sidebar-btn")
                        general_btn4 = gr.Button("Umuhimu wa historia ni nini?", elem_classes="sidebar-btn")

                with gr.Column(visible=False) as transcriber_sidebar:
                    gr.HTML(f"""
                    <div class="sidebar-title">🎙️ Transcriber Info</div>
                    <div class="transcriber-info">
                        <b>Model:</b> {kikuyu_transcriber.model_card or 'Not loaded'}<br>
                        <b>Type:</b> {'LLM' if kikuyu_transcriber.is_llm else 'CTC'}<br>
                        <b>Default lang:</b> kik_Latn<br><br>
                        Paste a YouTube URL and transcribe African language audio to text.
                        Results are saved as JSON and available for download.
                    </div>
                    """)

                with gr.Column(visible=False) as library_sidebar:
                    gr.HTML(f"""
                    <div class="sidebar-title">📖 Maktaba</div>
                    <div class="library-info">
                        Kazi <b>{len(LIBRARY_METADATA)}</b> za fasihi.<br><br>
                        Chagua kitabu kwenye orodha ili kusoma.
                    </div>
                    """)

            # ── Chat / Tool panels ───────────────────────────────
            with gr.Column(elem_classes="chat-main"):

                # ── Kiswahili Literature Chat ────────────────────
                with gr.Column(visible=True, elem_classes=["chat-container", "kiswahili-active"]) as kiswahili_container:
                    kiswahili_chatbot = gr.Chatbot(
                        height=300,
                        type="messages",
                        elem_classes="chatbot-container"
                    )
                    with gr.Row():
                        kiswahili_input = gr.Textbox(
                            placeholder="Andika swali lako hapa...",
                            lines=2, scale=4, elem_classes="textbox"
                        )
                    with gr.Row():
                        kiswahili_send = gr.Button("✉️ Tuma Swali", variant="primary", size="lg", elem_classes="brown-primary-btn")
                        kiswahili_clear = gr.Button("🗑️ Futa Majadiliano", size="lg", elem_classes="brown-primary-btn")
                    confidence_display = gr.HTML(
                        value="<div style='text-align:center;padding:10px;'>Confidence will appear here after response</div>"
                    )

                # ── General Chat ─────────────────────────────────
                with gr.Column(visible=False, elem_classes=["chat-container", "general-active"]) as general_container:
                    general_chatbot = gr.Chatbot(
                        height=300,
                        type="messages",
                        elem_classes="chatbot-container"
                    )
                    with gr.Row():
                        general_input = gr.Textbox(
                            placeholder="Andika ujumbe wako hapa...",
                            lines=2, scale=4, elem_classes="textbox"
                        )
                    with gr.Row():
                        general_send = gr.Button("✉️ Tuma Ujumbe", variant="primary", size="lg", elem_classes="general-send-btn")
                        general_clear = gr.Button("🗑️ Futa Mazungumzo", size="lg", elem_classes="general-send-btn")
                    sources_display = gr.HTML(
                        value="<div style='text-align:center;padding:10px;'>Vyanzo vitaonekana hapa</div>"
                    )

                # ── African Language Transcriber ────────────────────────────
                with gr.Column(visible=False, elem_classes=["chat-container", "transcriber-active"]) as transcriber_container:
                    gr.HTML("""
                    <h3 style="font-family:'Playfair Display',serif;color:#6B35A8;margin-bottom:1rem;">
                        🎙️ African Language YouTube Transcriber
                    </h3>
                    """)
                    with gr.Column(elem_classes="transcriber-panel"):
                        youtube_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            elem_classes="textbox"
                        )
                        lang_code = gr.Textbox(
                            label="Language Code",
                            value="kik_Latn",
                            elem_classes="textbox"
                        )
                        with gr.Row():
                            transcribe_btn = gr.Button(
                                "🎙️ Transcribe Video", variant="primary", size="lg",
                                elem_classes="transcriber-btn"
                            )
                            transcribe_clear_btn = gr.Button(
                                "🗑️ Clear", size="lg", elem_classes="transcriber-btn"
                            )

                    gr.HTML("<div style='margin:1rem 0;border-top:1px solid #e0d5f5;'></div>")

                    video_title = gr.Textbox(label="Video Title", interactive=False, elem_classes="textbox")
                    video_duration = gr.Textbox(label="Duration", interactive=False, elem_classes="textbox")
                    preview_text = gr.Textbox(
                        label="Preview (first 600 chars)", lines=5,
                        interactive=False, elem_classes="textbox"
                    )
                    full_text = gr.Textbox(
                        label="Full Transcription", lines=10,
                        interactive=False, elem_classes="textbox"
                    )
                    transcription_status = gr.HTML(
                        value="<div style='text-align:center;padding:10px;color:#888;'>Ready to transcribe</div>"
                    )
                    json_download = gr.File(label="📥 Download JSON", visible=False)
                
                
                # ── Maktaba ────────────────────────────
                with gr.Column(visible=False, elem_classes=["chat-container", "library-active"]) as library_container:
                    gr.HTML("""
                    <h3 style="font-family:'Playfair Display',serif;color:#C9A84C;margin-bottom:1rem;">
                        📖 Maktaba ya Sauti
                    </h3>
                    """)
                    # Book grid
                    with gr.Column(visible=True) as book_list_panel:
                        gr.HTML("""
                        <div style='margin-bottom:1rem;'>
                            <b style='font-family:Playfair Display,serif;font-size:1.1rem;'>Chagua Kitabu</b>
                            <span style='color:#888;font-size:13px;margin-left:1rem;'>Kazi za Fasihi ya Kiswahili</span>
                        </div>
                        """)
                        gr.HTML("<div style='color:#B8651B;font-weight:700;margin-bottom:0.5rem;font-size:0.9rem;'>📚 FASIHI YA KISWAHILI</div>")
                        swahili_books = [b for b in LIBRARY_METADATA if b["category"] == "Fasihi ya Kiswahili"]
                        swahili_btns = []
                        for book in swahili_books:
                            btn = gr.Button(
                                f"{book['title']} — {book['author']}",
                                elem_classes="sidebar-btn"
                            )
                            swahili_btns.append((btn, book))
                        gr.HTML("<div style='color:#1F5D4F;font-weight:700;margin:1rem 0 0.5rem;font-size:0.9rem;'>🌍 TAFSIRI ZA KAZI ZA DUNIA</div>")
                        tafsiri_books = [b for b in LIBRARY_METADATA if b["category"] == "Tafsiri"]
                        tafsiri_btns = []
                        for book in tafsiri_books:
                            btn = gr.Button(
                                f"{book['title']} — {book['author']}",
                                elem_classes="sidebar-btn"
                            )
                            tafsiri_btns.append((btn, book))
                    gr.HTML("<div style='margin:1rem 0;border-top:1px solid #e8d9a0;'></div>")
                    with gr.Column(visible=False) as reader_panel:
                        book_title_display = gr.HTML("")
                        page_display = gr.Textbox(
                            label="", lines=18, interactive=False,
                            elem_classes="textbox"
                        )
                        page_info = gr.HTML("")
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ Kurasa Iliyopita", elem_classes="library-btn")
                            back_to_list_btn = gr.Button("📚 Orodha ya Vitabu", elem_classes="library-btn")
                            next_btn = gr.Button("Kurasa Inayofuata ➡️", elem_classes="library-btn")
                    library_status = gr.HTML(
                        value="<div style='text-align:center;padding:10px;color:#888;'>Chagua kitabu kusoma</div>"
                    )
                    current_book_text = gr.State("")
                    current_page = gr.State(0)

        # ════════════════════════════════════════════════════════
        # RESPONSE FUNCTIONS
        # ════════════════════════════════════════════════════════
        def respond_kiswahili(message, chat_history):
            if not message.strip():
                return "", chat_history, "<div style='color:#e74c3c;text-align:center;'>Please enter a question</div>"
            if chat_history is None:
                chat_history = []
            response, confidence = rag_system.answer_query(message)
            try:
                conf_val = float(confidence.split(':')[-1].strip().replace('%', '')) / 100
            except:
                conf_val = 0.0
            if conf_val > 0.7:
                cls, label = "confidence-high", "High Confidence"
            elif conf_val > 0.4:
                cls, label = "confidence-medium", "Medium Confidence"
            else:
                cls, label = "confidence-low", "Low Confidence"
            conf_html = f"<div style='text-align:center;padding:10px;background:#f8f9fa;border-radius:5px;'><span class='{cls}'>{label}</span> | {confidence}</div>"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history, conf_html

        def respond_general(message, chat_history):
            if not message.strip():
                return "", chat_history, "<div style='text-align:center;padding:10px;'>Vyanzo vitaonekana hapa</div>"
            if chat_history is None:
                chat_history = []
            response, sources = general_chat.generate_response(message)
            if not response:
                response = "Samahani, sikupata jibu. Tafadhali jaribu tena."
            print(f"\n💬 Final Answer: {response}")
            if sources:
                print("📚 Sources:")
                for s in sources:
                    lang = " (Swahili)" if s.get('language') == 'sw' else ""
                    print(f"   • {s['title']}{lang}: {s['url']}")
            print("="*60)
            if sources:
                src_html = "<div class='sources-display'><b>📚 Vyanzo:</b><br>"
                for s in sources:
                    url = s.get('url', '')
                    title = s.get('title', 'Chanzo')
                    lang_icon = " 🇸🇼" if s.get('language') == 'sw' else " 🌐"
                    if url and title:
                        src_html += f"• <b>{title}</b>{lang_icon}: <a href='{url}' target='_blank'>{url}</a><br>"
                    elif url:
                        src_html += f"• <a href='{url}' target='_blank'>{url}</a>{lang_icon}<br>"
                    else:
                        src_html += f"• {title}{lang_icon}<br>"
                src_html += "</div>"
            else:
                src_html = "<div style='text-align:center;padding:10px;color:#888;'>Hakuna vyanzo vya nje</div>"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history, src_html

        def run_transcription(url, lang):
            if not url.strip():
                return ("", "", "", "", "<div style='color:#e74c3c;text-align:center;'>Please enter a YouTube URL</div>", gr.update(visible=False))
            try:
                result, full, json_path = kikuyu_transcriber.process_video(url.strip(), lang.strip())
                preview = full[:600] + "..." if len(full) > 600 else full
                status_html = f"""
                <div style='text-align:center;padding:10px;background:#f0ebff;border-radius:8px;color:#5a3a8a;'>
                    ✅ Transcription complete | Model: <b>{result['model']}</b> | Lang: <b>{result['lang']}</b>
                </div>"""
                return (result["title"], f"{result['duration']} seconds", preview, full, status_html, gr.update(value=json_path, visible=True))
            except Exception as e:
                error_html = f"<div style='color:#e74c3c;text-align:center;padding:10px;'>❌ Error: {str(e)}</div>"
                return "", "", "", "", error_html, gr.update(visible=False)

        def clear_transcription():
            return ("", "", "", "", "<div style='text-align:center;padding:10px;color:#888;'>Ready to transcribe</div>", gr.update(visible=False))

        def open_book(filename, title, author):
            text = load_book_text(filename)
            page_text, page_num, total = get_page(text, 0)
            title_html = f"""
            <div style='margin-bottom:1rem;'>
                <h4 style='font-family:Playfair Display,serif;color:#C9A84C;margin:0;'>{title}</h4>
                <span style='color:#888;font-size:13px;'>{author}</span>
            </div>"""
            info_html = f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {page_num+1} / {total}</div>"
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                page_text, title_html, info_html, text, 0
            )

        def turn_page(text, page, direction):
            _, _, total = get_page(text, 0)
            new_page = page + direction
            new_page = max(0, min(new_page, total - 1))
            page_text, page_num, total = get_page(text, new_page)
            info_html = f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {page_num+1} / {total}</div>"
            return page_text, info_html, new_page

        def back_to_booklist():
            return gr.update(visible=True), gr.update(visible=False)
        # ════════════════════════════════════════════════════════
        # TOGGLE FUNCTIONS
        # ════════════════════════════════════════════════════════
        def show_kiswahili():
            return (
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(variant="primary", elem_classes="toggle-btn kiswahili-active-mode"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
            )

        def show_general():
            return (
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="primary", elem_classes="toggle-btn general-active-mode"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
            )

        def show_transcriber():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="primary", elem_classes="toggle-btn transcriber-active-mode"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
            )

        def show_library():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="secondary", elem_classes="toggle-btn"),
                gr.update(variant="primary", elem_classes="toggle-btn library-active-mode"),
            )
        toggle_outputs = [
            kiswahili_container, general_container, transcriber_container, library_container,
            kiswahili_sidebar, general_sidebar, transcriber_sidebar, library_sidebar,
            kiswahili_toggle, general_toggle, transcriber_toggle, library_toggle,
        ]
        kiswahili_toggle.click(show_kiswahili, outputs=toggle_outputs)
        general_toggle.click(show_general, outputs=toggle_outputs)
        transcriber_toggle.click(show_transcriber, outputs=toggle_outputs)
        library_toggle.click(show_library, outputs=toggle_outputs)

        # Wire book buttons
        reader_outputs = [book_list_panel, reader_panel, page_display, book_title_display, page_info, current_book_text, current_page]
        for btn, book in swahili_btns + tafsiri_btns:
            btn.click(
                lambda f=book["filename"], t=book["title"], a=book["author"]: open_book(f, t, a),
                outputs=reader_outputs
            )
        prev_btn.click(
            lambda text, page: turn_page(text, page, -1),
            inputs=[current_book_text, current_page],
            outputs=[page_display, page_info, current_page]
        )
        next_btn.click(
            lambda text, page: turn_page(text, page, 1),
            inputs=[current_book_text, current_page],
            outputs=[page_display, page_info, current_page]
        )
        back_to_list_btn.click(
            back_to_booklist,
            outputs=[book_list_panel, reader_panel]
        )

        # ════════════════════════════════════════════════════════
        # CHAT WIRING
        # ════════════════════════════════════════════════════════
        kiswahili_input.submit(
            respond_kiswahili,
            [kiswahili_input, kiswahili_chatbot],
            [kiswahili_input, kiswahili_chatbot, confidence_display]
        )
        kiswahili_send.click(
            respond_kiswahili,
            [kiswahili_input, kiswahili_chatbot],
            [kiswahili_input, kiswahili_chatbot, confidence_display]
        )
        kiswahili_clear.click(
            lambda: ("", [], "<div style='text-align:center;padding:10px;'>Conversation cleared</div>"),
            outputs=[kiswahili_input, kiswahili_chatbot, confidence_display]
        )

        general_input.submit(
            respond_general,
            [general_input, general_chatbot],
            [general_input, general_chatbot, sources_display]
        )
        general_send.click(
            respond_general,
            [general_input, general_chatbot],
            [general_input, general_chatbot, sources_display]
        )
        general_clear.click(
            lambda: ("", [], "<div style='text-align:center;padding:10px;'>Mazungumzo yamefutwa</div>"),
            outputs=[general_input, general_chatbot, sources_display]
        )

        # ════════════════════════════════════════════════════════
        # TRANSCRIBER WIRING
        # ════════════════════════════════════════════════════════
        transcribe_outputs = [video_title, video_duration, preview_text, full_text, transcription_status, json_download]
        transcribe_btn.click(run_transcription, [youtube_url, lang_code], transcribe_outputs)
        transcribe_clear_btn.click(clear_transcription, outputs=transcribe_outputs)

        # ════════════════════════════════════════════════════════
        # QUICK ACTION BUTTONS
        # ════════════════════════════════════════════════════════
        def kiswahili_quick1(y):
            return respond_kiswahili("Eleza kuhusu Nomino katika Kiswahili.", y)
        def kiswahili_quick2(y):
            return respond_kiswahili("Taja aina za Nomino na utoe mifano.", y)
        def kiswahili_quick3(y):
            return respond_kiswahili("Fafanua maana ya Vitenzi katika Kiswahili.", y)
        def kiswahili_quick4(y):
            return respond_kiswahili("Eleza aina za Vitenzi na utoe mifano.", y)

        def general_quick1(y):
            return respond_general("Nisimulie hadithi kuhusu mwalimu", y)
        def general_quick2(y):
            return respond_general("Eleza kuhusu AI kwa kifupi", y)
        def general_quick3(y):
            return respond_general("Andika shairi kuhusu elimu", y)
        def general_quick4(y):
            return respond_general("Umuhimu wa historia ni nini?", y)

        kiswahili_btn1.click(kiswahili_quick1, [kiswahili_chatbot], [kiswahili_input, kiswahili_chatbot, confidence_display])
        kiswahili_btn2.click(kiswahili_quick2, [kiswahili_chatbot], [kiswahili_input, kiswahili_chatbot, confidence_display])
        kiswahili_btn3.click(kiswahili_quick3, [kiswahili_chatbot], [kiswahili_input, kiswahili_chatbot, confidence_display])
        kiswahili_btn4.click(kiswahili_quick4, [kiswahili_chatbot], [kiswahili_input, kiswahili_chatbot, confidence_display])

        general_btn1.click(general_quick1, [general_chatbot], [general_input, general_chatbot, sources_display])
        general_btn2.click(general_quick2, [general_chatbot], [general_input, general_chatbot, sources_display])
        general_btn3.click(general_quick3, [general_chatbot], [general_input, general_chatbot, sources_display])
        general_btn4.click(general_quick4, [general_chatbot], [general_input, general_chatbot, sources_display])

    return app


if __name__ == "__main__":
    os.environ['TORCH_LOGS'] = ''
    os.environ['TORCHDYNAMO_VERBOSE'] = '0'

    try:
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # =============================
    # BACKEND TEST RUNNER
    # =============================
    def run_backend_tests():
        test_queries = [
            "Gavana wa Mombasa wa sasa ni nani?",
            "Idadi ya watu nchini Kenya inakadiriwa kuwa kiasi gani kufikia katikati ya mwaka 2025?",
            "Unadhani ni mji upi unaovutia zaidi kati ya Mombasa na Dar es Salaam kwa mtalii?",
            "hesabu ni somo nzuri?",
            "andika shairi fupi kuhusu bahari",
            "Eleza kuhusu AI kwa kifupi",
            "Nieleze kuhusu mkutano wa Nairobi AI Forum wa Februari 2026.",
        ]

        print("\n" + "="*70)
        print("🚀 BACKEND TEST RUN — SWAHILI-GEMMA RAG SYSTEM")
        print("="*70)

        kb_test = WikipediaKnowledgeBase()
        tavily_test = TavilyRetriever()
        ddg_test = DuckDuckGoRetriever()

        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-"*60)
            try:
                response, sources = generate_with_rag(
                    query, kb_test, tavily_test, ddg_test,
                    show_context=True
                )
                print(f"\n💬 Final Answer: {response}")
                if sources:
                    print("\n📚 Sources:")
                    for s in sources:
                        lang = " (Swahili)" if s.get('language') == 'sw' else ""
                        print(f"   • {s['title']}{lang}: {s['url']}")
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
            print("-"*60)

        print("\n" + "="*70)
        print("✅ Backend tests complete")
        print("="*70)

    run_backend_tests()

    print("Creating Gradio app...")
    app = create_app()
    print("Launching...")
    try:
        app.launch(share=True, server_name="0.0.0.0", server_port=None, debug=True)
    except Exception as e:
        print(f"Error: {e}")
        app.launch()