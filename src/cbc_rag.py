%%writefile cbc_rag.py

"""
CBC Curriculum — SAUTI Module
=======================================
Generic module for ALL CBC subjects.
Fetches live content from Opiq, provides:
  1. READER MODE  — full chapter in English + Swahili (paginated)
  2. RAG MODE     — student asks a question, Groq answers in Swahili

To add a new subject: just add a URL to CBC_REGISTRY. Nothing else changes.
"""

import requests
from bs4 import BeautifulSoup, NavigableString
from groq import Groq
import re
from typing import Tuple, Optional, Dict
import os


# ================================================================
# CBC SUBJECT REGISTRY
# ================================================================
CBC_REGISTRY: Dict[str, dict] = {
    "kilimo_f1": {
        "url": "https://opiq.co.ke/kit/78/chapter/3980",
        "subject": "Kilimo",
        "form": "Kidato cha Kwanza",
        "emoji": "🌱",
        "keywords": [
            "kilimo", "agriculture", "mazao", "mifugo", "shamba",
            "zao", "mimea", "mkulima", "wakulima", "genetics",
            "pathology", "entomology", "ecology", "pedology",
            "ufugaji", "upandaji", "uvunaji", "ardhi"
        ],
        "quick_questions": [
            "Kilimo ni nini?",
            "Taja matawi makuu ya kilimo",
            "Eleza mifumo ya kilimo Kenya",
            "Umuhimu wa kilimo katika uchumi ni nini?"
        ]
    },
    "biolojia_f1": {
        "url": "https://opiq.co.ke/kit/36/chapter/1579",
        "subject": "Biolojia",
        "form": "Kidato cha Kwanza",
        "emoji": "🔬",
        "keywords": [
            "biolojia", "biology", "seli", "mimea", "wanyama",
            "zoology", "botany", "genetics", "ecology", "anatomy",
            "microbiology", "taxonomy", "viumbe", "mazingira"
        ],
        "quick_questions": [
            "Biolojia ni nini?",
            "Taja matawi makuu ya biolojia",
            "Eleza umuhimu wa kusoma biolojia",
            "Tofauti kati ya mimea na wanyama ni nini?"
        ]
    },
    "kemia_f1": {
        "url": "https://opiq.co.ke/kit/37/chapter/1599",
        "subject": "Kemia",
        "form": "Kidato cha Kwanza",
        "emoji": "⚗️",
        "keywords": [
            "kemia", "chemistry", "kemikali", "atomi", "molekuli",
            "asidi", "besi", "mchanganyiko", "oksijeni", "hidrojeni",
            "maabara", "vipengele", "misombo", "gesi", "maji"
        ],
        "quick_questions": [
            "Kemia ni nini?",
            "Kemia inatumika vipi katika jamii?",
            "Maabara ya kemia ni nini?",
            "Eleza umuhimu wa kusoma kemia"
        ]
    },
}


# ================================================================
# OPIQ NOISE FILTER
# ================================================================
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
    # ── UI chrome fixes ──
    "opiq current location:",
    "current location:",
    "agriculture f1",
    "biology sec 1",
    "chemistry sec 1",
    "introduction to agriculture – opiq",
    "introduction to biology – opiq",
    "introduction to chemistry – opiq",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TRANSLATOR_PROMPT = """Wewe ni mtafsiri mahiri wa Kiswahili sanifu.
Tafsiri maandishi yafuatayo kutoka Kiingereza hadi Kiswahili sanifu.
KANUNI:
1. Tafsiri kila kitu — vichwa, maelezo, mifano, orodha.
2. Tumia istilahi za kitaaluma za Kiswahili ambapo zinapatikana.
3. Hifadhi muundo wa asili — vichwa, nambari, orodha.
4. Jibu kwa Kiswahili tu."""

TEACHER_PROMPT = """Wewe ni mwalimu msaidizi wa SAUTI anayefundisha wanafunzi wa Kenya kwa Kiswahili sanifu.

KANUNI:
1. Jibu KWA KISWAHILI PEKEE.
2. Tumia TAARIFA za sura iliyotolewa PEKEE — usibuni taarifa.
3. Eleza kwa lugha rahisi inayofaa mwanafunzi.
4. Baada ya kujibu, uliza swali moja fupi la kufuatilia.
5. Maneno 80-180 pekee.
6. Jibu moja kwa moja kama mwalimu."""


# ================================================================
# FETCHER
# ================================================================
# ================================================================
# FETCHER
# ================================================================
def extract_tables(soup):
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        table_lines = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            line = " | ".join(cols)
            if line.replace("|", "").replace(" ", ""):
                table_lines.append(line)
        full_text = " ".join(table_lines).lower()
        if any(noise in full_text for noise in ["opiq score", "tasks", "chapter is studied", "remove"]):
            table.decompose()
            continue
        if table_lines:
            block = "||NEWROW||".join(table_lines)
            table.replace_with(NavigableString(block))
        else:
            table.decompose()


def fetch_opiq_chapter(url: str, max_chars: int = 12000) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        extract_tables(soup)
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "button", "form", "iframe", "noscript"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)
        raw_text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", raw_text)  # ← before replace
        raw_text = raw_text.replace("||NEWROW||", "\n")              # ← then replace
        raw_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text)
        raw_text = raw_text.replace("–", " – ")
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
        print(f"✅ Fetched {len(cleaned)} chars from Opiq ({url})")
        return cleaned[:max_chars]
    except Exception as e:
        print(f"⚠️ Opiq fetch error: {e}")
        return None
# ================================================================
# GENERIC CBC CHAPTER CLASS
# ================================================================
class CBCChapter:

    WORDS_PER_PAGE = 150

    def __init__(self, groq_api_key: str, subject_key: str,
                 groq_model: str = "llama-3.3-70b-versatile"):
        if subject_key not in CBC_REGISTRY:
            raise ValueError(f"Subject '{subject_key}' not found in CBC_REGISTRY. "
                             f"Available: {list(CBC_REGISTRY.keys())}")
        self.config = CBC_REGISTRY[subject_key]
        self.subject_key = subject_key
        self.groq = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self._english_cache: str = ""
        self._swahili_cache: str = ""

    def get_english_chapter(self, force_refresh: bool = False) -> str:
        if self._english_cache and not force_refresh:
            return self._english_cache
        content = fetch_opiq_chapter(self.config["url"])
        self._english_cache = content or "Samahani, sura haikuweza kupakiwa. Tafadhali angalia muunganisho wa intaneti."
        return self._english_cache

    def get_swahili_chapter(self, verbose: bool = False) -> str:
        if self._swahili_cache:
            return self._swahili_cache
        english = self.get_english_chapter()
        if verbose:
            print(f"🌍 Translating {self.config['subject']} to Swahili via Groq...")
        try:
            response = self.groq.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": TRANSLATOR_PROMPT},
                    {"role": "user", "content": english}
                ],
                max_tokens=3000,
                temperature=0.2,
            )
            self._swahili_cache = response.choices[0].message.content.strip()
            if verbose:
                print("✅ Translation complete and cached")
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            self._swahili_cache = "Samahani, tafsiri haikuwezekana. Tafadhali jaribu tena."
        return self._swahili_cache

    # ────────────────────────────────────────
    # READER MODE — PAGINATION
    # Preserves line breaks so headings, paragraphs and lists
    # display as structured text, not a wall of words.
    # ────────────────────────────────────────
    def get_page(self, text: str, page: int) -> Tuple[str, int, int]:
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_words = 0

        for line in lines:
            line_words = len(line.split())
            if current_words + line_words > self.WORDS_PER_PAGE and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_words = line_words
            else:
                current_chunk.append(line)
                current_words += line_words

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        total_pages = max(1, len(chunks))
        page = max(0, min(page, total_pages - 1))
        return chunks[page], page, total_pages

    # ────────────────────────────────────────
    # RAG MODE — Q&A
    # ────────────────────────────────────────
    def answer(self, query: str, verbose: bool = False) -> dict:
        result = {"response": "", "error": None}
        english_content = self.get_english_chapter()
        user_message = (
            f"TAARIFA ZA SURA — {self.config['subject']}, "
            f"{self.config['form']} (Chanzo: opiq.co.ke):\n\n"
            f"{english_content}\n\n"
            f"SWALI LA MWANAFUNZI: {query}\n"
            f"JIBU (kwa Kiswahili):"
        )
        if verbose:
            print(f"\n🎓 [{self.config['subject']}] Swali: {query}")
        try:
            response = self.groq.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": TEACHER_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.3,
            )
            result["response"] = response.choices[0].message.content.strip()
            if verbose:
                print(f"💬 Jibu:\n{result['response']}")
        except Exception as e:
            print(f"⚠️ Groq error: {e}")
            result["error"] = str(e)
            result["response"] = "Samahani, nimepata hitilafu. Tafadhali jaribu tena."
        return result


# ================================================================
# SUBJECT MATCHER
# ================================================================
def match_subject(query: str) -> Optional[str]:
    query_lower = query.lower()
    best_key = None
    best_score = 0
    for key, config in CBC_REGISTRY.items():
        score = sum(1 for kw in config["keywords"] if kw in query_lower)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key if best_score > 0 else None


# ================================================================
# CBC MANAGER
# ================================================================
class CBCManager:

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self._chapters: Dict[str, CBCChapter] = {}

    def get(self, subject_key: str) -> CBCChapter:
        if subject_key not in self._chapters:
            self._chapters[subject_key] = CBCChapter(
                groq_api_key=self.groq_api_key,
                subject_key=subject_key
            )
        return self._chapters[subject_key]

    def answer_query(self, query: str, verbose: bool = False) -> dict:
        subject_key = match_subject(query)
        if not subject_key:
            return {"response": None, "subject_key": None, "matched": False}
        chapter = self.get(subject_key)
        result = chapter.answer(query, verbose=verbose)
        result["subject_key"] = subject_key
        result["matched"] = True
        return result

    @property
    def available_subjects(self):
        return [
            f"{config['emoji']} {config['subject']} — {config['form']}"
            for config in CBC_REGISTRY.values()
        ]


# ================================================================
# GRADIO TAB BUILDER
# ================================================================
def build_cbc_subject_tab(chapter: CBCChapter):
    import gradio as gr
    config = chapter.config

    with gr.Column():
        gr.HTML(f"""
        <div style='background:linear-gradient(135deg,#2d6a4f,#40916c);
                    padding:1.2rem 1.5rem;border-radius:12px;color:white;margin-bottom:1rem;'>
            <h4 style='margin:0;font-family:Playfair Display,serif;'>
                {config['emoji']} {config['subject']} — {config['form']}
            </h4>
            <p style='margin:0.2rem 0 0;opacity:0.85;font-size:0.85rem;'>
                <a href='{config['url']}' target='_blank' style='color:#a8e6cf;'>
                    Chanzo: opiq.co.ke ↗
                </a>
            </p>
        </div>
        """)

        with gr.Tabs():

            # ── READER TAB ──────────────────────────────────────
            with gr.Tab("📖 Soma Sura / Read Chapter"):

                lang_toggle = gr.Radio(
                    choices=["🇬🇧 English", "🇰🇪 Kiswahili"],
                    value="🇬🇧 English",
                    label="Chagua Lugha / Choose Language",
                    interactive=True
                )
                load_status = gr.HTML(
                    value="<div style='text-align:center;color:#888;font-size:13px;'>"
                          "⏳ Bonyeza 'Pakia Sura' kupakua kutoka Opiq...</div>"
                )
                page_display = gr.Textbox(
                    label="", lines=14, interactive=False, value=""
                )
                page_info = gr.HTML(
                    value="<div style='text-align:center;color:#888;font-size:13px;'>—</div>"
                )
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Iliyopita", size="sm")
                    load_btn = gr.Button("📥 Pakia Sura", size="sm")
                    next_btn = gr.Button("Inayofuata ➡️", size="sm")
                    refresh_btn = gr.Button("🔄 Sasisha", size="sm")

                page_state = gr.State(0)
                text_state = gr.State("")
                reader_outputs = [page_display, page_info, load_status, page_state, text_state]

                def load_chapter():
                    text = chapter.get_english_chapter()
                    pt, pg, total = chapter.get_page(text, 0)
                    return pt, f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {pg+1}/{total}</div>", \
                           "<div style='text-align:center;color:#2d6a4f;font-size:13px;'>✅ Imepakiwa kutoka opiq.co.ke</div>", 0, text

                def change_lang(lang):
                    text = chapter.get_swahili_chapter(verbose=True) if "Kiswahili" in lang else chapter.get_english_chapter()
                    label = "Kiswahili" if "Kiswahili" in lang else "English"
                    pt, pg, total = chapter.get_page(text, 0)
                    return pt, f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {pg+1}/{total} ({label})</div>", \
                           "<div style='text-align:center;color:#2d6a4f;font-size:13px;'>✅ Tayari</div>", 0, text

                def turn(text, page, direction):
                    if not text:
                        return "", "<div style='text-align:center;color:#e74c3c;font-size:13px;'>Pakia sura kwanza</div>", page
                    pt, pg, total = chapter.get_page(text, page + direction)
                    return pt, f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {pg+1}/{total}</div>", pg

                def refresh():
                    text = chapter.get_english_chapter(force_refresh=True)
                    pt, pg, total = chapter.get_page(text, 0)
                    return pt, f"<div style='text-align:center;color:#888;font-size:13px;'>Ukurasa {pg+1}/{total}</div>", \
                           "<div style='text-align:center;color:#2d6a4f;font-size:13px;'>✅ Imesasishwa</div>", 0, text

                load_btn.click(load_chapter, outputs=reader_outputs)
                refresh_btn.click(refresh, outputs=reader_outputs)
                lang_toggle.change(change_lang, inputs=[lang_toggle], outputs=reader_outputs)
                prev_btn.click(lambda t, p: turn(t, p, -1), inputs=[text_state, page_state], outputs=[page_display, page_info, page_state])
                next_btn.click(lambda t, p: turn(t, p, 1), inputs=[text_state, page_state], outputs=[page_display, page_info, page_state])

            # ── Q&A TAB ─────────────────────────────────────────
            with gr.Tab("🎓 Uliza Swali / Ask a Question"):

                chatbot = gr.Chatbot(height=300, type="messages", label=f"Mwalimu wa {config['subject']}")
                with gr.Row():
                    inp = gr.Textbox(placeholder=f"Uliza swali kuhusu {config['subject']}...", lines=2, scale=4)
                with gr.Row():
                    send_btn = gr.Button("✉️ Tuma Swali", variant="primary")
                    clear_btn = gr.Button("🗑️ Futa")

                gr.HTML("<div style='margin-top:0.8rem;font-size:13px;color:#555;'>⚡ Maswali ya haraka:</div>")
                with gr.Row():
                    quick_btns = [gr.Button(q, size="sm") for q in config["quick_questions"]]

                cbc_status = gr.HTML(
                    value="<div style='text-align:center;padding:8px;color:#888;font-size:13px;'>Jibu litaonekana hapa</div>"
                )

                def ask(query, history):
                    if not query.strip():
                        return "", history, ""
                    if history is None:
                        history = []
                    result = chapter.answer(query, verbose=True)
                    history.append({"role": "user", "content": query})
                    history.append({"role": "assistant", "content": result["response"]})
                    status = f"<div style='text-align:center;color:#2d6a4f;font-size:13px;'>✅ Chanzo: opiq.co.ke — {config['subject']} {config['form']}</div>"
                    return "", history, status

                inp.submit(ask, [inp, chatbot], [inp, chatbot, cbc_status])
                send_btn.click(ask, [inp, chatbot], [inp, chatbot, cbc_status])
                clear_btn.click(lambda: ("", [], ""), outputs=[inp, chatbot, cbc_status])

                for btn, q in zip(quick_btns, config["quick_questions"]):
                    btn.click(lambda h, query=q: ask(query, h), inputs=[chatbot], outputs=[inp, chatbot, cbc_status])


# ================================================================
# QUICK TEST
# ================================================================
if __name__ == "__main__":
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    api_key = secrets.get_secret("GROQ_API_KEY")

    manager = CBCManager(groq_api_key=api_key)

    print("\n" + "="*65)
    print("📚 AVAILABLE SUBJECTS:")
    for s in manager.available_subjects:
        print(f"  {s}")

    print("\n" + "="*65)
    print("🌐 FETCHING KILIMO F1 FROM OPIQ...")
    print("="*65)
    chapter = manager.get("kilimo_f1")
    english = chapter.get_english_chapter()
    print(f"\n📄 English (first 500 chars):\n{english[:500]}")

    print("\n" + "="*65)
    print("🌍 SWAHILI TRANSLATION")
    print("="*65)
    swahili = chapter.get_swahili_chapter(verbose=True)
    print(f"\n📄 Kiswahili (first 500 chars):\n{swahili[:500]}")

    print("\n" + "="*65)
    print("🎓 RAG Q&A TEST")
    print("="*65)
    for q in ["Kilimo ni nini?", "Taja matawi makuu ya kilimo",
              "Genetics katika kilimo inamaanisha nini?"]:
        print(f"\n📝 Swali: {q}")
        result = chapter.answer(q, verbose=True)
        print(f"✅ Jibu:\n{result['response']}")

    print("\n" + "="*65)
    print("🔍 AUTO-MATCH TEST")
    print("="*65)
    for q in ["Kilimo ni nini?", "Hesabu ni somo gani?", "Biolojia inahusiana na nini?"]:
        matched = manager.answer_query(q)
        print(f"Query: '{q}' → Matched: {matched.get('subject_key', 'None')}")
