from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List

app = FastAPI(title="Maktaba Service")

# ── Config ──────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "./data")
LIBRARY_PATH = os.path.join(DATA_PATH, "LITERARY WORKS")
WORDS_PER_PAGE = 500

# ── Debug endpoint ──────────────────────────────────────
@app.get("/debug")
def debug():
    return {
        "cwd": os.getcwd(),
        "data_path": DATA_PATH,
        "library_path": LIBRARY_PATH,
        "app_exists": os.path.exists("/app"),
        "data_exists": os.path.exists("/app/data"),
        "library_exists": os.path.exists(LIBRARY_PATH),
        "app_contents": (
            os.listdir("/app")
            if os.path.exists("/app")
            else "missing"
        ),
        "data_contents": (
            os.listdir("/app/data")
            if os.path.exists("/app/data")
            else "missing"
        ),
        "library_contents": (
            os.listdir(LIBRARY_PATH)
            if os.path.exists(LIBRARY_PATH)
            else "missing"
        ),
    }

# ── Book catalogue ───────────────────────────────────────
LIBRARY_METADATA = [
    {
        "id": "kasiri",
        "title": "Kasiri ya Mwinyi Fuad",
        "author": "Adam Shafi Adam",
        "category": "Fasihi ya Kiswahili",
        "filename": "Kasiri ya Mwinyi Fuad.txt"
    },
    {
        "id": "kusadikika",
        "title": "Kusadikika",
        "author": "Shaaban Robert",
        "category": "Fasihi ya Kiswahili",
        "filename": "Kusadikika.txt"
    },
    {
        "id": "utengano",
        "title": "Utengano",
        "author": "Said Ahmed Mohamed",
        "category": "Fasihi ya Kiswahili",
        "filename": "Utengano.txt"
    },
    {
        "id": "walenisi",
        "title": "Walenisi",
        "author": "Katama Mkangi",
        "category": "Fasihi ya Kiswahili",
        "filename": "Walenisi.txt"
    },
    {
        "id": "mpambano",
        "title": "Mpambano (The Duel)",
        "author": "Anton Chekhov",
        "category": "Tafsiri",
        "filename": "Mpambano.txt"
    },
    {
        "id": "kifo",
        "title": "Kifo cha Ivan Ilyich",
        "author": "Leo Tolstoy",
        "category": "Tafsiri",
        "filename": "Kifo Cha Ivann Ilyich.txt"
    },
    {
        "id": "jekyll",
        "title": "Kisa cha Ajabu cha Dkt. Jekyll na Mr. Hyde",
        "author": "R.L. Stevenson",
        "category": "Tafsiri",
        "filename": "Kisa cha Ajabu cha Dkt. jekyll na Mr. Hyde.txt"
    },
    {
        "id": "manifesto",
        "title": "Manifesto ya Kikomunisti",
        "author": "Marx & Engels",
        "category": "Tafsiri",
        "filename": "Manifesto ya Kikomunisti.txt"
    },
]

# ── Request / Response models ─────────────────────────────
class PageRequest(BaseModel):
    book_id: str
    page: int = 0

class PageResponse(BaseModel):
    title: str
    author: str
    category: str
    page_text: str
    page_num: int
    total_pages: int

class Book(BaseModel):
    id: str
    title: str
    author: str
    category: str

class BookListResponse(BaseModel):
    books: List[Book]
    total: int

# ── Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "maktaba",
        "books_available": len(LIBRARY_METADATA)
    }

@app.get("/books", response_model=BookListResponse)
def list_books():
    books = [
        Book(
            id=b["id"],
            title=b["title"],
            author=b["author"],
            category=b["category"]
        )
        for b in LIBRARY_METADATA
    ]
    return BookListResponse(books=books, total=len(books))

@app.post("/page", response_model=PageResponse)
def get_page(request: PageRequest):
    book = next(
        (b for b in LIBRARY_METADATA if b["id"] == request.book_id),
        None
    )

    if not book:
        raise HTTPException(
            status_code=404,
            detail=f"Book '{request.book_id}' not found"
        )

    path = os.path.join(LIBRARY_PATH, book["filename"])

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Book file not found on server: {book['filename']}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read book: {str(e)}"
        )

    words = text.split()

    total_pages = max(
        1,
        (len(words) + WORDS_PER_PAGE - 1) // WORDS_PER_PAGE
    )

    page = max(0, min(request.page, total_pages - 1))
    start = page * WORDS_PER_PAGE

    page_text = " ".join(
        words[start:start + WORDS_PER_PAGE]
    )

    return PageResponse(
        title=book["title"],
        author=book["author"],
        category=book["category"],
        page_text=page_text,
        page_num=page,
        total_pages=total_pages
    )