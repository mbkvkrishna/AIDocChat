import os
import uuid
import fitz
import chromadb
from google import genai
from google.genai import types

_chroma = chromadb.Client()

CHUNK_SIZE = 2400
OVERLAP = 400
EMBED_BATCH_SIZE = 90
MAX_FILE_SIZE = 20 * 1024 * 1024


def _get_client() -> genai.Client:
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"api_version": "v1"},
    )


def extract_text(file_bytes: bytes, filename: str) -> str:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds the 20 MB limit.")

    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc).strip()

    elif ext in ("txt", "md"):
        text = file_bytes.decode("utf-8", errors="replace").strip()

    elif ext == "csv":
        text = file_bytes.decode("utf-8", errors="replace").strip()

    elif ext in ("doc", "docx"):
        import mammoth
        import io
        result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
        text = result.value.strip()

    elif ext == "rtf":
        from striprtf.striprtf import rtf_to_text
        raw = file_bytes.decode("utf-8", errors="replace")
        text = rtf_to_text(raw).strip()

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not text:
        raise ValueError("File contains no extractable text.")
    return text


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_embeddings.extend(e.values for e in result.embeddings)
    return all_embeddings


def create_session(file_bytes: bytes, filename: str) -> str:
    text = extract_text(file_bytes, filename)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks, "RETRIEVAL_DOCUMENT")
    session_id = str(uuid.uuid4())
    collection = _chroma.create_collection(name=session_id)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return session_id


def retrieve_chunks(session_id: str, question: str, top_k: int = 4) -> list[str]:
    try:
        collection = _chroma.get_collection(name=session_id)
    except Exception:
        raise ValueError(f"Session '{session_id}' not found. Please re-upload your document.")
    embeddings = embed_texts([question], "RETRIEVAL_QUERY")
    results = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=min(top_k, collection.count()),
    )
    return results["documents"][0]


def delete_session(session_id: str) -> None:
    try:
        _chroma.delete_collection(name=session_id)
    except Exception:
        pass
