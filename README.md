# AIDocChat — AI Document Chat

Upload a document and ask questions about it, answered by an LLM grounded in the document's content.

## Stack

- **Backend:** Python, FastAPI
- **PDF extraction:** PyMuPDF (fitz)
- **DOC/DOCX extraction:** mammoth
- **RTF extraction:** striprtf
- **Embeddings:** Google Gemini (`gemini-embedding-001`)
- **Vector store:** ChromaDB, in-memory
- **LLM:** Google Gemini (`gemini-2.5-flash`)
- **Frontend:** Single static HTML file, vanilla JS, served by FastAPI
- **Deployment:** Render free tier

## Supported Formats

PDF, TXT, DOC, DOCX, MD, CSV, RTF (max 20 MB)

## Project Structure

```
AIDocChat/
├── main.py
├── processor.py
├── requirements.txt
├── render.yaml
└── static/
    └── index.html
```

## Local Setup

```bash
git clone https://github.com/mbkvkrishna/AIDocChat.git
cd AIDocChat
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="AIza..."
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Deploying to Render

1. Push to GitHub and connect the repository on [render.com](https://render.com).
2. Add `GEMINI_API_KEY` as an environment variable in the Render dashboard.
3. Render auto-detects `render.yaml` — no manual config needed.

## How It Works

**Upload:** Text is extracted from the document, split into overlapping chunks (~600 tokens), embedded with Gemini in batches, and stored in an in-memory ChromaDB collection keyed by a UUID session ID.

**Chat:** The question is embedded, the top 4 matching chunks are retrieved, and a grounded prompt is sent to Gemini 2.5 Flash which returns the answer.

## Limitations

- Sessions live in RAM — a restart clears all uploaded documents.
- Render free tier spins down after ~15 min of inactivity (30–60 s cold start).
- No authentication — anyone with the URL can use the app.

---
