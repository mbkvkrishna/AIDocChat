import os
import traceback
from pathlib import Path

from google import genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from processor import create_session, retrieve_chunks, delete_session

ACCEPTED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "md", "csv", "rtf"}
MAX_HISTORY_TURNS = 10

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=api_key, http_options={"api_version": "v1"})


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    history: list[ChatMessage]
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        session_id = create_session(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to process file.")
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    return {"deleted": session_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks = retrieve_chunks(req.session_id, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve context.")

    context_block = "\n\n---\n\n".join(chunks)

    trimmed_history = req.history[-(MAX_HISTORY_TURNS * 2):]
    history_block = "".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}\n"
        for m in trimmed_history
    )

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the document excerpts below. "
        "If the answer is not in the excerpts, say so clearly.\n\n"
        f"DOCUMENT EXCERPTS:\n{context_block}\n\n"
        f"CONVERSATION SO FAR:\n{history_block}\n"
        f"User: {req.question}\nAssistant:"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = response.text.strip()
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        msg = str(e)
        if "API_KEY" in msg.upper() or "PERMISSION" in msg.upper():
            raise HTTPException(status_code=500, detail="Invalid or missing Gemini API key.")
        raise HTTPException(status_code=502, detail=f"Gemini API error: {msg}")

    return {"answer": answer}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
