# api/main.py - API endpoints for the DocuChat application
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.rag_engine import DocuChatEngine

app = FastAPI(title="DocuChat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG engine
engine = DocuChatEngine()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DocuChat API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file temporarily
        upload_dir = Path("uploaded_docs")
        upload_dir.mkdir(exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the file
        result = engine.process_uploaded_file(file)

        # Clean up temporary file
        file_path.unlink(missing_ok=True)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_documents(query: Dict[str, str]):
    """Query the documents"""
    try:
        user_question = query.get("question", "")
        if not user_question:
            raise HTTPException(status_code=400, detail="Question is required")

        result = engine.query_documents(user_question)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = engine.get_session_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        info = engine.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
