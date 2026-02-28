import os
import io
import uuid
import requests
import csv
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from google import genai

# Dependencies for RAG & Extractions
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document

# Load Environment Variables
load_dotenv()

app = FastAPI(title="RAG Chatbot API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount frontend files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Configuration & Clients
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Chatbot will fail to generate answers.")

# We initialize clients dynamically if values aren't present so startup doesn't crash on invalid credentials, 
# but log warning.
collection = None
try:
    if CHROMA_TENANT and CHROMA_DATABASE and CHROMA_API_KEY:
        chroma_client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY
        )
        # Use a new collection name to avoid the "soft deleted" error in Chroma Cloud
        collection = chroma_client.get_or_create_collection(
            name="rag_collection_v2"
        )
    else:
        print("WARNING: ChromaDB environment variables missing. Vector database not initialized.")
except Exception as e:
    print(f"Failed to initialize Chroma Cloud Client: {e}")

# Load the embeddings model
print("Loading sentence-transformer all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Setting up Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# In-memory store of processed sources
added_sources = []

class ScrapeRequest(BaseModel):
    url: str

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/api/sources")
def get_sources():
    return {"sources": added_sources}

@app.delete("/api/sources")
def clear_all_sources():
    global added_sources
    if collection:
        try:
            # Re-create the collection or delete everything
            # The exact method depends on chromadb version, but typically we can delete by where condition or re-create
            for source in added_sources:
                collection.delete(where={"source": source["name"]})
        except Exception as e:
            print(f"Failed to clear ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear vector db: {e}")
    
    added_sources = []
    return {"status": "success", "message": "Successfully cleared all sources"}

@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    global added_sources
    source_to_delete = next((s for s in added_sources if s["id"] == source_id), None)
    
    if not source_to_delete:
        raise HTTPException(status_code=404, detail="Source not found.")
        
    source_name = source_to_delete["name"]
    
    if collection:
        try:
            # Delete from ChromaDB based on source metadata
            collection.delete(where={"source": source_name})
        except Exception as e:
            print(f"Failed to delete from ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete from vector db: {e}")
            
    # Remove from in-memory list
    added_sources = [s for s in added_sources if s["id"] != source_id]
    
    return {"status": "success", "message": f"Successfully deleted {source_name}"}

def add_texts_to_chroma(text: str, source_name: str, source_type: str):
    """"Helper function to chunk texts and add to ChromaDB."""
    if not collection:
        raise HTTPException(status_code=500, detail="Chroma DB collection is not initialized.")
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found.")

    splits = text_splitter.split_text(text)
    if not splits:
        raise HTTPException(status_code=400, detail="Text couldn't be chunked.")

    embeddings = embedding_model.encode(splits).tolist()
    ids = [str(uuid.uuid4()) for _ in splits]
    metadatas = [{"source": source_name, "type": source_type} for _ in splits]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=splits,
        metadatas=metadatas
    )

    added_sources.append({"id": str(uuid.uuid4()), "name": source_name, "type": source_type})
    return len(splits)

@app.post("/api/scrape")
async def scrape_url(req: ScrapeRequest):
    try:
        response = requests.get(req.url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        num_chunks = add_texts_to_chroma(clean_text, req.url, "url")
        return {"status": "success", "message": f"Successfully scraped {req.url} into {num_chunks} chunks."}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()
    text = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".docx":
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == ".csv":
            decoded = content.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(decoded))
            for row in reader:
                text += " ".join(row) + "\n"
        elif ext in [".txt", ".md"]:
            text = content.decode("utf-8", errors="replace")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        num_chunks = add_texts_to_chroma(text, file.filename, "file")
        return {"status": "success", "message": f"Processed {file.filename} into {num_chunks} chunks."}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not collection:
        raise HTTPException(status_code=500, detail="Chroma DB collection is not initialized.")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
        
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([req.message]).tolist()
        
        # Cap n_results to available document count to avoid ChromaDB errors
        doc_count = collection.count()
        n_results = min(5, doc_count) if doc_count > 0 else 0

        context_text = ""
        if n_results > 0:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            if results and 'documents' in results and results['documents']:
                doc_list = results['documents'][0]
                context_text = "\n\n---\n\n".join(doc_list)

        # Build conversation history for multi-turn context
        history_text = ""
        if req.history:
            history_lines = []
            for msg in req.history:
                role = "User" if msg.role == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
            history_text = "\n".join(history_lines) + "\n\n"

        # Construct prompt for the LLM
        context_section = f"Context:\n{context_text}\n\n" if context_text else "No context documents available.\n\n"
        prompt = (
            f"You are an intelligent assistant. Use the following context documents to answer the user's question. "
            f"If the context does not contain the answer, say you don't know based on the provided documents.\n\n"
            f"{context_section}"
            f"{history_text}"
            f"User: {req.message}\n\nAnswer:"
        )
             
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return {"response": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
