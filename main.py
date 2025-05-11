import os
import io
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import uvicorn
from PyPDF2 import PdfReader

from pinecone_utils import init_pinecone_index, upsert_documents, retrieve_from_pinecone, PINECONE_INDEX_NAME
from groq_utils import init_groq_client, get_groq_streaming_response
from web_utils import WebContentFetcher

# Load environment variables
load_dotenv()

# Global variables for clients/index
pinecone_idx = None
groq_llm_client = None
web_fetcher = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global pinecone_idx, groq_llm_client, web_fetcher
    print("Starting up application and initializing services...")
    try:
        pinecone_idx = init_pinecone_index()
        if pinecone_idx is None: raise Exception("Pinecone index init failed.")
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' initialized.") 
    except Exception as e:
        print(f"Pinecone initialization error: {e}")
        pinecone_idx = None

    try:
        groq_llm_client = await init_groq_client()
        if groq_llm_client is None: raise Exception("Groq client init failed.")
        print("Groq client initialized.")
    except Exception as e:
        print(f"Groq client initialization error: {e}")
        groq_llm_client = None
    
    web_fetcher = WebContentFetcher()
    print("WebContentFetcher initialized.")
    
    if not pinecone_idx or not groq_llm_client or not web_fetcher:
        print("Warning: One or more services (Pinecone/Groq/WebFetcher) failed to initialize. API may not function fully.")
    
    yield

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with HTML and UI Support",
    description="API for RAG with PDF and URL upload, and a simple UI.",
    version="0.0.1",
    lifespan=lifespan # Added lifespan manager
)

# Mount static files directory (for UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Pydantic Models ---
class PDFUploadResponse(BaseModel):
    message: str
    filename: str
    total_pages_processed: int

class UrlUploadRequest(BaseModel):
    url: HttpUrl

class UrlUploadResponse(BaseModel):
    message: str
    url: str
    content_length: int

class ChatQueryRequest(BaseModel):
    query: str
    top_k: int = 6

# --- Dependencies ---
async def get_pinecone_index_dependency():
    if pinecone_idx is None: raise HTTPException(status_code=503, detail="Pinecone service unavailable.")
    return pinecone_idx

async def get_groq_client_dependency():
    if groq_llm_client is None: raise HTTPException(status_code=503, detail="Groq service unavailable.")
    return groq_llm_client

async def get_web_fetcher_dependency():
    if web_fetcher is None: raise HTTPException(status_code=503, detail="Web content fetching service unavailable.")
    return web_fetcher

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_index():
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>UI not found</h1><p>Please create static/index.html</p></body></html>", status_code=404)

@app.post("/upload-pdf/", response_model=PDFUploadResponse, summary="Upload PDF, store pages with source/page as top-level fields")
async def upload_pdf_endpoint(file: UploadFile = File(...),
                                current_pinecone_index = Depends(get_pinecone_index_dependency)):
    internal_namespace = "" 
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. PDFs only.")

    documents_to_upsert = []
    pages_processed = 0
    try:
        pdf_content = await file.read()
        pdf_stream = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_stream)
        num_pages = len(reader.pages)
        pages_processed = num_pages

        for i, page_obj in enumerate(reader.pages):
            text_content = page_obj.extract_text()
            if text_content and text_content.strip():
                documents_to_upsert.append({
                    "id": f"{file.filename}-page-{i+1}",
                    "text": text_content.strip(),
                    "source": file.filename,
                    "page_number": i + 1
                })
            else:
                print(f"No text or empty text on page {i+1} of {file.filename}")

        if not documents_to_upsert:
            raise HTTPException(status_code=400, detail=f"No text found in PDF: {file.filename}")

        upsert_result = upsert_documents(current_pinecone_index, documents_to_upsert, namespace=internal_namespace)
        
        return PDFUploadResponse(
            message=upsert_result["message"],
            filename=file.filename,
            total_pages_processed=pages_processed
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        await file.close()

@app.post("/upload-url/", response_model=UrlUploadResponse, summary="Fetch content from URL and store it")
async def upload_url_endpoint(request: UrlUploadRequest,
                              current_pinecone_index = Depends(get_pinecone_index_dependency),
                              current_web_fetcher = Depends(get_web_fetcher_dependency)):
    internal_namespace = ""
    url_str = str(request.url)

    try:
        print(f"Received request to fetch URL: {url_str}")
        text_content = await current_web_fetcher.fetch_and_parse(url_str)

        if text_content.startswith("Error:"):
            if "timed out" in text_content.lower():
                raise HTTPException(status_code=408, detail=text_content)
            elif "status code: 404" in text_content.lower():
                 raise HTTPException(status_code=404, detail=text_content)
            else:
                raise HTTPException(status_code=400, detail=text_content)

        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail=f"No meaningful text content found at URL: {url_str}")

        document_to_upsert = {
            "id": f"url-{url_str}",
            "text": text_content.strip(),
            "source": url_str,
            "page_number": 1
        }

        upsert_result = upsert_documents(current_pinecone_index, [document_to_upsert], namespace=internal_namespace)
        
        return UrlUploadResponse(
            message="URL content processed and attempt to store made.",
            url=url_str,
            content_length=len(text_content)
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error processing URL {url_str}: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

async def generate_chat_stream(current_groq_client, current_pinecone_index, query: str, top_k: int):
    internal_namespace = ""
    retrieved_contexts = retrieve_from_pinecone(
        current_pinecone_index,
        query,
        top_k,
        namespace=internal_namespace
    )

    # Prepare source information and context for LLM
    source_map = {}
    contexts_for_llm = []

    if retrieved_contexts:
        print(f"Retrieved {len(retrieved_contexts)} contexts from Pinecone.")
        for ctx in retrieved_contexts:
            source_key = ctx.get("source", "N/A")
            page_number = ctx.get("page_number", "N/A")
            text_content = ctx.get('text', '')
            
            contexts_for_llm.append(text_content)

            if source_key not in source_map:
                is_url = source_key.startswith("http://") or source_key.startswith("https://")
                source_map[source_key] = {"source": source_key, "pages": [], "is_url": is_url}
            
            if page_number not in source_map[source_key]["pages"] and page_number != "N/A":
                source_map[source_key]["pages"].append(page_number)
        
        # Sort page numbers
        for key in source_map:
            source_map[key]["pages"].sort()

        source_list_for_json = list(source_map.values())    
        full_context_for_llm = "\n\n---\n\n".join(contexts_for_llm)
    else:
        print("No context retrieved from Pinecone for the query.")
        source_list_for_json = [] # Ensure it's an empty list if no contexts
        full_context_for_llm = "No specific context was found to answer the query."

    # Yield sources as a JSON object, followed by a separator
    initial_payload = {"sources": source_list_for_json}
    yield json.dumps(initial_payload) + "\n###LLM_ANSWER###\n"

    # Stream LLM response
    async for chunk in get_groq_streaming_response(current_groq_client, full_context_for_llm, query):
        yield chunk

@app.post("/chat/", summary="Ask question, get streamed answer with source/page of top hit")
async def chat_endpoint(request: ChatQueryRequest,
                        current_pinecone_index = Depends(get_pinecone_index_dependency),
                        current_groq_client = Depends(get_groq_client_dependency)):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return StreamingResponse(
            generate_chat_stream(current_groq_client, current_pinecone_index, request.query, request.top_k),
            media_type="text/event-stream"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error in /chat/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")

@app.get("/", summary="Root endpoint with API status")
async def read_root():
    return {
        "message": "Welcome to RAG API! (Now with URL support and UI placeholder)"
    }

if __name__ == "__main__":
    required_vars = ["PINECONE_API_KEY", "GROQ_API_KEY", "PINECONE_INDEX"]
    if any(not os.getenv(var) for var in required_vars):
        print(f"Error: Missing env vars: {', '.join(v for v in required_vars if not os.getenv(v))}")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000) 