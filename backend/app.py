import io
import json
import hashlib
import os
import base64
import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import asyncio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from rag_pipeline import get_vector_index
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure advanced logging with rotation
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs('/app/logs', exist_ok=True)

# Configure logging with rotation
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            '/app/logs/app.log', 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            mode='a'
        )
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from rag_pipeline import get_vector_index, load_all_nfts_to_index

# Initialize FastAPI app
app = FastAPI(
    title="Vector Record API",
    description="Backend API for Vector Record NFT project with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url} - Client: {request.client.host if request.client else 'unknown'}")
    
    # Process request
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Log response
        logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")
        
        return response
    except Exception as e:
        process_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request failed: {str(e)} - Time: {process_time:.3f}s")
        raise

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# IPFS/Web3.Storage configuration
WEB3_STORAGE_URL = "https://api.web3.storage/upload"
W3_TOKEN = os.getenv("WEB3_STORAGE_TOKEN")
PINATA_JWT = os.getenv("PINATA_JWT")

# Contract configuration
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

# Utility functions
def extract_html_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract metadata from HTML soup"""
    metadata = {}
    
    # Extract title
    title = soup.find('title')
    if title:
        metadata['title'] = title.get_text().strip()
    else:
        # Try og:title
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['title'] = og_title.get('content', '').strip()
        else:
            metadata['title'] = 'Untitled'
    
    # Extract description
    description = soup.find('meta', attrs={'name': 'description'})
    if description:
        metadata['description'] = description.get('content', '').strip()
    else:
        # Try og:description
        og_description = soup.find('meta', property='og:description')
        if og_description:
            metadata['description'] = og_description.get('content', '').strip()
        else:
            metadata['description'] = ''
    
    # Extract author
    author = soup.find('meta', attrs={'name': 'author'})
    if author:
        metadata['author'] = author.get('content', '').strip()
    else:
        # Try article:author
        article_author = soup.find('meta', property='article:author')
        if article_author:
            metadata['author'] = article_author.get('content', '').strip()
        else:
            metadata['author'] = ''
    
    # Extract keywords
    keywords = soup.find('meta', attrs={'name': 'keywords'})
    if keywords:
        keywords_content = keywords.get('content', '')
        metadata['keywords'] = [k.strip() for k in keywords_content.split(',') if k.strip()]
    else:
        metadata['keywords'] = []
    
    return metadata

def vector_hash_from_vec(vec: np.ndarray) -> str:
    """Generate deterministic hash from embedding vector"""
    try:
        # Normalize vector
        v = vec.astype("float32")
        v /= np.linalg.norm(v) + 1e-12
        
        # Convert to binary (0/1)
        bits = (v >= 0).astype(np.uint8)
        
        # Fold 1536 bits to 256 bits using XOR
        chunks = np.array_split(bits, 256)
        folded = np.zeros(256, dtype=np.uint8)
        for chunk in chunks:
            folded[:len(chunk)] ^= chunk
        
        # Convert to bytes and hash
        b = bytes(folded)
        h = hashlib.sha3_256(b).hexdigest()
        return "0x" + h
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector hash generation failed: {str(e)}")

def ipfs_upload_bytes(name: str, data: bytes) -> str:
    """Upload bytes to IPFS via Web3.Storage or Pinata (fallback)"""
    # Prefer Web3.Storage if configured
    if W3_TOKEN:
        try:
            response = requests.post(
                WEB3_STORAGE_URL,
                headers={"Authorization": f"Bearer {W3_TOKEN}"},
                files={"file": (name, data, "application/octet-stream")},
                timeout=30
            )
            response.raise_for_status()
            cid = response.json()["cid"]
            return f"ipfs://{cid}"
        except requests.RequestException as e:
            logger.error(f"IPFS upload via Web3.Storage failed: {e}")
            raise HTTPException(status_code=500, detail=f"IPFS upload failed (Web3.Storage): {str(e)}")

    # Fallback to Pinata if JWT is provided
    if PINATA_JWT:
        try:
            pinata_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
            response = requests.post(
                pinata_url,
                headers={"Authorization": f"Bearer {PINATA_JWT}"},
                files={"file": (name, data, "application/octet-stream")},
                timeout=30
            )
            response.raise_for_status()
            cid = response.json().get("IpfsHash")
            if not cid:
                raise ValueError("Pinata response missing IpfsHash")
            return f"ipfs://{cid}"
        except (requests.RequestException, ValueError) as e:
            logger.error(f"IPFS upload via Pinata failed: {e}")
            raise HTTPException(status_code=500, detail=f"IPFS upload failed (Pinata): {str(e)}")

    # If neither credential exists
    raise HTTPException(status_code=500, detail="No IPFS credentials configured. Set WEB3_STORAGE_TOKEN or PINATA_JWT")

def canonicalize_text(text: str) -> str:
    """Canonicalize text for consistent hashing"""
    # Simple canonicalization - in production, use more sophisticated methods
    return text.lower().strip()

def sha3_hex(data: bytes) -> str:
    """Generate SHA3-256 hash"""
    return "0x" + hashlib.sha3_256(data).hexdigest()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding vector for text using OpenAI API with improved error handling"""
    try:
        response = client.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def ensure_index_loaded(vector_index, always_load: bool = False) -> None:
    """Ensure the vector index is loaded with NFT data"""
    try:
        ntotal = getattr(getattr(vector_index, "index", None), "ntotal", 0)
        if always_load or ntotal == 0:
            logger.info("Loading NFTs into index (this may take some time)...")
            load_all_nfts_to_index(vector_index)
            ntotal = getattr(getattr(vector_index, "index", None), "ntotal", 0)
            logger.info(f"Index loaded. ntotal={ntotal}")
        else:
            logger.info(f"Index already populated. ntotal={ntotal}")
    except Exception as e:
        logger.error(f"Failed to ensure index is loaded: {e}")
        raise HTTPException(status_code=500, detail=f"Index loading failed: {str(e)}")

def search_with_embeddings(vector_index, client_instance: OpenAI, query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """Search vector index using query embeddings"""
    try:
        q_emb = get_embedding(query)
        hits = vector_index.search(q_emb, k=topk)
        return hits
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

def rag_answer_with_context(client_instance: OpenAI, query: str, context: str, model: str = "gpt-4o-mini") -> str:
    """Generate RAG answer using OpenAI with improved system prompt"""
    try:
        sys_prompt = (
            "You are a helpful assistant. Answer the user using ONLY the provided context. "
            "If the answer is not contained in the context, say you don't have enough information. "
            "Be concise and accurate in your response."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        resp = client_instance.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"RAG answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG answer generation failed: {str(e)}")

# Pydantic models
class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    model: str = Field("text-embedding-3-small", description="Embedding model to use")

class EmbedResponse(BaseModel):
    dim: int
    vector: List[float]
    model: str

class UploadAndManifestRequest(BaseModel):
    title: str = Field(..., description="Document title")
    source_url: Optional[str] = Field(None, description="Source URL (optional)")
    raw_text: str = Field(..., description="Raw text content")
    tags: List[str] = Field(default_factory=list, description="Optional tags")

class UploadAndManifestResponse(BaseModel):
    contentHash: str
    vectorHash: str
    manifestCID: str
    embeddingURI: str
    canonicalText: str
    sourceUrl: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    topk: int = Field(10, ge=1, le=50, description="Number of results to return")
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")

class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]
    query_embedding_dim: int
    total_results: int

# New models for RAG
class RagRequest(BaseModel):
    query: str = Field(..., description="LLM query text")
    topk: int = Field(5, ge=1, le=50, description="Number of context hits")
    chat_model: str = Field("gpt-4o-mini", description="OpenAI chat model for RAG")

class RagResponse(BaseModel):
    answer: str
    hits: List[Dict[str, Any]]
    query_embedding_dim: int
    used_topk: int
    model: str

class AddIndexRequest(BaseModel):
    tokenId: int = Field(..., description="NFT token ID")
    embedding_uri: str = Field(..., description="IPFS URI of embedding file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class IndexStatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    unique_tokens: int
    total_chunks: int

class FetchUrlRequest(BaseModel):
    url: str = Field(..., description="URL to fetch content from")

class FetchUrlResponse(BaseModel):
    url: str
    title: str
    content: str
    success: bool
    description: str = Field(default="", description="Page description from meta tags")
    author: str = Field(default="", description="Page author from meta tags")
    keywords: List[str] = Field(default_factory=list, description="Page keywords from meta tags")

# Render function for vector visualization
def render_dot_image(vec: np.ndarray, label: str = "") -> io.BytesIO:
    """Render a 1536-dimensional vector as a dot visualization"""
    if vec.size != 1536:
        raise ValueError("Vector length must be 1536.")
    mat = vec.reshape(32, 48)

    vals = mat.astype(float)
    vals = np.clip(vals, np.percentile(vals, 1), np.percentile(vals, 99))
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    sizes = 0 + norm * (200 - 10) * .3

    yy, xx = np.indices((32, 48))
    x = xx.ravel()
    y = yy.ravel()
    s = sizes.ravel()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(x, y, s=s, c="#FFD400", edgecolors="none")
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(31.5, -0.5)
    ax.axis("off")
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.04, label, color="white", ha="center", va="center", fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", dpi=200, bbox_inches="tight", pad_inches=0.2, facecolor="black", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf

# API endpoints
@app.get("/")
async def root():
    # Dinamik olarak kayıtlı yolları app.router.routes üzerinden topla (docs/openapi hariç)
    exclude = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    endpoints = sorted({getattr(r, "path", "") for r in getattr(app, "router", app).routes if getattr(r, "path", "") not in exclude})
    return {
        "message": "Vector Record API",
        "version": "1.0.0",
        "endpoints": endpoints,
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embedding for given text"""
    vector = get_embedding(request.text, request.model)
    
    return EmbedResponse(
        dim=len(vector),
        vector=vector.tolist(),
        model=request.model
    )

@app.post("/upload_and_manifest", response_model=UploadAndManifestResponse)
async def upload_and_manifest(request: UploadAndManifestRequest):
    """Process text, generate embedding, upload to IPFS, and create manifest"""
    
    # 1. Canonicalize text and generate content hash
    canonical_text = canonicalize_text(request.raw_text)
    content_hash = sha3_hex(canonical_text.encode("utf-8"))
    
    logger.info(f"Processing text with content hash: {content_hash}")
    
    # 2. Generate embedding
    embedding_vector = get_embedding(canonical_text)
    
    # 2.5. Generate vector hash
    vector_hash = vector_hash_from_vec(embedding_vector)
    
    # 3. Upload embedding to IPFS
    npy_buffer = io.BytesIO()
    np.save(npy_buffer, embedding_vector)
    npy_buffer.seek(0)
    embedding_cid = ipfs_upload_bytes("embeddings.npy", npy_buffer.read())
    
    # 4. Create and upload manifest
    manifest = {
        "schema": "vector-record/1.0",
        "content": {
            "title": request.title,
            "canonicalization": "lower_strip",
            "source_refs": (
                [{"type": "http", "uri": request.source_url}] 
                if request.source_url else []
            ),
            "content_hash": content_hash,
            "created_at": int(np.datetime64('now').astype('datetime64[s]').astype(int))
        },
        "embedding": {
            "model_id": "text-embedding-3-small",
            "dim": len(embedding_vector),
            "metric": "cosine",
            "file": {
                "uri": embedding_cid,
                "format": "npy"
            },
            "vector_hash": vector_hash,
            "hash_method": "sign-fold-xor"
        },
        "tags": request.tags
    }
    
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_cid = ipfs_upload_bytes("manifest.json", manifest_bytes)
    
    logger.info(f"Created manifest with CID: {manifest_cid}")
    
    return UploadAndManifestResponse(
        contentHash=content_hash,
        vectorHash=vector_hash,
        manifestCID=manifest_cid,
        embeddingURI=embedding_cid,
        canonicalText=canonical_text,
        sourceUrl=request.source_url
    )

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """Search for similar vectors in the index"""
    
    # Generate query embedding
    query_embedding = get_embedding(request.query)
    
    # Search in vector index
    vector_index = get_vector_index()
    hits = vector_index.search(query_embedding, k=request.topk)
    
    # Filter by minimum score if specified
    if request.min_score > 0:
        hits = [hit for hit in hits if hit["score"] >= request.min_score]
    
    logger.info(f"Search query: '{request.query}' returned {len(hits)} results")
    
    return SearchResponse(
        hits=hits,
        query_embedding_dim=len(query_embedding),
        total_results=len(hits)
    )

# Helper to build RAG context from hits
def build_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """Build context string from search hits with improved text extraction"""
    # Try common keys to extract textual context; fall back to metadata/json dump
    parts: List[str] = []
    for i, h in enumerate(hits):
        txt = None
        # Try common text fields first
        for key in ("text", "content", "chunk_text", "body"):
            if isinstance(h.get(key), str) and h[key].strip():
                txt = h[key]
                break
        
        # If no direct text found, try metadata
        if not txt:
            meta = h.get("metadata") or {}
            # Try common metadata text fields
            for key in ("text", "content", "description", "title"):
                if isinstance(meta.get(key), str) and meta[key].strip():
                    txt = meta[key]
                    break
        
        # Last resort: compact JSON line (excluding vector data)
        if not txt:
            txt = json.dumps({k: v for k, v in h.items() if k not in ("vector",)}, ensure_ascii=False)
        
        parts.append(f"[Doc {i+1}]\n{txt}")
        
        # Check character limit
        if sum(len(p) for p in parts) > max_chars:
            break
    
    return "\n\n".join(parts)

@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(request: RagRequest):
    """Run RAG: search FAISS for context and generate LLM answer with improved functionality"""
    try:
        # 1) Get vector index and ensure it's loaded
        vector_index = get_vector_index()
        ensure_index_loaded(vector_index)
        
        # 2) Search using improved search function
        hits = search_with_embeddings(vector_index, client, request.query, topk=request.topk)
        
        # 3) Build context with improved text extraction
        context = build_context_from_hits(hits)
        
        # 4) Generate answer using improved RAG function
        answer = rag_answer_with_context(client, request.query, context, model=request.chat_model)
        
        # 5) Get query embedding for response metadata
        query_embedding = get_embedding(request.query)
        
        return RagResponse(
            answer=answer,
            hits=hits,
            query_embedding_dim=len(query_embedding),
            used_topk=len(hits),
            model=request.chat_model
        )
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")

@app.post("/add_index", include_in_schema=False)
async def add_to_index(request: AddIndexRequest):
    """Add embedding from IPFS to search index"""
    
    try:
        # Download embedding from IPFS
        ipfs_url = request.embedding_uri.replace("ipfs://", "https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/")
        
        logger.info(f"Downloading embedding from: {ipfs_url}")
        
        response = requests.get(ipfs_url, timeout=30)
        response.raise_for_status()
        
        # Load numpy array
        embedding_vector = np.load(io.BytesIO(response.content))
        
        # Add to vector index
        vector_index = get_vector_index()
        vector_index.add(
            embedding_vector.reshape(1, -1),
            [(request.tokenId, "c0")],  # Single chunk for MVP
            {request.tokenId: request.metadata} if request.metadata else None
        )
        
        logger.info(f"Added token {request.tokenId} to search index")
        
        return {"success": True, "tokenId": request.tokenId, "message": "Added to index"}
        
    except Exception as e:
        logger.error(f"Failed to add to index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add to index: {str(e)}")

@app.post("/clear_index")
async def clear_index():
    """Clear all vectors from the index without re-instantiating it."""
    try:
        # Global index nesnesini al
        vector_index = get_vector_index()
        
        # Nesneyi değiştirmek yerine içeriğini temizle
        vector_index.clear()
        
        return {"message": "Index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get vector index statistics"""
    vector_index = get_vector_index()
    try:
        import rag_pipeline as rp
        logger.info(f"/stats diag: get_vector_index id={id(vector_index)}, type={type(vector_index)}, module={getattr(vector_index.__class__, '__module__', None)}")
        logger.info(f"/stats diag: rp.VECTOR_INDEX id={id(rp.VECTOR_INDEX)}, same={vector_index is rp.VECTOR_INDEX}")
        ntotal = getattr(getattr(vector_index, "index", None), "ntotal", None)
        logger.info(f"/stats diag: ntotal={ntotal}, ids_len={len(getattr(vector_index, 'ids', []))}")
    except Exception as e:
        logger.error(f"/stats diag failed: {e}")
    stats = vector_index.get_stats()
    
    return IndexStatsResponse(**stats)

@app.get("/stats_diag", include_in_schema=True)
async def get_index_stats_diag():
    """Diagnostic info about the vector index instance and FAISS state"""
    vector_index = get_vector_index()
    try:
        import rag_pipeline as rp
        info = {
            "get_vector_index_id": id(vector_index),
            "rp_VECTOR_INDEX_id": id(rp.VECTOR_INDEX),
            "same_instance": vector_index is rp.VECTOR_INDEX,
            "ntotal": getattr(getattr(vector_index, "index", None), "ntotal", None),
            "ids_len": len(getattr(vector_index, 'ids', [])),
            "ids_sample": getattr(vector_index, 'ids', [])[:5],
        }
    except Exception as e:
        info = {"error": str(e)}
    return info
    stats = vector_index.get_stats()
    
    return IndexStatsResponse(**stats)

@app.post("/fetch_url", response_model=FetchUrlResponse)
async def fetch_url_content(request: FetchUrlRequest):
    """Fetch content from a URL"""
    try:
        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(request.url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata from HTML
        metadata = extract_html_metadata(soup)
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Remove extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        logger.info(f"Successfully fetched content from: {request.url}")
        
        return FetchUrlResponse(
            url=request.url,
            title=metadata.get('title', 'No title found'),
            content=content,
            success=True,
            description=metadata.get('description', ''),
            author=metadata.get('author', ''),
            keywords=metadata.get('keywords', [])
        )
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch URL {request.url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing URL {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.delete("/index/{token_id}")
async def remove_from_index(token_id: int):
    """Remove token from search index"""
    try:
        vector_index = get_vector_index()
        vector_index.remove_token(token_id)
        
        return {"success": True, "tokenId": token_id, "message": "Removed from index"}
        
    except Exception as e:
        logger.error(f"Failed to remove from index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove from index: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "web3_storage_configured": bool(os.getenv("WEB3_STORAGE_TOKEN")),
        "pinata_configured": bool(os.getenv("PINATA_JWT")),
        "contract_address": CONTRACT_ADDRESS
    }

@app.get("/logs")
async def get_logs(lines: int = 50):
    """Get recent log entries"""
    try:
        log_file_path = '/app/logs/app.log'
        if not os.path.exists(log_file_path):
            return {"logs": [], "message": "Log file not found"}
        
        with open(log_file_path, 'r') as f:
            log_lines = f.readlines()
        
        # Get last N lines
        recent_logs = log_lines[-lines:] if len(log_lines) > lines else log_lines
        
        return {
            "logs": [line.strip() for line in recent_logs],
            "total_lines": len(log_lines),
            "showing_lines": len(recent_logs)
        }
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

@app.get("/logs/errors")
async def get_error_logs(lines: int = 20):
    """Get recent error log entries only"""
    try:
        log_file_path = '/app/logs/app.log'
        if not os.path.exists(log_file_path):
            return {"error_logs": [], "message": "Log file not found"}
        
        with open(log_file_path, 'r') as f:
            log_lines = f.readlines()
        
        # Filter error logs
        error_logs = [line.strip() for line in log_lines if 'ERROR' in line]
        recent_errors = error_logs[-lines:] if len(error_logs) > lines else error_logs
        
        return {
            "error_logs": recent_errors,
            "total_errors": len(error_logs),
            "showing_errors": len(recent_errors)
        }
    except Exception as e:
        logger.error(f"Error reading error logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading error logs: {str(e)}")

# Uygulama başlangıcında (server process içinde) otomatik yükleme
@app.on_event("startup")
async def startup_autoload_nfts():
    try:
        rpc_url = os.getenv("RPC_URL")
        logger.debug(f"Startup auto-load check: RPC_URL={rpc_url}")
        if rpc_url and "your-infura-project-id" not in rpc_url:
            vector_index = get_vector_index()
            # Duplicate yüklemeyi önlemek için kontrol et
            ntotal = getattr(getattr(vector_index, "index", None), "ntotal", 0)
            if ntotal == 0:
                logger.info("Startup: Auto-loading NFTs to index (server process)...")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, load_all_nfts_to_index, vector_index)
                logger.info(f"Startup: Auto-loading complete. Total vectors: {vector_index.index.ntotal}")
            else:
                logger.info(f"Startup: Index already populated (ntotal={ntotal}), skipping auto-load.")
        else:
            logger.info("Startup: Auto-loading skipped due to RPC_URL config.")
        # Route diagnostics
        try:
            route_paths = [getattr(r, 'path', None) for r in getattr(app.router, 'routes', [])]
            logger.info(f"Startup: Registered routes count={len(route_paths)} paths={route_paths}")
            logger.info(f"Startup: Has /stats_diag? {'/stats_diag' in route_paths}")
        except Exception as e:
            logger.error(f"Startup: Route introspection failed: {e}")
    except Exception as e:
        logger.error(f"Startup: Auto-loading failed: {e}")

@app.get("/render")
def render_from_url(url: str = Query(...), label: str = Query("")):
    """Render a vector visualization from a .npy file URL"""
    try:
        # replace ipfs:// with https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/
        url = url.replace("ipfs://","https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/")
        r = requests.get(url)
        r.raise_for_status()
        arr = np.load(io.BytesIO(r.content))

        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2 and arr.shape[1] == 1536:
            vec = arr[0]
        else:
            return {"error": "Invalid .npy format. Must be a 1536-length vector or (N, 1536)."}

        img_buf = render_dot_image(vec, label)
        return Response(content=img_buf.read(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Render error: {e}")
        return {"error": str(e)}

@app.post("/render_vector")
def render_vector_direct(vector: List[float], label: str = ""):
    """Render vector visualization from direct vector input"""
    try:
        vec = np.array(vector)
        if vec.size != 1536:
            return {"error": "Vector length must be 1536."}
        
        img_buf = render_dot_image(vec, label)
        return Response(content=img_buf.read(), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)