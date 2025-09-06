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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

from rag_pipeline import get_vector_index
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure advanced logging with rotation
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs('/app/logs', exist_ok=True)

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
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
    """Get text embedding from OpenAI"""
    try:
        response = client.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

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
    topk: int = Field(5, ge=1, le=50, description="Number of results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")

class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]
    query_embedding_dim: int
    total_results: int

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

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Vector Record API",
        "version": "1.0.0",
        "endpoints": [
            "/embed",
            "/upload_and_manifest",
            "/search",
            "/add_index",
            "/stats"
        ]
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

@app.post("/add_index")
async def add_to_index(request: AddIndexRequest):
    """Add embedding from IPFS to search index"""
    
    try:
        # Download embedding from IPFS
        ipfs_url = request.embedding_uri.replace("ipfs://", "https://ipfs.io/ipfs/")
        
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

@app.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get vector index statistics"""
    vector_index = get_vector_index()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)