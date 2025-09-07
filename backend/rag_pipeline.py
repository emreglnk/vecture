import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
import logging
import os
import requests
import io
import asyncio
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class VectorIndex:
    """FAISS tabanlı vector index sınıfı"""
    
    def __init__(self, dim: int = 1536):  # text-embedding-3-small dimension
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity için Inner Product
        self.ids = []  # (tokenId, chunkId) eşleşmesi
        self.metadata = {}  # tokenId -> metadata mapping
        
    def add(self, vectors: np.ndarray, id_pairs: List[Tuple[int, str]], metadata: Dict[int, Dict] = None):
        """
        Vector'leri indekse ekle
        
        Args:
            vectors: shape (N, dim) numpy array
            id_pairs: [(tokenId, chunkId), ...] listesi
            metadata: tokenId -> metadata dict
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dim}")
            
        # L2 normalize for cosine similarity
        vectors_normalized = vectors.astype('float32')
        faiss.normalize_L2(vectors_normalized)
        
        # Add to FAISS index
        self.index.add(vectors_normalized)
        
        # Store ID mappings
        self.ids.extend(id_pairs)
        
        # Store metadata
        if metadata:
            self.metadata.update(metadata)
            
        logger.info(f"Added {len(id_pairs)} vectors to index. Total: {self.index.ntotal}")
        
    def search(self, query_vector: np.ndarray, k: int = 8) -> List[Dict[str, Any]]:
        """
        Similarity search yap
        
        Args:
            query_vector: shape (dim,) numpy array
            k: return top-k results
            
        Returns:
            List of {"tokenId": int, "chunkId": str, "score": float, "metadata": dict}
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
            
        # Reshape and normalize query
        q = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(q)
        
        # Search
        scores, indices = self.index.search(q, min(k, self.index.ntotal))
        
        # Format results
        hits = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            token_id, chunk_id = self.ids[idx]
            hit = {
                "tokenId": int(token_id),
                "chunkId": str(chunk_id),
                "score": float(scores[0][rank]),
                "rank": rank + 1
            }
            
            # Add metadata if available
            if token_id in self.metadata:
                hit["metadata"] = self.metadata[token_id]
                
            hits.append(hit)
            
        return hits
    
    def get_stats(self) -> Dict[str, Any]:
        """Index istatistiklerini döndür"""
        logger.debug(f"get_stats called - index type: {type(self.index)}, index.ntotal: {self.index.ntotal}")
        logger.debug(f"self.ids length: {len(self.ids)}, self.ids content: {self.ids[:5] if self.ids else 'empty'}")
        
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "unique_tokens": len(set(token_id for token_id, _ in self.ids)),
            "total_chunks": len(self.ids)
        }
        logger.debug(f"Final stats: {stats}")
        return stats
    
    def clear(self):
        """Mevcut index'in içeriğini temizler ve sıfırlar."""
        logger.info(f"Clearing index. Current size: {self.index.ntotal}")
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = []
        self.metadata = {}
        logger.info("Index has been cleared.")
    
    def remove_token(self, token_id: int):
        """
        Belirli bir token'ın tüm chunk'larını indexten çıkar
        Not: FAISS IndexFlatIP remove desteklemez, yeniden build gerekir
        """
        # Filter out the token
        new_ids = [(tid, cid) for tid, cid in self.ids if tid != token_id]
        
        if len(new_ids) == len(self.ids):
            logger.warning(f"Token {token_id} not found in index")
            return
            
        # Rebuild index (expensive operation)
        logger.info(f"Rebuilding index to remove token {token_id}")
        
        # This is a simplified version - in production, you'd want to store vectors separately
        # and rebuild from stored data
        self.ids = new_ids
        if token_id in self.metadata:
            del self.metadata[token_id]
            
        logger.info(f"Removed token {token_id}. New size: {len(self.ids)}")

def load_all_nfts_to_index(vector_index: VectorIndex):
    """Contract'taki tüm NFT'leri otomatik olarak index'e yükle"""
    try:
        # Web3 bağlantısı
        rpc_url = os.getenv("RPC_URL", "https://sepolia.infura.io/v3/YOUR_PROJECT_ID")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        
        if not contract_address:
            logger.warning("CONTRACT_ADDRESS not found in environment")
            return
            
        # Web3 instance
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Contract address'i checksum format'a çevir
        contract_address = w3.to_checksum_address(contract_address)
        
        # Contract ABI (minimal - sadece gerekli fonksiyonlar)
        contract_abi = [
            {
                "inputs": [],
                "name": "nextId",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                "name": "getMeta",
                "outputs": [{
                    "components": [
                        {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
                        {"internalType": "bytes32", "name": "vectorHash", "type": "bytes32"},
                        {"internalType": "string", "name": "manifestCID", "type": "string"},
                        {"internalType": "string", "name": "sourceUrl", "type": "string"}
                    ],
                    "internalType": "struct vecture_01.Meta",
                    "name": "",
                    "type": "tuple"
                }],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Get next ID (total minted NFTs + 1)
        try:
            next_id = contract.functions.nextId().call()
            total_supply = next_id - 1  # nextId starts from 1, so total minted is nextId - 1
            logger.info(f"Found {total_supply} NFTs in contract (nextId: {next_id})")
        except Exception as e:
            logger.error(f"Failed to get nextId: {e}")
            return
            
        if total_supply == 0:
            logger.info("No NFTs found in contract")
            return
            
        # Her NFT için metadata ve embedding yükle
        loaded_count = 0
        for token_id in range(1, next_id):
            try:
                # Token metadata al
                meta = contract.functions.getMeta(token_id).call()
                manifest_cid = meta[2]  # manifestCID
                
                if not manifest_cid:
                    continue
                    
                # IPFS URI'yi HTTP'ye çevir
                if manifest_cid.startswith("ipfs://"):
                    metadata_url = manifest_cid.replace("ipfs://", "https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/")
                else:
                    metadata_url = manifest_cid
                    
                # Metadata indir
                response = requests.get(metadata_url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to download metadata for NFT {token_id}: HTTP {response.status_code}")
                    continue
                    
                metadata = response.json()
                logger.debug(f"NFT {token_id} metadata: {metadata}")
                
                # Embedding URI'yi bul
                embedding_uri = None
                if "embedding" in metadata and "file" in metadata["embedding"] and "uri" in metadata["embedding"]["file"]:
                    embedding_uri = metadata["embedding"]["file"]["uri"]
                elif "properties" in metadata and "embeddingURI" in metadata["properties"]:
                    embedding_uri = metadata["properties"]["embeddingURI"]
                elif "embeddingURI" in metadata:
                    embedding_uri = metadata["embeddingURI"]
                    
                if not embedding_uri:
                    logger.warning(f"No embedding URI found for NFT {token_id}")
                    continue
                    
                logger.debug(f"NFT {token_id} embedding URI: {embedding_uri}")
                    
                # Embedding indir
                if embedding_uri.startswith("ipfs://"):
                    embedding_url = embedding_uri.replace("ipfs://", "https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/")
                else:
                    embedding_url = embedding_uri
                    
                embedding_response = requests.get(embedding_url, timeout=10)
                if embedding_response.status_code != 200:
                    continue
                    
                # Numpy array yükle
                embedding_vector = np.load(io.BytesIO(embedding_response.content))
                
                # Index'e ekle
                vector_index.add(
                    embedding_vector.reshape(1, -1),
                    [(token_id, "c0")],
                    {token_id: metadata}
                )
                
                loaded_count += 1
                logger.info(f"Loaded NFT {token_id} to index")
                
            except Exception as e:
                logger.warning(f"Failed to load NFT {token_id}: {e}")
                continue
                
        logger.info(f"Successfully loaded {loaded_count}/{total_supply} NFTs to index")
        
    except Exception as e:
        logger.error(f"Failed to auto-load NFTs: {e}")

# Global index instance
VECTOR_INDEX = VectorIndex()

# Startup'ta NFT'leri yükle (sadece RPC_URL doğru ayarlanmışsa)
rpc_url = os.getenv("RPC_URL")
logger.debug(f"Auto-loading check: RPC_URL={rpc_url}")
if rpc_url and "your-infura-project-id" not in rpc_url:
    try:
        logger.info("Auto-loading NFTs to index...")
        load_all_nfts_to_index(VECTOR_INDEX)
    except Exception as e:
        logger.error(f"Auto-loading failed: {e}")
else:
    logger.info("Auto-loading skipped: RPC_URL not configured properly. Use manual /add_index endpoint.")

def get_vector_index() -> VectorIndex:
    """Global vector index'i döndür"""
    return VECTOR_INDEX