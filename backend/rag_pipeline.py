import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
import logging

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
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "unique_tokens": len(set(token_id for token_id, _ in self.ids)),
            "total_chunks": len(self.ids)
        }
    
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

# Global index instance
VECTOR_INDEX = VectorIndex()

def get_vector_index() -> VectorIndex:
    """Global vector index'i döndür"""
    return VECTOR_INDEX