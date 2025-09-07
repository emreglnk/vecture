import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv

# OpenAI SDK (>=1.0 style)
from openai import OpenAI

# Local pipeline utilities
import rag_pipeline as rp
from rag_pipeline import get_vector_index, load_all_nfts_to_index


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY is not set. Embeddings or RAG may fail.")
    return OpenAI(api_key=api_key)


def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")


def ensure_index_loaded(vector_index, always_load: bool = False) -> None:
    ntotal = getattr(getattr(vector_index, "index", None), "ntotal", 0)
    if always_load or ntotal == 0:
        logging.info("Loading NFTs into index (this may take some time)...")
        load_all_nfts_to_index(vector_index)
        ntotal = getattr(getattr(vector_index, "index", None), "ntotal", 0)
        logging.info(f"Index loaded. ntotal={ntotal}")
    else:
        logging.info(f"Index already populated. ntotal={ntotal}")


def search(vector_index, client: OpenAI, query: str, topk: int = 5) -> List[Dict[str, Any]]:
    q_emb = get_embedding(client, query)
    hits = vector_index.search(q_emb, k=topk)
    return hits


def build_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    # Try common keys to extract textual context; fall back to metadata/json dump
    parts: List[str] = []
    for i, h in enumerate(hits):
        txt = None
        for key in ("text", "content", "chunk_text", "body"):
            if isinstance(h.get(key), str) and h[key].strip():
                txt = h[key]
                break
        if not txt:
            meta = h.get("metadata") or {}
            # Try common metadata text fields
            for key in ("text", "content", "description", "title"):
                if isinstance(meta.get(key), str) and meta[key].strip():
                    txt = meta[key]
                    break
        if not txt:
            # Last resort: compact JSON line
            txt = json.dumps({k: v for k, v in h.items() if k not in ("vector",)}, ensure_ascii=False)
        parts.append(f"[Doc {i+1}]\n{txt}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n\n".join(parts)


def rag_answer(client: OpenAI, query: str, context: str, model: str = "gpt-4o-mini") -> str:
    sys_prompt = (
        "You are a helpful assistant. Answer the user using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't have enough information."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Local FAISS search + OpenAI RAG test runner for NFT index")
    parser.add_argument("--query", type=str, default=None, help="Query text for similarity search and RAG")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results to retrieve")
    parser.add_argument("--rag", action="store_true", help="Run OpenAI RAG over top-K hits")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chat_model", type=str, default="gpt-4o-mini", help="OpenAI chat model for RAG")
    parser.add_argument("--always-load", action="store_true", help="Force loading NFTs into the index even if already populated")
    parser.add_argument("--print-stats", action="store_true", help="Print index stats and exit if no query provided")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.verbose)

    # Diagnostics about rag_pipeline singleton
    vi = get_vector_index()
    logging.info(f"get_vector_index() id={id(vi)}")
    logging.info(f"rag_pipeline.VECTOR_INDEX id={id(rp.VECTOR_INDEX)} same_instance={vi is rp.VECTOR_INDEX}")

    ensure_index_loaded(vi, always_load=args.always_load)

    # Print stats
    try:
        stats = vi.get_stats()
        logging.info(f"Index stats: {stats}")
    except Exception as e:
        logging.warning(f"Could not get stats from vector index: {e}")

    if not args.query:
        if args.print_stats:
            return
        # If no query, provide a hint and exit
        print("No --query provided. Example: python node_test.py --query 'rock and roll nedir' --rag --topk 5")
        return

    client = get_openai_client()

    # Run FAISS search
    hits = search(vi, client, args.query, topk=args.topk)
    print("Top-K results:")
    for i, h in enumerate(hits, 1):
        token = h.get("token_id") or h.get("tokenId")
        chunk = h.get("chunk_id") or h.get("chunkId")
        score = h.get("score")
        title = None
        meta = h.get("metadata") or {}
        if isinstance(meta, dict):
            title = meta.get("title")
        print(f"{i}. token={token} chunk={chunk} score={score:.4f} title={title}")

    # Optional RAG answer
    if args.rag:
        try:
            context = build_context_from_hits(hits)
            answer = rag_answer(client, args.query, context, model=args.chat_model)
            print("\nRAG Answer:\n" + answer)
        except Exception as e:
            logging.error(f"RAG failed: {e}")


if __name__ == "__main__":
    main()