# Vecture

**Vecture** is an open protocol that turns content (articles, blogs, code, etc.) into **vector embeddings** stored on IPFS, while registering the identity on-chain as an **NFT**.  
The goal: make knowledge **verifiable on-chain**, provide **transparent sources for LLM/RAG systems**, and ensure **creators get paid**.

---

## 🚀 Key Features

- **Vector Minting:** Store content as embedding + manifest on IPFS, mint as NFT on-chain.  
- **Two-Level Duplication Guard:**  
  - `contentHash` → blocks exact duplicates  
  - `vectorHash` → blocks semantic duplicates (paraphrases, translations)  
- **Semantic Search + RAG:** FAISS/HNSW-based search + LLM answers with sources.  
- **Future Economy:** Query-based micro-fees automatically split among Indexers, Creators, and the Protocol.

---

## 📦 Architecture Layers

1. **Client UI**  
   - Wallet connect, paste link/text, one-click mint  

2. **Node Service**  
   - Generates embeddings  
   - Computes `contentHash` + `vectorHash`  
   - Uploads manifest + embeddings to IPFS  

3. **Blockchain Layer**  
   - `VectorRecordNFT` contract  
   - Stores manifestCID and hash references  

4. **Indexer & RAG**  
   - Adds embeddings to ANN (FAISS/HNSW)  
   - Processes queries and returns answers via LLM + sources  

5. **(Future) Payment Layer**  
   - `payQuery` mechanism  
   - Fee split: %α indexer, %β creator, %γ protocol  

6. **(Future) Auto-Onboarding**  
   - Wikipedia / Medium / GitHub auto-mint  
   - Creators can later verify and claim ownership  

---

## 🔑 What’s in the MVP?

- Wallet connect + content minting  
- IPFS upload (manifest + embeddings)  
- On-chain NFT registration  
- Basic indexing + FAISS search  
- RAG for attributed answers  

---

## 🛠️ Next Stages

- **Stage 2:** Payment flow (`payQuery`) & micro-royalties  
- **Stage 3:** Multi-indexer, sharding, SLAs & metrics  
- **Stage 4:** Auto-onboarding (Wikipedia/Medium/GitHub) + creator claims  
- **Stage 5:** Domain-specific embedding models (e.g. legal, health), automated citations  

---

## 🌍 Vision

**Vecture** aims to become a **verifiable, sustainable alternative to web search for LLMs**, while building a global knowledge economy where **creators are rewarded fairly**.

---

## 🖼️ High-Level Flow

```mermaid
flowchart LR
  classDef mvp fill:#0b5,stroke:#083,color:#fff
  classDef fut fill:#1b2e4d,stroke:#88b,color:#dfe
  classDef sys fill:#0e1420,stroke:#3a5678,color:#e7f

  subgraph UI[Client UI • Wallet]
    U[User + Wallet]:::sys
    M[Mint Form]:::sys
  end

  subgraph NODE[Node Service]
    EMB[Embedder]:::sys
    HASH[contentHash + vectorHash]:::sys
    IPFS[IPFS Upload]:::sys
  end

  subgraph CHAIN[Blockchain]
    NFT[VectorRecordNFT]:::sys
  end

  subgraph SEARCH[Indexer & RAG]
    IDX[Indexer Node]:::sys
    ANN[ANN Index]:::sys
    RAG[LLM Answer + Sources]:::sys
  end

  %% MVP Flow
  U -->|connect| M:::mvp
  M -->|send content| EMB:::mvp
  EMB -->|embedding| HASH:::mvp
  HASH --> IPFS:::mvp
  IPFS -->|manifestCID| M:::mvp
  M -->|mint| NFT:::mvp
  IPFS --> IDX:::mvp
  IDX --> ANN:::mvp
  U -->|query| ANN:::mvp
  ANN --> RAG:::mvp
  RAG -->|answer + sources| U:::mvp

  %% Future Flow
  HASH -. preCheck near-dup .-> IDX:::fut
  U -. payQuery .-> NFT:::fut

  subgraph PAY[Payments Future]
    SPLIT[Fee Split\nIndexer • Creator • Protocol]:::sys
  end
  NFT -. distribute fees .-> SPLIT:::fut

  subgraph AUTO[Auto-Onboarding Future]
    WIKI[Wikipedia]:::sys
    MED[Medium]:::sys
    GIT[GitHub]:::sys
    VER[Creator Verification]:::sys
  end
  WIKI -. auto-mint .-> IPFS:::fut
  MED  -. auto-mint .-> IPFS:::fut
  GIT  -. auto-mint .-> IPFS:::fut
  VER  -. claim ownership .-> NFT:::fut

  class U,M,EMB,HASH,IPFS,NFT,IDX,ANN,RAG mvp
  class SPLIT,WIKI,MED,GIT,VER fut

Sepholia Contract Addr: 0x3129dd4d0454e94fcc98c7880a730038fd325063
Rise Contract Addr: 0x3129DD4d0454E94fcC98C7880A730038fD325063 ???