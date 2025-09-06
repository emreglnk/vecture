# VECTURE — Vector Knowledge Platform

Web3 tabanlı içerik vektörleştirme ve arama platformu. İçerikleri doğrulanabilir vektör varlıklarına dönüştürür.

## Özellikler

- **Vektör Dönüştürme**: Metinleri OpenAI embedding modeli ile vektörlere çevirir
- **Semantik Arama**: FAISS ile hızlı benzerlik araması
- **IPFS Depolama**: Merkezi olmayan içerik saklama
- **NFT Entegrasyonu**: Blockchain tabanlı sahiplik kaydı
- **RESTful API**: FastAPI ile backend hizmetleri
- **Web Arayüzü**: Basit HTML/JS frontend

## Teknik Mimari

### Sistem Bileşenleri
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │    │   Blockchain    │
│   (HTML/JS)     │◄──►│   (FastAPI)     │◄──►│   (Ethereum)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │      IPFS       │
                       │ (Web3.Storage)  │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     FAISS       │
                       │ (Vector Index)  │
                       └─────────────────┘
```

### Veri Akışı
```
Metin Girişi → OpenAI Embedding → FAISS Index → Arama Sonuçları
     │              │                  ▲             │
     ▼              ▼                  │             ▼
  IPFS Hash → Blockchain NFT ──────────┘      Kullanıcı Arayüzü
```

### İşleyiş Adımları
1. **Metin Yükleme**: Kullanıcı metin içeriği yükler
2. **Embedding Üretimi**: OpenAI API ile vektör oluşturulur
3. **IPFS Depolama**: Orijinal metin IPFS'e kaydedilir
4. **Vektör İndeksleme**: FAISS index'ine eklenir
5. **NFT Minting**: Opsiyonel blockchain kaydı
6. **Arama**: Benzer içerikler vektör benzerliği ile bulunur

## 📁 Project Structure

```
vector-mvp/
├── contracts/          # Smart contracts (Solidity)
│   ├── VectorRecordNFT.sol
│   ├── hardhat.config.ts
│   ├── package.json
│   └── scripts/
├── backend/           # API server (Python/FastAPI)
│   ├── app.py
│   ├── rag_pipeline.py
│   └── requirements.txt
├── frontend/          # Web interface (Vanilla JS)
│   ├── index.html
│   └── src/
├── cli/              # Command-line tool (Python)
│   ├── vector_uploader.py
│   └── requirements.txt
└── README.md
```

## Kurulum

### Gereksinimler
- Python 3.8+
- Node.js 16+ (smart contracts için)
- OpenAI API key
- Web3.Storage API key

### Hızlı Başlangıç

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env  # API keylerini düzenle
python app.py

# Frontend
cd frontend
python -m http.server 3000
```

## Kullanım

### Web Arayüzü
1. `http://localhost:3000` - Ana sayfa
2. `app.html` - Uygulama arayüzü
3. MetaMask bağlantısı
4. Metin yükleme ve arama

### API Endpoints
- `POST /generate-embedding` - Metin vektörü oluştur
- `POST /upload-ipfs` - IPFS'e yükle
- `POST /add-vector` - Vektör index'ine ekle
- `POST /search` - Benzer vektörleri ara
- `GET /index/stats` - İstatistikler

### CLI Kullanımı
```bash
python cli/vector_uploader.py upload document.txt
python cli/vector_uploader.py search "arama terimi"
```

## Konfigürasyon

### Backend (.env)
```env
OPENAI_API_KEY=your_key
WEB3_STORAGE_TOKEN=your_token
ETHEREUM_RPC_URL=https://sepolia.infura.io/v3/your_key
CONTRACT_ADDRESS=0x...
```

## Teknik Detaylar

### Embedding Süreci
```python
# OpenAI API ile vektör oluşturma
embedding = openai.Embedding.create(
    input=text,
    model="text-embedding-3-small"
)
vector = embedding['data'][0]['embedding']
```

### FAISS Index Yönetimi
```python
# Vektör ekleme
index.add(np.array([vector]).astype('float32'))

# Benzerlik araması
scores, indices = index.search(query_vector, k=5)
```

### IPFS Entegrasyonu
```python
# Web3.Storage ile yükleme
response = requests.post(
    'https://api.web3.storage/upload',
    headers={'Authorization': f'Bearer {token}'},
    files={'file': content}
)
```

## Proje Yapısı
```
vector-mvp/
├── backend/           # FastAPI backend
│   ├── app.py        # Ana API server
│   └── rag_pipeline.py # Vektör işleme
├── frontend/         # HTML/JS arayüz
├── contracts/        # Solidity contracts
└── cli/             # Python CLI tool
```

## Lisans
MIT License