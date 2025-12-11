# 🏛️ Moroccan Legal RAG Assistant  
**Fine-tuned FLAN-T5 for Moroccan Family Code Analysis**

## 🌐 Project Overview  
In the complex landscape of Moroccan legal documentation, accessing and interpreting the Family Code requires specialized expertise.  
This project implements a **Retrieval-Augmented Generation (RAG)** system fine-tuned specifically on Moroccan legal texts, providing instant, accurate answers to legal questions in French contexts.

Our system bridges the gap between **legal complexity** and **public accessibility**, offering a specialized AI assistant that understands Moroccan legal terminology, articles, and procedures.

## 🎯 Objectives  
- Provide **accurate, context-aware answers** to Moroccan Family Code questions  
- **Fine-tune FLAN-T5** on Moroccan legal texts for domain specialization  
- Implement **vector search** for precise legal document retrieval  
- Compare **Original vs Fine-tuned** model performance  
- Create an **intuitive web interface** for legal professionals and citizens  
# 🚀 Features Overview

### **1. Model Comparison UI**
Compare:
- Original FLAN-T5 Large  
- Fine-tuned FLAN-T5 (trained on Moroccan Family Code)

Metrics displayed:
- Precision  
- Speed  
- Completeness  
- Citation accuracy  
- Final LL.M judgement

### **2. Legal-Aware Backend (FastAPI)**
- Embedding-based retriever
- Domain-adapted generation
- Article-level grounding

### **3. Vector Database (Qdrant)**
- Stores 768-dim embeddings
- Fast cosine search
- Scalable for large corpora

## ⚙️ Technical Stack  
| Category | Tools / Libraries |
|----------|-------------------|
| **Language Models** | FLAN-T5 Large, SentenceTransformers |
| **Vector Database** | Qdrant |
| **Backend Framework** | FastAPI |
| **Frontend** | HTML/CSS/JavaScript, Jinja2 |
| **Containerization** | Docker |
| **Machine Learning** | Transformers, PyTorch, HuggingFace |
| **Text Processing** | NLTK, regex, pandas |

## 🏗️ Architecture  
### Pipeline Steps  
1. **Document Processing**  
   - PDF extraction → text cleaning → semantic chunking  
   - Article segmentation and metadata extraction  

2. **Embedding & Indexing**  
   - SentenceTransformer embeddings (384 dimensions)  
   - Qdrant vector storage with cosine similarity  

3. **Query Processing**  
   - Question analysis → article number extraction → keyword identification  
   - Vector search → top 5 document retrieval  

4. **Response Generation**  
   - Context construction → prompt engineering  
   - FLAN-T5 generation → response cleaning & formatting  

5. **Web Interface**  
   - Dual-model comparison → real-time statistics → user-friendly display  

## 📊 Performance Comparison  

| Métrique | Original | Fine-tuned | Différence |
|----------|----------|------------|------------|
| **Précision** | 🎯 75% | 🎯 85% | **+10%** |
| **Vitesse** | ⏱️ 58.4s | ⚡ 10.3s | **+82.4%** |
| **Succès questions** | 90% | 95% | **+5%** |
| **Complétude réponses** | 📊 70% | 📊 92% | **+22%** |
| **Citation articles** | 📄 88% | 📄 45% | **-43%** |

### Key Insights  
- **Fine-tuned model is 6× faster** with **10% higher accuracy**  
- **Specialized vocabulary** improves understanding of legal terminology  
- **Trade-off**: Faster responses cite fewer specific articles  
- **Best use case**: Quick, accurate answers for general legal queries  


---

## 📁 Repository Structure

```
moroccan-law-rag-v1/
│
├── docker-compose.yml              # Docker services (Qdrant + optional services)
├── requirements.txt                # Python dependencies
│
├── backend/                        # Core RAG system
│   ├── app/
│   │   ├── main.py                # FastAPI server with endpoints
│   │   └── services/
│   │       ├── rag_pipeline.py    # Main RAG pipeline (Original model)
│   │       ├── rag_finetuned.py   # Fine-tuned RAG pipeline
│   │       ├── embedding.py       # SentenceTransformers embeddings
│   │       └── retrieval.py       # Qdrant search interface
│   │
│   ├── scripts/
│   │   ├── ingest_data.py         # Document ingestion and indexing
│   │   ├── unsupervised_finetuning.py  # Fine-tuning script (8 hours)
│   │   └── pdf_to_text.py         # PDF to text conversion
│   │
│   ├── tests/                     # Unit tests
│   └── requirements.txt           # Backend-specific dependencies
│
├── frontend/                      # Web interface
│   ├── main_frontend.py          # FastAPI frontend server
│   ├── templates/
│   │   └── index.html            # Comparison interface (Original vs Fine-tuned)
│   └── static/                   # CSS / JavaScript / assets
│
├── data/                          # Legal documents
│   ├── raw/                      # Original PDFs (Moroccan Family Code)
│   └── cleaned/                  # Processed text files (for ingestion)
│
├── models/                        # AI models (local - not pushed to GitHub)
│   ├── t5-legal-marocain/        # Fine-tuned model (8 hours training)
│   └── t5-legal-unsupervised/    # Training checkpoints
│
├── docker/                        # Docker configurations
│   └── Dockerfile                # Containerization setup
│
└── README.md                      # Project documentation

```

---


## 🚀 Quick Start  


## 🧩 Prerequisites

Before running the application, ensure that your environment meets the following requirements:

### **System Requirements**
- Python **3.10 or higher** (fully compatible with **3.11**)
- Docker installed (required to run **Qdrant** locally)
- Minimum **8 GB RAM** (16 GB recommended for inference workloads)

### **Internet Access**
- An internet connection is required **only during the first launch** to download:
  - The FLAN-T5 model (original and/or fine-tuned)
  - SentenceTransformers embeddings
- **After the models are downloaded, the entire system can run offline.**

### **Python (pip)**
The following libraries will be installed automatically:
- FastAPI  
- Uvicorn  
- Qdrant Client  
- SentenceTransformers  
- Transformers  
- Pydantic  
- NumPy  

### **Docker**
Ensure the Docker service is running:


# 🚀 How to Run the Application From Scratch

Follow this step-by-step guide to set up and launch the entire application starting from a clean environment.

---

## 1. Install System Requirements

Make sure the following are installed:

- Python **3.10+**  
- Docker (required for Qdrant)  
- Git
- 
## 2. Clone Repository  
```bash
git clone https://github.com/AamirAyoub123/RAG-LawAssistant-Morocco-V1.git
cd RAG-LawAssistant-Morocco-V1
```

## 3. Create a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

## 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```
## 5. Start Qdrant (Vector Database)
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

## 6. Start the Backend API

Navigate to the backend folder:


```bash
cd backend
```

Launch the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Backend available at:
```bash
http://localhost:8000
```

## 7. Start the Frontend
```bash
cd frontend
```
Frontend available at:
```bash
http://localhost:8001
```

---

## 📜 License

This project uses open-source components:

* **FLAN-T5**
* **Qdrant – Apache 2.0**
* **FastAPI – MIT**
* **SentenceTransformers **
  
Legal texts used for **educational and research purposes only.**

---

## 👨‍💻 Author
**Ayoub Aamir**  

🎓 **Master Big Data & IoT**  
📍 *ENSAM Casablanca*  
📧 [aamir.ayoub@ensam-casa.ma](mailto:aamir.ayoub@ensam-casa.ma)

🔗 **Connect with me:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayoub-aamir)  
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AamirAyoub123)


