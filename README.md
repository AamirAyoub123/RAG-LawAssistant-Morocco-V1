# 🇲🇦 Moroccan Legal RAG Assistant  
**Fine-tuned FLAN-T5 for Moroccan Family Code Analysis**

## 🌐 Project Overview  
In the complex landscape of Moroccan legal documentation, accessing and interpreting the Family Code requires specialized expertise.  
This project implements a **Retrieval-Augmented Generation (RAG)** system fine-tuned specifically on Moroccan legal texts, providing instant, accurate answers to legal questions in French and Arabic contexts.

Our system bridges the gap between **legal complexity** and **public accessibility**, offering a specialized AI assistant that understands Moroccan legal terminology, articles, and procedures.

## 🎯 Objectives  
- Provide **accurate, context-aware answers** to Moroccan Family Code questions  
- **Fine-tune FLAN-T5** on Moroccan legal texts for domain specialization  
- Implement **vector search** for precise legal document retrieval  
- Compare **Original vs Fine-tuned** model performance  
- Create an **intuitive web interface** for legal professionals and citizens  

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

## 🚀 Quick Start  

### 1. Clone Repository  
```bash
git clone https://github.com/AamirAyoub123/RAG-LawAssistant-Morocco-V1.git
cd RAG-LawAssistant-Morocco-V1

---

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
