# backend/app/services/rag_finetuned.py - VERSION SIMPLIFIÉE
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    try:
        from app.services.rag_pipeline import RAGPipeline
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rag_pipeline import RAGPipeline

class FineTunedRAGPipeline(RAGPipeline):
    def __init__(self, use_finetuned=True):
        self.client = QdrantClient(url="http://localhost:6333")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        
        if use_finetuned:
            possible_paths = [
                "C:/moroccan-law-rag-v1/backend/models/t5-legal-marocain",  # Absolute path
                "./models/t5-legal-marocain",  # Relative from backend
                "../models/t5-legal-marocain",  # From backend/app/services
                "models/t5-legal-marocain",  
                "E:/moroccan-law-rag-v1/backend/models/t5-legal-unsupervised/checkpoint-15",  # Checkpoint
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"🔄 Chargement du modèle fine-tuné: {path}")
                    self.model_name = path
                    break
            else:
                print("⚠️  Modèle fine-tuné non trouvé, utilisation du modèle original")
                self.model_name = "google/flan-t5-large"
        else:
            print("🔄 Chargement du modèle FLAN-T5 original...")
            self.model_name = "google/flan-t5-large"
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.tokenizer.model_max_length = 512 
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        print("✅ RAG avec modèle fine-tuné initialisé")

# Test
if __name__ == "__main__":
    rag = FineTunedRAGPipeline(use_finetuned=True)
    print(rag.query("Quel est le délai selon l'article 51 ?", debug=True))