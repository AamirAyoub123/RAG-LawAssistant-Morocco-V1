# backend/scripts/ingest_data_improved.py
import os
import re
from typing import Dict, List
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

def semantic_chunk_text(text: str, max_chunk_size: int = 800) -> List[str]:
    """Découpe le texte en chunks sémantiques"""
    chunks = []
    
    # 1. D'abord diviser par articles 
    articles = re.split(r'(?=Article\s+\d+[\.\s\-])', text)
    
    for article in articles:
        article = article.strip()
        if not article or len(article) < 50:
            continue
        
        # Si l'article est raisonnable, le garder entier
        if len(article) <= max_chunk_size:
            chunks.append(article)
        else:
            # Sinon, diviser par paragraphes naturels
            paragraphs = re.split(r'\n\s*\n', article)
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    # 2. Si pas d'articles, diviser par paragraphes
    if not chunks:
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    # Filtrer les chunks trop courts
    chunks = [chunk for chunk in chunks if len(chunk) >= 100]
    
    return chunks

def extract_metadata(text: str) -> Dict:
    """Extrait des métadonnées du texte"""
    metadata = {
        'has_articles': bool(re.search(r'Article\s+\d+', text)),
        'article_numbers': re.findall(r'Article\s+(\d+)', text),
        'has_decret': bool(re.search(r'Décret\s+n°', text, re.IGNORECASE)),
        'has_arrete': bool(re.search(r'Arrêté', text, re.IGNORECASE)),
        'has_loi': bool(re.search(r'Loi\s+n°', text, re.IGNORECASE)),
        'word_count': len(text.split()),
        'contains_dates': bool(re.search(r'\d{1,2}\s+\w+\s+\d{4}', text)),
    }
    return metadata

def ingest_with_metadata():
    """Ingère les documents avec métadonnées enrichies"""
    client = QdrantClient(url="http://localhost:6333")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Créer ou réinitialiser la collection
    try:
        client.delete_collection("moroccan_law")
        print("🗑️  Ancienne collection supprimée")
    except:
        pass
    
    client.create_collection(
        collection_name="moroccan_law",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    print("🔄 INGESTION AVEC MÉTADONNÉES")
    print("=" * 60)
    
    data_folder = "E:/moroccan-law-rag-v1/data/cleaned"
    point_id = 0
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_folder, filename)
            print(f"\n📖 {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Créer des chunks sémantiques
            chunks = semantic_chunk_text(text)
            print(f"   📦 Chunks sémantiques: {len(chunks)}")
            
            # Traiter chaque chunk
            for i, chunk in enumerate(chunks):
                # Extraire les métadonnées
                metadata = extract_metadata(chunk)
                
                # Créer l'embedding
                embedding = embedder.encode([chunk])[0].tolist()
                
                # Préparer le point
                point = {
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "filename": filename,
                        "chunk_id": i,
                        "chunk_count": len(chunks),
                        **metadata, 
                        "is_legal_document": any([
                            metadata['has_articles'],
                            metadata['has_decret'],
                            metadata['has_arrete'],
                            metadata['has_loi']
                        ])
                    }
                }
                client.upsert(collection_name="moroccan_law", points=[point])
                point_id += 1
            
            print(f"   ✅ {len(chunks)} chunks uploadés")
    
    collection_info = client.get_collection("moroccan_law")
    print(f"\n🎯 INGESTION TERMINÉE!")
    print(f"📊 Points totaux: {collection_info.points_count}")

if __name__ == "__main__":
    ingest_with_metadata()