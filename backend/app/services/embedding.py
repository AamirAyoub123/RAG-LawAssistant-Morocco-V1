from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer
import os
import glob

# Initialize Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "moroccan_law"

# Create collection 
if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance="Cosine")
    )

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Ingest text files
folder = "data/cleaned"
for txt_file in glob.glob(f"{folder}/*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)] 
    vectors = embedder.encode(chunks).tolist()

    # Upload to Qdrant
    for i, vec in enumerate(vectors):
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": f"{txt_file}_{i}",
                    "vector": vec,
                    "payload": {"text": chunks[i]}
                }
            ]
        )
    print(f"Ingested: {txt_file}")
