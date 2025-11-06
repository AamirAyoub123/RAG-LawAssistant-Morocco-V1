# backend/app/services/load_qdrant.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Name of the collection
collection_name = "moroccan_law"

# Create the collection if it doesn't exist
if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vector_size=384,  # all-MiniLM-L6-v2 output dimension
        distance="Cosine"
    )
    print(f"Collection '{collection_name}' created!")

# Load your documents
documents_folder = "backend/app/data"  # replace with your folder path
embedder = SentenceTransformer("all-MiniLM-L6-v2")

for filename in os.listdir(documents_folder):
    if filename.endswith(".txt"):  # or ".pdf" if you preprocessed PDFs
        with open(os.path.join(documents_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        embedding = embedder.encode([text])[0]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": filename,
                    "vector": embedding.tolist(),
                    "payload": {"text": text}
                }
            ]
        )
        print(f"Uploaded {filename}")
