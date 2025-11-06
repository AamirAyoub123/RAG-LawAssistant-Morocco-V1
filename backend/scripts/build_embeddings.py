from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer
import os

input_folder = "data/cleaned"
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Folder not found: {input_folder}")

# Connect to local Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")

# Collection name
COLLECTION_NAME = "moroccan_law"

# Delete old collection if exists
if COLLECTION_NAME in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

# Load model first to get embedding dimension
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()  # returns 384

# Create new collection with correct vector size
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=embedding_dim, distance="Cosine")
)
print(f"Created collection: {COLLECTION_NAME} with vector size {embedding_dim}")

# Upload embeddings
for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        vector = model.encode([text])[0]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": idx,
                "vector": vector,
                "payload": {"text": text, "filename": filename}
            }]
        )
        print(f"Uploaded {filename} as ID {idx}")
