# backend/app/services/embedding.py
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    return embedder.encode(chunks).tolist()


if __name__ == "__main__":
    sample_text = "Your long document text here..."
    chunks = chunk_text(sample_text)
    vectors = embed_chunks(chunks)
    print(f"Chunks: {len(chunks)}, Vectors: {len(vectors)}")
