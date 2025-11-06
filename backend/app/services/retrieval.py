from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.app.services.embedding import COLLECTION_NAME  # if you have it there


class QdrantRetriever:
    def __init__(self, collection_name="law_documents"):
        # Connect to local Qdrant
        self.client = QdrantClient(url="http://localhost:6333")
        
        self.collection_name = COLLECTION_NAME

        # Create collection if not exists
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

    def search(self, query_embedding, top_k=3):
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        # Return list of dicts with text
        return [{"text": hit.payload.get("text", "")} for hit in hits]
