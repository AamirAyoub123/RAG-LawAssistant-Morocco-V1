from qdrant_client import QdrantClient
from backend.app.services.load_qdrant import COLLECTION_NAME

class QdrantRetriever:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = COLLECTION_NAME

    def search(self, query_embedding, top_k=5):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,  
            limit=top_k
        )
        hits = response.points
        return [{"text": hit.payload.get("text", "")} for hit in hits]
