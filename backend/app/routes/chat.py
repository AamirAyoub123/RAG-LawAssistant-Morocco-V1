from fastapi import APIRouter
from pydantic import BaseModel
from backend.app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# Initialize pipeline (loads embedding model, Qdrant client, and generator)
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

@router.post("/")
def ask_question(request: QueryRequest):
    answer = rag_pipeline.query(request.query)

    return {"query": request.query, "answer": answer}
