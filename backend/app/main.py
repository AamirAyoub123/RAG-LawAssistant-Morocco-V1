# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .services.rag_finetuned import FineTunedRAGPipeline

rag = FineTunedRAGPipeline(use_finetuned=True)

# Initialize FastAPI
app = FastAPI(title="Moroccan Law RAG Chatbot")


# Request model for the API
class QuestionRequest(BaseModel):
    question: str

# Response model 
class AnswerResponse(BaseModel):
    answer: str

# POST endpoint
@app.post("/ask_question_chat", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer = rag.query(request.question)  
    return {"answer": answer}



@app.get("/")
async def root():
    return {"message": "Backend is running!"}
