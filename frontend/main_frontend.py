# fixed_frontend_v2.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import sys
import os
import uvicorn
from pydantic import BaseModel


backend_dir = Path("C:/moroccan-law-rag-v1/backend")

# Ajouter au sys.path
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

print(f"📁 Répertoire backend: {backend_dir}")
print(f"📁 Python path: {sys.path[:3]}")


import importlib.util

# Import rag_pipeline
rag_pipeline_path = backend_dir / "app" / "services" / "rag_pipeline.py"
spec = importlib.util.spec_from_file_location("rag_pipeline", rag_pipeline_path)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
RAGPipeline = rag_module.RAGPipeline

# Import rag_finetuned
rag_finetuned_path = backend_dir / "app" / "services" / "rag_finetuned.py"
spec = importlib.util.spec_from_file_location("rag_finetuned", rag_finetuned_path)
finetuned_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetuned_module)
FineTunedRAGPipeline = finetuned_module.FineTunedRAGPipeline

print("✅ Import alternatif réussi")

app = FastAPI(title="Moroccan Law RAG Chatbot - Comparaison")

# Path to frontend folder
FRONTEND_DIR = Path(__file__).parent

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")

# Initialize both RAG models
print("🔄 Initialisation des modèles RAG...")
original_rag = RAGPipeline()
finetuned_rag = FineTunedRAGPipeline(use_finetuned=True)
print("✅ Modèles RAG initialisés")

# Pydantic model for request
class CompareRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/compare")
async def api_compare(request: CompareRequest):
    """API endpoint for comparison (for AJAX calls)"""
    import time
    
    print(f"\n📝 Question reçue: {request.question}")
    
    try:
        # Get original model response
        original_start = time.time()
        print("🔄 Original model processing...")
        original_response = original_rag.query(request.question, debug=False)
        original_time = (time.time() - original_start) * 1000  # ms
        print(f"✅ Original model: {original_time:.0f}ms")
        
        # Get finetuned model response
        finetuned_start = time.time()
        print("🔄 Fine-tuned model processing...")
        finetuned_response = finetuned_rag.query(request.question, debug=False)
        finetuned_time = (time.time() - finetuned_start) * 1000  # ms
        print(f"✅ Fine-tuned model: {finetuned_time:.0f}ms")
        
        response_data = {
            "question": request.question,
            "original": {
                "response": original_response,
                "time_ms": round(original_time, 2)
            },
            "finetuned": {
                "response": finetuned_response,
                "time_ms": round(finetuned_time, 2)
            },
            "comparison": {
                "time_difference": round(finetuned_time - original_time, 2),
                "is_finetuned_faster": finetuned_time < original_time,
                "original_length": len(original_response),
                "finetuned_length": len(finetuned_response)
            }
        }
        
        print(f"📊 Response sent successfully")
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"❌ Error in api_compare: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "service": "frontend"})

if __name__ == "__main__":
    print("🚀 Serveur frontend démarré sur http://localhost:8001")
    print("📡 Backend: http://localhost:8000")
    print("📊 Modèles disponibles:")
    print("   • Modèle Original: FLAN-T5-large")
    print("   • Modèle Fine-tuné: FLAN-T5 fine-tuné sur droit marocain")
    uvicorn.run(app, host="0.0.0.0", port=8001)