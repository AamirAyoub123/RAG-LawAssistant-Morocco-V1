from backend.app.services.rag_pipeline import RAGPipeline

def test_pipeline():
    pipeline = RAGPipeline()
    result = pipeline.run("What is the law about traffic violations?")
    assert isinstance(result, str)
    print("RAG pipeline test passed!")
