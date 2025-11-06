import os
from backend.app.services.retrieval import QdrantRetriever
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from backend.app.services.embedding import COLLECTION_NAME  # or define it here
retriever = QdrantRetriever(collection_name=COLLECTION_NAME)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

DATA_PATH = "data/raw/"

for idx, file_name in enumerate(os.listdir(DATA_PATH)):
    if file_name.lower().endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file_name)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        vector = embedder.encode(text).tolist()

        # Use integer ID
        retriever.client.upsert(
            collection_name=retriever.collection_name,
            points=[{
                "id": idx,  
                "vector": vector,
                "payload": {"text": text}
            }]
        )
        print(f"Ingested: {file_name} as ID {idx}")
