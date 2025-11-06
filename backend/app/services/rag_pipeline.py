from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import textwrap

class RAGPipeline:
    def __init__(self, qdrant_url="http://localhost:6333", collection_name="moroccan_law"):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1  # CPU
        )

    def query(self, question, top_k=5, chunk_size=1000):
        # Embed the question
        q_vec = self.embedder.encode([question])[0]

        # Retrieve top-k chunks from Qdrant
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            limit=top_k
        )

        # Split context if chunks are too long
        context_chunks = []
        for hit in hits:
            text = hit.payload["text"]
            # Split text into smaller pieces to avoid exceeding model limits
            pieces = textwrap.wrap(text, chunk_size)
            context_chunks.extend(pieces)

        # Generate answer per chunk in French
        answers = []
        for i, chunk in enumerate(context_chunks):
            prompt = (
                f"Réponds à la question suivante en français en utilisant le contexte ci-dessous :\n\n"
                f"Contexte : {chunk}\n\nQuestion : {question}\nRéponse :"
            )
            out = self.generator(prompt, max_new_tokens=256)[0]['generated_text']
            answers.append(out.strip())

        # Merge all chunk answers
        final_answer = " ".join(answers)
        return final_answer


# test
if __name__ == "__main__":
    rag = RAGPipeline()
    while True:
        question = input("Entrez votre question (ou 'exit' pour quitter) : ")
        if question.lower() == "exit":
            break
        answer = rag.query(question)
        print("Réponse:", answer)
