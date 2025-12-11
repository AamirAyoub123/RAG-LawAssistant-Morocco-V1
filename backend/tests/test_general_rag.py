# backend/tests/test_general_rag.py
from backend.app.services.rag_pipeline import SmartRAGPipeline

def test_general_queries():
    """Test le RAG avec divers types de questions"""
    rag = SmartRAGPipeline()
    
    test_cases = [
        {
            "category": "Questions spécifiques d'article",
            "questions": [
                "Quel est le délai selon l'article 51 ?",
                "Comment notifier les griefs selon l'article 52 ?",
                "Que prévoit l'article 49 pour la saisine ?",
                "Quelles sont les règles procédurales de l'article 53 ?"
            ]
        },
        {
            "category": "Questions générales",
            "questions": [
                "Qui est chargé du secrétariat du collège ?",
                "Quelles sont les modalités de saisine ?",
                "Comment se déroule une audition ?",
                "Quels sont les droits de la partie mise en cause ?"
            ]
        },
        {
            "category": "Questions avec mots-clés",
            "questions": [
                "délai de traitement",
                "notification des griefs",
                "procédure de sanction",
                "réunions du collège"
            ]
        }
    ]
    
    print("🧪 TEST COMPLET DU RAG GÉNÉRAL")
    print("=" * 70)
    
    for test_case in test_cases:
        print(f"\n📚 CATÉGORIE: {test_case['category']}")
        print("-" * 70)
        
        for question in test_case['questions']:
            print(f"\n❓ {question}")
            answer = rag.query(question)
            print(f"✅ {answer}")
            print()

if __name__ == "__main__":
    test_general_queries()