from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from typing import List, Dict, Optional, Tuple

class RAGPipeline:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.model_name = "google/flan-t5-large"
        print("🔄 Chargement du modèle FLAN-T5...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        print("✅ RAG Pipeline initialisé")
    
    def _extract_article_info(self, question: str) -> Tuple[Optional[str], List[str]]:
        """Extrait le numéro d'article et les mots-clés de la question"""
        article_num = None
        patterns = [
            r'article\s+(\d+)',
            r'article\s+n°\s*(\d+)',
            r'Article\s+(\d+)',
            r'art\.\s*(\d+)',
            r'l\'article\s+(\d+)',
            r'art\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                article_num = match.group(1)
                break
        
        # Extraire les mots-clés
        stop_words = {
            'quel', 'quelle', 'quels', 'quelles', 'est', 'sont', 'était', 'étaient',
            'dans', 'pour', 'avec', 'selon', 'l\'article', 'article', 'art', 'n°',
            'du', 'de', 'la', 'le', 'les', 'un', 'une', 'des', 'et', 'ou', 'à', 'au',
            'aux', 'en', 'par', 'sur', 'sous', 'chez', 'dont', 'que', 'qui', 'quoi',
            'comment', 'pourquoi', 'quand', 'où', 'combien', 'quelle', 'quel'
        }
        
        question_lower = question.lower()
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return article_num, keywords
    
    def _calculate_relevance_score(self, hit, text: str, article_num: Optional[str], keywords: List[str]) -> float:
        """Calcule un score de pertinence personnalisé"""
        score = hit.score
        
        if article_num:
            article_patterns = [
                f"Article {article_num}",
                f"article {article_num}",
                f"Article {article_num}.",
                f"Art. {article_num}",
                f"Art {article_num}",
            ]
            if any(pattern in text for pattern in article_patterns):
                score += 0.3
        
        keyword_bonus = sum(0.05 for keyword in keywords if keyword in text.lower())
        score += keyword_bonus
        
        legal_terms = ['délai', 'notification', 'saisine', 'procédure', 'sanction', 
                      'collège', 'AMMC', 'griefs', 'audition', 'instruction']
        legal_bonus = sum(0.02 for term in legal_terms if term in text.lower())
        score += legal_bonus
        
        return score
    
    def _create_context_prompt(self, question: str, context: str, article_num: Optional[str]) -> str:
        
        # Identifier le type de question
        if "délai" in question.lower() or "durée" in question.lower():
            instruction = "Donne uniquement la durée ou le délai mentionné."
        elif "comment" in question.lower():
            instruction = "Explique brièvement la procédure en 1-2 phrases."
        elif "qui" in question.lower():
            instruction = "Identifie la personne ou l'entité responsable."
        else:
            instruction = "Réponds de manière concise et précise."
        
        if article_num:
            return f"""Question sur l'article {article_num}: {question}

    Contexte juridique: {context}

    {instruction}
    Réponds en français en utilisant uniquement les informations du contexte.

    Réponse:"""
        else:
            return f"""Question: {question}

    Contexte juridique: {context}

    {instruction}
    Réponds en français en utilisant uniquement les informations du contexte.

    Réponse:"""
    
    def _clean_and_format_response(self, response: str, question: str) -> str:
        response = response.strip()
        response = re.sub(r'Article\s+\d+\.?\s*–.*?(?=\.|$)', '', response)
        if not response or len(response) < 10:
            article_num, _ = self._extract_article_info(question)
            if article_num:
                return f"Selon l'Article {article_num}, cette information n'est pas spécifiée."
            return "Information non disponible."
        
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        corrections = {
            'and': 'et',
            'of': 'de',
            'his': 'son',
            'the': '',
            'Article': 'Article',
            'articles': 'articles',
            'à sa défense': 'de sa défense',
            'his droit': 'son droit',
        }
        
        for wrong, correct in corrections.items():
            response = response.replace(wrong, correct)
        
        return response
    
    def query(self, question: str, debug: bool = False) -> str:
        try:
            if debug:
                print(f"\n🔍 QUESTION: {question}")
            
            # 1. Analyser la question
            article_num, keywords = self._extract_article_info(question)
            
            if debug:
                print(f"📊 Analyse: Article={article_num}, Mots-clés={keywords}")
            
            # 2. Recherche vectorielle
            q_vec = self.embedder.encode([question])[0]
            response = self.client.query_points(
                collection_name="moroccan_law",
                query=q_vec,  
                limit=12,
                with_payload=True
            )
            hits = response.points  
            
            if debug:
                print(f"📈 {len(hits)} documents trouvés")
            
            if not hits:
                return "Aucun document pertinent trouvé dans la base de données."
            
            # 3. Sélectionner les documents les plus pertinents
            scored_hits = []
            for hit in hits:
                text = hit.payload.get("text", "")
                relevance_score = self._calculate_relevance_score(hit, text, article_num, keywords)
                
                scored_hits.append({
                    'hit': hit,
                    'text': text,
                    'original_score': hit.score,
                    'relevance_score': relevance_score,
                    'has_article': article_num and any(
                        pattern in text 
                        for pattern in [f"Article {article_num}", f"article {article_num}"]
                    )
                })
            
            
            scored_hits.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            if debug and len(scored_hits) > 0:
                print(f"\n🎯 TOP 3 DOCUMENTS:")
                for i, sh in enumerate(scored_hits[:3]):
                    article_status = "✅ CONTIENT L'ARTICLE" if sh['has_article'] else ""
                    print(f"{i+1}. Pertinence: {sh['relevance_score']:.4f} {article_status}")
                    print(f"   {sh['text'][:80]}...")
            
            # 4. Construire le contexte
            context_parts = []
            for sh in scored_hits[:3]:  
                context_parts.append(sh['text'])
            
            context = " ".join(context_parts)
            context = re.sub(r'\s+', ' ', context)[:1500]
            
            if debug:
                print(f"\n📝 CONTEXTE UTILISÉ ({len(context)} caractères):")
                print(f"{context[:300]}...")
            
            # 5. Générer la réponse avec prompt simplifié
            prompt = self._create_context_prompt(question, context, article_num)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True,
                padding=True
            )
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=350,
                min_length=30,
                do_sample=False,
                num_beams=5,
                temperature=0.2,
                repetition_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                length_penalty=1.0,
            )
            
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 6. Nettoyer la réponse
            final_response = self._clean_and_format_response(raw_response, question)
            
            return final_response
            
        except Exception as e:
            error_msg = f"Erreur technique: {str(e)[:100]}"
            if debug:
                print(f"❌ {error_msg}")
            return error_msg

# Exemple de test
if __name__ == "__main__":
    
    rag = RAGPipeline()

    test_questions = [
        "Quel est le délai selon l'article 51 ?",
        "Comment notifier les griefs selon l'article 52 ?",
        "Que prévoit l'article 49 pour la saisine du collège ?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        response = rag.query(question, debug=True)
        print(f"✅ RÉPONSE: {response}")
        