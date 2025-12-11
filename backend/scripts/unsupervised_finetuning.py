# backend/scripts/unsupervised_finetuning.py - VERSION CORRIGÉE
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import os
import re
from typing import List
import json
import warnings
warnings.filterwarnings('ignore') 

class LegalTextProcessor:
    """Prépare vos textes juridiques pour le fine-tuning"""
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
        
    def load_legal_texts(self, data_folder: str = "E:/moroccan-law-rag-v1/data/cleaned") -> List[str]:
        """Charge tous vos textes juridiques"""
        texts = []
        
        for filename in os.listdir(data_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Nettoyer et diviser en chunks
                    cleaned_texts = self._clean_and_chunk_text(content)
                    texts.extend(cleaned_texts)
        
        print(f"✅ {len(texts)} chunks de texte chargés")
        return texts
    
    def _clean_and_chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Nettoie et divise le texte en chunks pour l'entraînement"""
        # Nettoyer le texte
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Diviser par articles pour garder la structure
        chunks = []
        
        # Diviser par articles
        articles = re.split(r'(?=Article\s+\d+)', text)
        
        for article in articles:
            article = article.strip()
            if not article:
                continue
            
            # Si l'article est trop long, le diviser
            if len(article) > max_length * 3:
                # Diviser en paragraphes
                paragraphs = re.split(r'(?:\n\s*){2,}', article)
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) < max_length:
                        current_chunk += para + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(article)
        
        return chunks
    
    def create_t5_pretraining_examples(self, texts: List[str]) -> Dataset:
        """Crée des exemples pour le pré-entraînement style T5"""
        
        examples = []
        
        for text in texts:
            # 1. Tâche: Compléter le texte (text infilling)
            if len(text) > 100:
                # Masquer une partie du texte
                words = text.split()
                if len(words) > 20:
                    # Masquer 15% des mots
                    mask_count = max(3, len(words) // 7)
                    
                    # Créer plusieurs exemples avec différentes parties masquées
                    for _ in range(2):
                        masked_text, target = self._create_masked_example(text, mask_count)
                        
                        examples.append({
                            "input_text": f"remplir les mots manquants: {masked_text}",
                            "target_text": target
                        })
            
            # 2. Tâche: Résumer 
            if len(text) > 200:
                summary = self._create_summary_prompt(text)
                examples.append({
                    "input_text": f"résumer le texte juridique: {text[:300]}...",
                    "target_text": summary[:150]
                })
            
            # 3. Tâche: Identifier les articles
            article_match = re.search(r'Article\s+(\d+)', text)
            if article_match:
                article_num = article_match.group(1)
                examples.append({
                    "input_text": f"identifier l'article: {text[:200]}",
                    "target_text": f"Article {article_num}"
                })
        
        print(f"🎯 {len(examples)} exemples créés pour le pré-entraînement")
        return Dataset.from_list(examples)
    
    def _create_masked_example(self, text: str, mask_count: int = 5) -> tuple:
        """Crée un exemple avec des mots masqués"""
        words = text.split()
        
        if len(words) <= mask_count * 2:
            return text, text
        
        # Choisir des positions aléatoires à masquer
        import random
        mask_positions = random.sample(range(len(words)), min(mask_count, len(words)//3))
        mask_positions.sort()
        
        # Créer le texte masqué et la cible
        masked_words = []
        target_words = []
        
        for i, word in enumerate(words):
            if i in mask_positions:
                masked_words.append("<extra_id_0>")
                target_words.append(f"<extra_id_0>{word}")
            else:
                masked_words.append(word)
        
        # Ajouter le dernier token spécial
        target_words.append("<extra_id_1>")
        
        masked_text = " ".join(masked_words)
        target_text = " ".join(target_words)
        
        return masked_text, target_text
    
    def _create_summary_prompt(self, text: str) -> str:
        # Extraire les phrases clés
        sentences = re.split(r'[.!?]', text)
        key_sentences = []
        
        keywords = ['délai', 'procédure', 'notification', 'saisine', 'collège', 'AMMC']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords) and len(sentence) > 20:
                key_sentences.append(sentence)
                if len(key_sentences) >= 3:
                    break
        
        if not key_sentences:
            # Prendre les premières phrases
            key_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        
        return ". ".join(key_sentences) + "."

def prepare_unsupervised_dataset():
    """Prépare le dataset pour le fine-tuning non supervisé"""
    processor = LegalTextProcessor()
    
    # 1. Charger les textes
    print("📖 Chargement des textes juridiques...")
    texts = processor.load_legal_texts()
    
    if not texts:
        print("❌ Aucun texte trouvé dans data/cleaned/")
        return None
    
    # 2. Créer les exemples d'entraînement
    print("🔧 Création des exemples d'entraînement...")
    dataset = processor.create_t5_pretraining_examples(texts)
    
    # 3. Tokenizer les données
    print("🔡 Tokenization des données...")
    
    def tokenize_function(examples):
        # Tokenizer les inputs
        model_inputs = processor.tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenizer les targets avec text_target
        labels = processor.tokenizer(
            text_target=examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def train_unsupervised():
    """Exécute le fine-tuning non supervisé"""
    
    print("🚀 DÉBUT DU FINE-TUNING NON SUPERVISÉ")
    print("=" * 50)
    
    # 1. Préparer les données
    tokenized_dataset = prepare_unsupervised_dataset()
    
    if tokenized_dataset is None:
        return
    
    # 2. Charger le modèle et tokenizer
    print("🔄 Chargement du modèle T5...")
    model_name = "google/flan-t5-base"  
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 3. Configurer l'entraînement (VERSION CORRIGÉE)
    training_args = TrainingArguments(
        output_dir="./models/t5-legal-unsupervised",
        overwrite_output_dir=True,
        num_train_epochs=5,  
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps", 
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  
        no_cuda=not torch.cuda.is_available(),  
    )
    
    # 4. Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(10, len(tokenized_dataset)))),
        tokenizer=tokenizer,
    )
    
    # 5. Entraîner
    print("🏋️  Début de l'entraînement...")
    trainer.train()
    
    # 6. Sauvegarder le modèle
    print("💾 Sauvegarde du modèle fine-tuné...")
    model.save_pretrained("./models/t5-legal-marocain")
    tokenizer.save_pretrained("./models/t5-legal-marocain")
    
    print("✅ Fine-tuning non supervisé terminé avec succès!")
    print(f"📁 Modèle sauvegardé dans: ./models/t5-legal-marocain")
    
    return model, tokenizer

def test_finetuned_model():
    
    print("\n🧪 TEST DU MODÈLE FINE-TUNÉ")
    print("=" * 50)
    
    # Charger le modèle fine-tuné
    model_path = "./models/t5-legal-marocain"
    
    if not os.path.exists(model_path):
        print("❌ Modèle fine-tuné non trouvé. Exécutez d'abord l'entraînement.")
        return
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Exemples de test
    test_cases = [
        ("question juridique: Quel est le délai selon l'article 51 ?", "Délai"),
        ("question juridique: Comment notifier les griefs ?", "Notification"),
        ("récapituler le texte: Article 51. Le collège dispose d'un délai", "Résumé"),
    ]
    
    for prompt, test_type in test_cases:
        print(f"\n🔍 Test: {test_type}")
        print(f"📝 Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            min_length=10,
            num_beams=2,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Réponse: {response}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning non supervisé pour T5")
    parser.add_argument("--train", action="store_true", help="Exécuter l'entraînement")
    parser.add_argument("--test", action="store_true", help="Tester le modèle fine-tuné")
    parser.add_argument("--light", action="store_true", help="Utiliser le fine-tuning léger")
    
    args = parser.parse_args()
    
    if args.light:
        print("⚡ MODE LÉGER ACTIVÉ")
        exec(open("lightweight_finetuning.py").read())
    elif args.train:
        train_unsupervised()
    elif args.test:
        test_finetuned_model()
    else:
        print("⚠️  Options disponibles:")
        print("  --train    : Fine-tuning complet")
        print("  --test     : Tester le modèle fine-tuné")
        print("  --light    : Fine-tuning léger (recommandé pour débuter)")