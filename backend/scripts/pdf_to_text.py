# backend/scripts/pdf_to_text.py - VERSION AVEC PDFPLUMBER
import os
import re
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extrait le texte avec pdfplumber"""
    print(f"📄 Extraction de {os.path.basename(pdf_path)} avec pdfplumber...")
    
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # Extraire le texte
                page_text = page.extract_text()
                
                if page_text:
                    # Nettoyage de base
                    cleaned = clean_page_text(page_text, page_num)
                    if cleaned.strip():
                        full_text += cleaned + "\n\n"
                        
            except Exception as e:
                print(f"   ⚠️  Page {page_num} erreur: {e}")
    
    return full_text

def clean_page_text(text, page_num):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
    
        if is_garbage(line):
            continue
        
        if is_legal_content(line):
            
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'\.{3,}', ' ', line)
            cleaned_lines.append(line)
    
    return " ".join(cleaned_lines)

def is_garbage(line):
    
    garbage_patterns = [
        r'\d{2}\.\d{2}\.\d{2}\.\d{2}',
        r'\d{2}\s*\.\s*\d{2}\s*\.\s*\d{2}\s*\.\s*\d{2}',
        r'Tél\.?\s*:',
        
        # Prix, DH, euros
        r'\d+\s*DH',
        r'\d+\.?\d*\s*€',
        r'\d+\.?\d*\s*\$',
        
        # Numéros de page
        r'^\s*\d+\s*$',
        r'Page\s+\d+',
        
        # Adresses, emails
        r'@',
        r'\.com',
        r'\.org',
        r'\.net',
        
        # BULLETIN OFFICIEL noise
        r'BULLETIN\s+OFFICIEL',
        r'Nº\s*\d+',
        r'rabii I\s+\d+',
        r'\d{4}[-\s]\d{1,2}[-\s]\d{1,2}',
        
        # Répétitions
        r'CHARIOT RISSANA LIMITED',
        r'Atlantic Free Zone Investment',
        
        # Fragments courts sans sens
        r'^\s*\.+\s*$',
        r'^\s*–+\s*$',
        r'^\s*\*\s*$',
    ]
    
    for pattern in garbage_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    if len(line) < 20 and not any(keyword in line for keyword in ['Article', 'Décret', 'Arrêté', 'Loi']):
        return True
    
    return False

def is_legal_content(line):
    legal_keywords = [
        r'Article\s+\d+',
        r'Décret\s+n°\s*\d',
        r'Arrêté\s+du',
        r'Loi\s+n°\s*\d',
        r'Vu\s+le',
        r'Vu\s+la',
        r'Conformément\s+aux',
        r'Au\s+termes\s+de',
        r'Il\s+est\s+prévu',
        r'dispose\s+que',
        r'prévu\s+à',
        r'alinéa',
        r'paragraphe',
        r'TITRE',
        r'CHAPITRE',
        r'SECTION',
    ]
    
    for pattern in legal_keywords:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    # Ou être une phrase complète avec une structure juridique
    if '.' in line and len(line) > 50 and any(char.isupper() for char in line[:10]):
        words = line.split()
        if 5 <= len(words) <= 100:
            return True
    
    return False

def post_process_text(text):
    """Post-traitement pour corriger les problèmes communs"""
    # 1. Réparer les articles coupés
    text = re.sub(r'«\s*Article', 'Article', text)
    text = re.sub(r'«\s*', ' ', text)  # Supprimer les guillemets ouvrants isolés
    
    # 2. Réparer les fins de ligne coupée
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Mots coupés avec tiret
    
    # 3. Supprimer les numéros de ligne
    text = re.sub(r'\s*\d{3,4}\s*$', '', text, flags=re.MULTILINE)
    
    # 4. Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def process_pdf_file(pdf_path, output_folder):
    """Traite un fichier PDF et sauvegarde le texte nettoyé"""
    try:
        # Extraction
        raw_text = extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            print(f"   ⚠️  Aucun texte extrait!")
            return False
        
        # Post-traitement
        cleaned_text = post_process_text(raw_text)
        
        # Vérification de qualité
        if len(cleaned_text) < 500:
            print(f"   ⚠️  Texte trop court ({len(cleaned_text)} chars)")
            return False
        
        # Sauvegarde
        output_filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Statistiques
        word_count = len(cleaned_text.split())
        print(f"   ✅ Sauvegardé: {len(cleaned_text)} caractères, {word_count} mots")
        
        # Afficher un extrait
        print(f"   📝 Extrait:")
        lines = cleaned_text.split('\n')
        for line in lines[:3]:
            if line.strip():
                print(f"      {line[:100]}..." if len(line) > 100 else f"      {line}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    input_folder = "data/raw"
    output_folder = "data/cleaned"
    os.makedirs(output_folder, exist_ok=True)
    
    print("🔄 CONVERSION PDF → TEXTE (AVEC PDFPLUMBER)")
    print("=" * 60)
    
    success_count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"\n📖 {filename}")
            
            if process_pdf_file(pdf_path, output_folder):
                success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"🎉 TERMINÉ! {success_count} fichiers convertis avec succès")

if __name__ == "__main__":
    main()