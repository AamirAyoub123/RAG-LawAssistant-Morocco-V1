import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += " " + page_text.replace("\n", " ")
    return text

def save_txt(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    input_folder = "data/raw"
    output_folder = "data/cleaned"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            text = pdf_to_text(pdf_path)
            output_path = os.path.join(output_folder, file.replace(".pdf", ".txt"))
            save_txt(text, output_path)
            print(f"Saved: {output_path}")
