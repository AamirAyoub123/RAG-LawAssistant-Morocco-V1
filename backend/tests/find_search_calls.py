
import os

def search_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '.search(' in line:
            print(f"{filepath}: Line {i+1}: {line.strip()}")
            
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(f"  {j+1}: {lines[j]}")


base_dir = "E:/moroccan-law-rag-v1"
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            search_in_file(filepath)