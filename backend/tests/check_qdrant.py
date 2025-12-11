from qdrant_client import QdrantClient

def check_stored_data():
    client = QdrantClient(url="http://localhost:6333")
    
    
    collection_info = client.get_collection("moroccan_law")
    print(f"📊 Points totaux: {collection_info.points_count}")
    
    
    points = client.scroll(
        collection_name="moroccan_law",
        limit=5,
        with_payload=True
    )[0]
    
    print("\n📝 EXEMPLES DE POINTS STOCKÉS:")
    for i, point in enumerate(points):
        print(f"\n--- Point {i+1} ---")
        print(f"ID: {point.id}")
        payload = point.payload
        print(f"Fichier: {payload.get('filename', 'N/A')}")
        print(f"Chunk: {payload.get('chunk_id', 'N/A')}/{payload.get('chunk_count', 'N/A')}")
        print(f"Texte ({len(payload.get('text', ''))} chars):")
        print(payload.get('text', '')[:200] + "...")
        
        
        if payload.get('article_numbers'):
            print(f"Articles: {payload.get('article_numbers')}")

if __name__ == "__main__":
    check_stored_data()