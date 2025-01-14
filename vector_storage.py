import chromadb
from qdrant_client import QdrantClient
from openai import OpenAI
client = OpenAI()

qdrant_client = QdrantClient(
    url="https://7c489c6d-3ee2-4145-89ef-6f315377430f.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="NxBdPnPlBWCvzhxHoUdQwp2m0RYOU12I0TuJY9ypibEEieHEZOS1tQ",
)

print(qdrant_client.get_collections())
def store_chunks(chunks, storageStrategy):
    """
    Stores the chunks in chosen vector storage
    
    Parameters:
        chunks (list): List of document chunks
        storageStrategy (String): the strategy to store the chunks.
    
    Returns:
        collection: Collection of chunks stored
    """
    chroma_client = chromadb.PersistentClient(path="./databases")
    collection = chroma_client.create_collection("leidraad_ai_in_zorg")

# Add chunks to ChromaDB
    for idx, chunk in enumerate(chunks):
        embedding = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        # print(embedding)
        collection.add(
            ids=[str(idx)],
            documents=[chunk],
            embeddings=[embedding.data[0].embedding]  # Store embeddings along with documents
        )
    return collection

def get_chunks(collection, user_query):
    query_embedding = client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
    results = collection.query(
    query_embeddings=[query_embedding.data[0].embedding],  # Use embeddings for search
    n_results=10
)

    return results['documents'][0][0]  # Top relevant chunk
