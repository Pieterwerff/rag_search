import chromadb
from qdrant_client import QdrantClient, models
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client
load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Initialize QdrantClient

qdrant_api_key = os.getenv('Qdrant_KEY')


qdrant_client = QdrantClient(
    url="https://7c489c6d-3ee2-4145-89ef-6f315377430f.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api_key,
)


chroma_client = chromadb.PersistentClient(path="./databases")


def store_chunks(chunks, storageStrategy, leidraad):
    """
    Stores the chunks in chosen vector storage
    
    Parameters:
        chunks (list): List of document chunks
        storageStrategy (String): the strategy to store the chunks.
    
    Returns:
        collection: Collection of chunks stored
    """
    qdrant_client.create_collection(
        collection_name="{leidraad}",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )


    existing_collections = [col.name for col in chroma_client.list_collections()]
    if leidraad in existing_collections:
        collection = chroma_client.get_collection(leidraad)
        print("return bestaande")
        return collection
    else:
        print("Nieuwe collectie maken")
        collection = chroma_client.create_collection(leidraad)
    
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
