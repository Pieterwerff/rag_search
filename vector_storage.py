import chromadb
from qdrant_client.http import models
from qdrant_client import QdrantClient
from openai import OpenAI

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

openAI_client = OpenAI()

# Initialize QdrantClient

qdrant_api_key = os.getenv('Qdrant_KEY')

qdrant_client = QdrantClient(
    url="https://7c489c6d-3ee2-4145-89ef-6f315377430f.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api_key,
)

from uuid import uuid4

chroma_client = chromadb.PersistentClient(path="./databases")


def store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad):
    """
    Stores the chunks in chosen vector storage
    
    Parameters:
        chunks (list): List of document chunks
        storageStrategy (String): the strategy to store the chunks.
    
    Returns:
        collection: Collection of chunks stored
    """
    if storageStrategy == "Qdrant":
        try:
            qdrant_client.delete_collection(collection_name=leidraad)
            qdrant_client.get_collection(collection_name=leidraad)
            print(f"Qdrant: Collection '{leidraad}' already exists.")
            return leidraad
        except Exception as e:
            # If it doesn't exist, create it
            print(f"Qdrant: Collection '{leidraad}' does not exist. Creating it...")
            qdrant_client.create_collection(
                collection_name=leidraad,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
            print(f"Qdrant: Collection '{leidraad}' created successfully.")

    # This runs for ChromaDB
    if storageStrategy == "ChromaDB":
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if leidraad in existing_collections:
            leidraad = chroma_client.get_collection(leidraad)
            print("ChromaDB: return bestaande")
            return leidraad
        else:
            print("ChromaDB: Nieuwe collectie maken")
            leidraad = chroma_client.create_collection(leidraad)

    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx}")
        
        # Generate embedding
        embedding = openAI_client.embeddings.create(
            input=chunk,
            model=embeddingStrategy
        )

        # Validate embedding dimensions
        embedding_vector = embedding.data[0].embedding
        if len(embedding_vector) != 1536:
            print(f"Error: Embedding for chunk {idx} has invalid dimensions: {len(embedding_vector)}")
            continue

        # Add to storage based on strategy
        if storageStrategy == "ChromaDB":
            print(f"Chunk {idx} added to ChromaDB storage")
            leidraad.add(
                ids=[str(idx)],
                documents=[chunk],
                embeddings=[embedding_vector]  # Store embeddings along with documents
            )
        elif storageStrategy == "Qdrant":
            print(f"Chunk {idx} added to Qdrant storage")
            qdrant_client.upsert(
                collection_name=leidraad,  # Ensure this is a string, not a collection object
                points=[
                    models.PointStruct(
                        id=str(uuid4()),  # Unique ID for this chunk
                        payload={"document": chunk},  # Storing the document
                        vector=embedding_vector  # Embedding vector
                    )
                ]
            )
            print(f"Chunk {idx} added to collection '{leidraad}'.")
        

    print("Embedding and storage process completed.")





def get_chunks(collection, user_query, storageStrategy):
    query_embedding = openAI_client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
    query_vector = query_embedding.data[0].embedding
    if storageStrategy == "ChromaDB":
        results = collection.query(
        query_embeddings=[query_vector],  # Use embeddings for search
        n_results=10
            )
        return results['documents'][0][0]  # Top relevant chunk from ChromaDB
    
    elif storageStrategy == "Qdrant":
        results = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,  # Embedding for similarity search
        limit=10,  # Number of results to return
        with_payload=True  # Include payload (e.g., original chunk text)
        )
        if results:
            return results  # Top relevant chunk
        else:
            return None  # No results found

