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


def store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy):
    """
    Slaat de chunks op in de gekozen database
    
    Parameters:
        chunks (list): list of chunks
        storageStrategy (String): de strategie om de chunks op te slaan.
        embeddingStrategy (String): de strategie om de chunks te vertalen naar vectors.
        leidraad (String): de naam van de lijdraad
        chunkingStrategy (String): Strategie voor het chunken, alleen bedoelt voor de benaming van de database

    Returns:
        collection: Returs naam van de huidige collectie
    """

    # Maak naam van huidige leidraad + chunk strategie
    leidraad = leidraad+"_"+chunking_strategy

    # Maakt Qdrant storage aan als deze nog niet bestaan
    if storageStrategy == "Qdrant":
        try:
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

    # Maakt ChromaDB storage aan als deze nog niet bestaan
    if storageStrategy == "ChromaDB":
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if leidraad in existing_collections:
            leidraad = chroma_client.get_collection(leidraad)
            print("ChromaDB: return bestaande")
            return leidraad
        else:
            print("ChromaDB: Nieuwe collectie maken")
            leidraad = chroma_client.create_collection(leidraad)

    # Als de storage aangemaakt moest worden wordt hier het embedden en toevoegen van de chunks gedaan
    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx}")
        
        # Generate embedding
        embedding = openAI_client.embeddings.create(
            input=chunk,
            model=embeddingStrategy
        )

        # Controleer de embedding op de lengte
        embedding_vector = embedding.data[0].embedding
        if len(embedding_vector) != 1536:
            print(f"Error: Embedding for chunk {idx} has invalid dimensions: {len(embedding_vector)}")
            continue

        # Voeg aan storage toe
        if storageStrategy == "ChromaDB":
            print(f"Chunk {idx} added to ChromaDB storage")
            leidraad.add(
                ids=[str(idx)],
                documents=[chunk],
                embeddings=[embedding_vector]
            )
        elif storageStrategy == "Qdrant":
            print(f"Chunk {idx} added to Qdrant storage")
            qdrant_client.upsert(
                collection_name=leidraad,
                points=[
                    models.PointStruct(
                        id=str(uuid4()),  # Unieke ID voor de string
                        payload={"document": chunk},  # Payload = document, willen we later uitbreiden met metadata
                        vector=embedding_vector  # vector
                    )
                ]
            )
            print(f"Chunk {idx} added to collection '{leidraad}'.")
    # Returt de leidraad als vectors
    print("Embedding and storage process completed.")
    return leidraad



def get_chunks(collection, user_query, storageStrategy):
    """
    Zoekt de meest relevante chunks op basis van vector similarity search
    
    Parameters:
        collection (String): naam van de leidraad
        user_query (String): query van de gebruiker
        storageStrategy (String): strategie om de opslag te doen, om te bepalen in welke database hij op zoek gaat naar de chunks
    
    Returns:
        collection: Returns top chunks op basis van de query gegeven door de gebruiker
    """
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
        limit=2,  # Number of results to return
        with_payload=True  # Include payload (e.g., original chunk text)
        )
        if results:
            return results  # Top relevant chunk
        else:
            return None  # No results found

