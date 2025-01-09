import chromadb

def store_chunks(chunks, storageStrategy):
    """
    Stores the chunks in chosen vector storage
    
    Parameters:
        chunks (list): List of document chunks
        storageStrategy (String): the strategy to store the chunks.
    
    Returns:
        collection: Collection of chunks stored
    """
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("leidraad_ai_in_zorg")

# Add chunks to ChromaDB
    for idx, chunk in enumerate(chunks):
        collection.add(
        ids=[str(idx)],
        documents=[chunk]
    )
        
    return collection

def get_chunks(collection, user_query):
    results = collection.query(
    query_texts=[user_query],
    n_results=10
    )
    return results['documents'][0][0]  # Top relevant chunk
