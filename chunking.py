def chunk_file(document, chunk_size=1000, overlap=200):
    """
    Splits a document into chunks of specified size with overlapping text.
    
    Parameters:
        document (str): The full document as a single string.
        chunk_size (int): Size of each chunk in characters.
        chunkingStrategy (String): the strategy to chunk the document.
    
    Returns:
        list: List of document chunks.
    """
    
    return [
        document[i:i + chunk_size]
        for i in range(0, len(document) - chunk_size + 1, chunk_size - overlap)
 
    ]
