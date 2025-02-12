from chunking import chunk_file
from api_calls import query_llm
from vector_storage import store_chunks, get_chunks
from agent import language_agent
from agentic_chunker import chunk_file_with_metadata 
# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph' # might become a list of strings to iterate through
llm = 'gpt4o-mini' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"

# --- Step: Load markdown Document ---
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()  # Read content into a string

# --- Step: Load markdown Document ---

# language = language_agent(input("Stel je vraag: "))
# print(language)
user_query = input("Stel je vraag: ")

# --- Step: Chunk Document using different types of chunking strategies ---

chunks = chunk_file(document, chunk_size=chunk_size, chunking_strategy=chunking_strategy)

# --- Step: Store Chunks in specific kind of chunk database ---

collection = store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy)

# --- step: Recieve relevant chunks
retrieved_text = get_chunks(collection, user_query, storageStrategy)

# Query the LLM
response = query_llm(retrieved_text=retrieved_text, user_query=user_query, llm=llm)

# --- Step 6: Output Result ---
print("Vraag:", user_query)
print("Antwoord:", response)

# --- Step 7: Validate Output ---
# TODO