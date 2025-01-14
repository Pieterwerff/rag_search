from chunking import chunk_file
from api_calls import query_llm
from vector_storage import store_chunks, get_chunks

# --- Settings ---
chunk_size = 1000
chunkingStrategy = '' # might become a list of strings to iterate through
llm = 'gpt' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
# --- Step 1: Load markdown Document ---
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()  # Read content into a string

# --- Step 2: Chunk Document using different types of chunking strategies ---

chunks = chunk_file(document, chunk_size=chunk_size, overlap=200)

# --- Step 3: Store Chunks in specific kind of chunk database ---

collection = store_chunks(chunks, chunkingStrategy, leidraad)

# --- Step 4: Handle User Query ---
user_query = input("Stel je vraag: ")

# --- step 5: Recieve relevant chunks
retrieved_text = get_chunks(collection, user_query)

# Query the LLM
response = query_llm(retrieved_text=retrieved_text, user_query=user_query, llm=llm)

# --- Step 6: Output Result ---
print("Vraag:", user_query)
print("Antwoord:", response)

# --- Step 7: Validate Output ---
