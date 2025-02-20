from preprocessing.chunking import chunk_file
from llm_calls.api_calls import query_llm
from vector_storage import store_chunks, get_chunks
from agents.language_agent import language_agent
from preprocessing.agentic_chunker import chunk_file_with_metadata 
import pandas as pd

# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph' # might become a list of strings to iterate through
llm = 'gpt-4o-mini' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"

# --- Step: Laad leidraad uit csv in pandas dataframe ---
document_df = pd.read_csv(r'brondocumenten\hoofstukken.csv')

# language = language_agent(input("Stel je vraag: "))
# print(language)
user_query = input("Stel je vraag: ")

# --- Step: Chunk Document using different types of chunking strategies ---

chunks = chunk_file(document_df, chunk_size=chunk_size, chunking_strategy=chunking_strategy)

# --- Step: Store Chunks in specific kind of chunk database ---

collection = store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy)

# --- step: Recieve relevant chunks
retrieved_text = get_chunks(collection, user_query, storageStrategy)

# Query the LLM
response = query_llm(retrieved_text, user_query, llm)

# --- Step 6: Output Result ---
print("Vraag:", user_query)
print("Antwoord:", response)
