# Voeg de bovenliggende map toe aan sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.chunking import chunk_file
from llm_calls.api_calls import query_llm
from vector_storage import store_chunks, get_chunks
from agents.language_agent import language_agent
from preprocessing.agentic_chunker import chunk_file_with_metadata 
import pandas as pd

# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph'

# Lijst met mogelijke LLM's
llm_list = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "meta-llama/Llama-3.3-70B-Instruct",
    "microsoft/phi-4",
    "deepseek-ai/DeepSeek-V3",
    "NousResearch/Hermes-3-Llama-3.1-405B",
    "Qwen/QwQ-32B-Preview",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "01-ai/Yi-34B-Chat",
    "databricks/dbrx-instruct"
]

leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"

# --- Stap 1: Laad leidraad uit CSV in een pandas DataFrame ---
document_df = pd.read_csv(r'brondocumenten\hoofstukken.csv')

# Vaste vraag
user_query = "Wat is de sociale impact van de technische robuustheid van mijn AI-model?"

# --- Stap 2: Chunk het document op volgens de ingestelde strategie ---
chunks = chunk_file(document_df, chunk_size=chunk_size, chunking_strategy=chunking_strategy)

# --- Stap 3: Sla de chunks op in de gekozen vector database ---
collection = store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy)

# --- Stap 4: Haal relevante chunks op (dit gebeurt slechts 1 keer zodat alle LLM's dezelfde input krijgen) ---
retrieved_text = get_chunks(collection, user_query, storageStrategy)

# --- Stap 5: Query elke LLM in de lijst en verzamel de antwoorden ---
results = {}
for current_llm in llm_list:
    response = query_llm(retrieved_text, user_query, current_llm)
    results[current_llm] = response

# --- Stap 6: Schrijf de resultaten naar een tekstbestand ---
with open("resultatenIrrelevanteQuery.txt", "w", encoding="utf-8") as file:
    file.write("Dit bestand bevat de resultaten van het volgende proces:\n")
    file.write("1. Inladen van de leidraad uit een CSV-bestand.\n")
    file.write("2. Opsplitsen van de tekst in chunks op basis van de 'paragraph'-strategie.\n")
    file.write("3. Opslaan van deze chunks in een vector database.\n")
    file.write("4. Ophalen van relevante chunks op basis van de vraag: 'Waar gaat fase 5 van de leidraad over?'.\n")
    file.write("5. Queryen van verschillende LLM's met dezelfde input en verzamelen van hun antwoorden.\n\n")
    
    for llm_name, answer in results.items():
        file.write(f"Model: {llm_name}\n")
        file.write("Antwoord:\n")
        file.write(f"{answer}\n")
        file.write("-" * 80 + "\n")
