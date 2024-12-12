import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Step 1: Load Document ---
# Read the plain text file
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()  # Read content into a string


# --- Step 2: Store Text in ChromaDB ---
# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("leidraad_ai_in_zorg")

# Split document into small chunks (naive split)
chunks = [document[i:i+1000] for i in range(0, len(document), 1000)]  # 500-char chunks
for idx, chunk in enumerate(chunks):
    collection.add(
        ids=[str(idx)],
        documents=[chunk]
    )
# print(chunks)
# --- Step 3: User Query ---
user_query = "Wat is de uitdrukking van deze leidraad?"

# Retrieve relevant chunks using Chroma
results = collection.query(
    query_texts=[user_query],
    n_results=10
)
retrieved_text = results['documents'][0][0]  # Top relevant chunk
print(retrieved_text)
# --- Step 4: Generate Answer with OpenAI GPT ---
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
        {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
    ]
)

# --- Step 5: Output Result ---
print("Vraag:", user_query)
print("Antwoord:", response.choices[0].message.content)
