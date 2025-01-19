
import sys
import os

# Voeg de bovenliggende map toe aan sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, render_template
from chunking import chunk_file
from api_calls import query_llm
from vector_storage import store_chunks, get_chunks

# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph' # might become a list of strings to iterate through
llm = 'gpt4o-mini' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"

# --- Step 1: Load markdown Document ---
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()  # Read content into a string

# --- Step 2: Chunk Document using different types of chunking strategies ---

chunks = chunk_file(document, chunk_size=chunk_size, chunking_strategy=chunking_strategy)

# --- Step 3: Store Chunks in specific kind of chunk database ---

collection = store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["question"]
        chunks = get_chunks(collection, user_query, storageStrategy, n_chunks=request.form["chunks"])
        llm_response = query_llm(retrieved_text=chunks, user_query=user_query, llm=llm)
        return render_template("result.html", question=user_query, chunks=chunks, llm_response=llm_response, n_chunks=request.form["chunks"])
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
