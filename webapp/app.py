
import sys
import os
# Voeg de bovenliggende map toe aan sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, json, request, render_template
from preprocessing.chunking import chunk_file
from llm_calls.api_calls import query_llm
from vector_storage import store_chunks, get_chunks
from agents.question_preprocessing import preprocessor_agent
import pandas as pd
import json

import pandas as pd
from agents.question_preprocessing import language_agent



# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph' # might become a list of strings to iterate through
llm = 'gpt-4o-mini' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"

# --- Step: Load markdown Document ---
document_df = pd.read_csv('brondocumenten/hoofstukken.csv')

# --- Step 2: Chunk Document using different types of chunking strategies ---

chunks = chunk_file(document_df, chunk_size=chunk_size, chunking_strategy=chunking_strategy)

# --- Step 3: Store Chunks in specific kind of chunk database ---

collection = store_chunks(chunks, storageStrategy, embeddingStrategy, leidraad, chunking_strategy)


app = Flask(__name__)


def extract_named_sources(response, given_chunks):
    # Zorg dat de response een dict is
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError("Kon de JSON-response niet parsen: " + str(e))
    
    # Lees het CSV-bestand en maak een dictionary:
    # (chapter_number, reference_index) -> reference_text
    refs_df = pd.read_csv('referenties/extracted_references.csv')
    ref_dict = {
        (row["chapter_number"], row["reference_index"]): row["reference_text"]
        for _, row in refs_df.iterrows()
    }
    
    # Helperfunctie om een chunk-id te verkrijgen, ongeacht of de chunk een dict of object is.
    def get_chunk_id(chunk):
        return chunk.get("id") if isinstance(chunk, dict) else getattr(chunk, "id", None)
    
    # Maak een mapping van chunk id naar chunk-object uit given_chunks
    chunk_mapping = {
        get_chunk_id(chunk): chunk for chunk in given_chunks if get_chunk_id(chunk) is not None
    }
    
    named_sources = []
    
    # Itereer over de chunks in de LLM-response
    for chunk_info in response.get("chunks", []):
        chunk_id = chunk_info.get("id")
        if chunk_id not in chunk_mapping:
            print(f"Waarschuwing: Geen chunk gevonden voor id {chunk_id}.")
            continue
        
        # Verkrijg de payload en parse deze indien nodig
        chunk = chunk_mapping[chunk_id]
        payload = chunk.get("payload") if isinstance(chunk, dict) else getattr(chunk, "payload", {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        chapter_number = payload.get("chapter_number")
        
        # Voor elke referentie-index in de LLM-response, zoek de bijbehorende referentietekst
        for ref_index in chunk_info.get("bronnen", []):
            reference_text = ref_dict.get((chapter_number, ref_index))
            if reference_text:
                named_sources.append(reference_text)
            else:
                print(f"Waarschuwing: Geen referentie gevonden voor chapter {chapter_number} met reference_index {ref_index}.")
    
    return named_sources



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["question"]

        use_query_preprocessor = request.form.get("use_query_translator") == "true"
        if use_query_preprocessor:
            user_query = language_agent(user_query)
            print("Translated question:", user_query)

        chunks = get_chunks(collection, user_query, storageStrategy, n_chunks=int(request.form["chunks"]))
        llm_response = query_llm(retrieved_object=chunks, user_query=user_query, llm=llm)

        sources = extract_named_sources(llm_response, chunks)
        print(sources)
        object_response = json.loads(llm_response)
        
        return render_template("result.html", question=user_query, chunks=chunks, llm_response=object_response.get("antwoord"), sources=sources, n_chunks=request.form["chunks"])
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
