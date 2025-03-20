# RAG Search Tool

## Overzicht
------------

Deze tool is een Retrieval-Augmented Generation (RAG)-gebaseerde applicatie die de Leidraad AI in de Zorg gebruikt om informatieve antwoorden te genereren. Het combineert documentchunking, vectoropslag, en een Large Language Model (LLM) om vragen van gebruikers te beantwoorden met behulp van relevante informatie uit de leidraad.

## **Projectstructuur**
-----------------
* **webapp/**: Flask-webapplicatie  
  + **static/**: JavaScript-functionaliteit en styling  
  + **templates/**: HTML-pagina's voor gebruikersinvoer en resultaten  
  + `app.py`: Webapplicatie met Flask  
* **agents/**: Agents voor taalherkenning en vraagvoorbewerking  
  + `language_agent.py`: Taalherkenning en reranking  
  + `question_preprocessing.py`: Voorbewerking van gebruikersvragen  
* **brondocumenten/**: Referentiemateriaal en bronnen  
  + `hoofstukken.csv`: Structuur van hoofdstukken  
  + `leidraad.txt`: Bewerkt tekstbestand  
  + `Leidraad kwaliteit AI in de zorg NL.pdf`: Originele PDF van de leidraad  
* **databases/**: Opslag van vectoren en indexbestanden  
  + `chroma.sqlite3`: SQL-database voor vectoropslag  
* **llm_calls/**: Interacties met de LLM (zoals GPT)  
  + `api_calls.py`: Communicatie met LLM  
  + `user_select.py`: Selectiemechanismen voor gebruikersinteractie  
* **preprocessing/**: Tools voor het opsplitsen van documenten  
  + `agentic_chunker.py`: Meer geavanceerde chunking-methode, niet meegenomen in eindproduct 
  + `chunking.py`: Basisscript voor chunking  
* **referenties/**: Scripts en bestanden voor referentie-extractie  
  + `extracted_references.csv`: Geëxtraheerde referenties  
  + `get_chapters.py`: Haalt hoofdstukken uit documenten  
  + `get_citations.py`: Haalt citaties uit tekst  
* **Validatie/**: Validatiescripts en testresultaten  
  + `eersteValidatie.py`: Validatiescript, niet meegenomen in eindproduct
  + `Ragas.py`: Validatie met RAGAS  
* `vector_storage.py`: Opslag en ophalen van vectoren (ChromaDB/Qdrant)  
* `main.py`: Debug-tool en script om alles samen te testen  
* `run.ipynb`: Jupyter Notebook voor experimenten  
* `.env`: Configuratiebestand (bijv. API-sleutels)  

## Functionaliteit
--------------

### 1. Chunking van documenten

* Bestand: `chunking.py`
* Documenten worden opgesplitst in kleinere chunks met verschillende chunking-strategieën, zoals:
	+ Op basis van paragraaf
	+ Op basis van vaste tekstlengte

### 2. Vectoropslag

* Bestand: `vector_storage.py`
* Maakt gebruik van vectoropslag (bijv. ChromaDB en Qdrant) om chunks en hun embeddings op te slaan.
* Ondersteunt het ophalen van de meest relevante chunks op basis van een gebruikersquery.

### 3. LLM-interactie

* Bestand: `api_calls.py`
* Stuurt de gebruikersvraag en de opgehaalde chunks naar een LLM (zoals GPT) om een samenvatting of antwoord te genereren.

### 4. Agents

* Bestanden: `agentic_chunking.py`, `lanuage_agent.py`, `question_preprocessing.py`
* Bevat modules zoals:
	+ Agentic chunking strategie
 	+ Taalherkenningsagent	
	+ Vragen verwerker (vertaald vragen naar vakjargon)

### 5. Webinterface

* Bestand: [webapp/app.py]
* Flask-webapplicatie die gebruikers toelaat vragen te stellen via een eenvoudige interface:
	+ Indexpagina: Gebruikers kunnen een vraag stellen.
	+ Resultpagina: Toont het antwoord van de LLM en de bijbehorende bronnen.

### 6. RAGAS-validatiemethode

* Bestand: `Ragas.py`
* Implementeert de RAGAS-validatiemethode voor het evalueren van Retrieval-Augmented Generation (RAG) systemen:
	+ Antwoordrelevantie: Meet hoe relevant het gegenereerde antwoord is ten opzichte van de gestelde vraag.
	+ Getrouwheid: Evalueert de feitelijke consistentie van het antwoord met de gegeven context.
	+ Contextuele precisie: Beoordeelt of de relevante items in de context correct zijn gerangschikt.
	+ Contextuele recall: Meet of alle relevante contextuele informatie is opgehaald.

## Installatie
------------

### Vereisten

* Python 3.8 of hoger
* Vereiste Python-pakketten 

### Instellen van .env

* Maak een `.env`-bestand en voeg je configuratie toe, zoals API-sleutels:
	+ `OPENAI_API_KEY=je_openai_api_sleutel_hier`
	+ `Qdrant_KEY=je_qdrant_key_hier`
	+ `RAGAS_APP_TOKEN=je_ragas_key_hier`
	+ `DEEP_INFRA_API_KEY=je_deep_infra_llm_key_hier`
 + 

## Gebruik
-----

### Installeren van dependencies

* Installeer de vereiste dependencies met: `pip install -r requirements.txt`

### Via de webinterface

* Run `app.py`, ga in je browser naar http://127.0.0.1:5000/
* Open de indexpagina.
* Stel een vraag in het tekstveld en geef aan hoeveel chunks op moeten worden gehaald.
* Ontvang een antwoord samen met relevante bronnen op de resultpagina.

### Via main.py

* Gebruik `main.py` om de pipeline te testen en te debuggen: `python main.py`

## Toekomstige verbeteringen
-------------------------

* Implementatie van de agentic chunker
* Implementatie topic enhanced reranker
* Toevoeging hyperlink naar pdf + paginanummer

----
_dit readme-bestand is geschreven met behulp van de codeium-extensie in vscode en daarna met de hand aangepast._
