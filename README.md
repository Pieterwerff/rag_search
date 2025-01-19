# RAG Search Tool

## Overzicht
------------

Deze tool is een Retrieval-Augmented Generation (RAG)-gebaseerde applicatie die de Leidraad AI in de Zorg gebruikt om informatieve antwoorden te genereren. Het combineert documentchunking, vectoropslag, en een Large Language Model (LLM) om vragen van gebruikers te beantwoorden met behulp van relevante informatie uit de leidraad.

## Projectstructuur
-----------------

* `webapp/`: Flask-webapplicatie
	+ `static/`: JavaScript-functionaliteit en styling
	+ `templates/`: HTML-pagina's voor gebruikersinvoer en resultaten
	+ `app.py`: Webapplicatie met Flask
* `agent.py`: Agents voor taalherkenning en reranking
* `api_calls.py`: Interacties met de LLM (zoals GPT)
* `chunking.py`: Tools voor het opsplitsen van documenten in kleinere stukken
* `vector_storage.py`: Opslag en ophalen van vectoren (ChromaDB/Qdrant)
* `main.py`: Debug-tool en script om alles samen te testen
* `run.ipynb`: Jupyter Notebook voor experimenten
* `leidraad.txt`: Tekstbestand van de AI-leidraad
* `Leidraad kwaliteit AI in de zorg NL.pdf`: Originele PDF van de leidraad
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

* Bestand: `agent.py`
* Bevat modules zoals:
	+ Taalherkenningsagent
	+ Reranking-agent voor beter matchen van relevante informatie

### 5. Webinterface

* Bestand: [webapp/app.py](cci:7://file:///c:/Users/jangl/OneDrive/Documents/GitHub/rag_search/webapp/app.py:0:0-0:0)
* Flask-webapplicatie die gebruikers toelaat vragen te stellen via een eenvoudige interface:
	+ Indexpagina: Gebruikers kunnen een vraag stellen.
	+ Resultpagina: Toont het antwoord van de LLM en de bijbehorende bronnen.

## Installatie
------------

### Vereisten

* Python 3.8 of hoger
* Vereiste Python-pakketten 

### Instellen van .env

* Maak een `.env`-bestand en voeg je configuratie toe, zoals API-sleutels:
	+ `OPENAI_API_KEY=je-api-sleutel-hier`
	+ `Qdrant_KEY=je_qdrant_key_hier`

## Gebruik
-----

### Via de webinterface

* Run app.py, ga in je browser naar http://127.0.0.1:5000/
* Open de indexpagina.
* Stel een vraag in het tekstveld en geef aan hoeveel chunks op moeten worden gehaald.
* Ontvang een antwoord samen met relevante bronnen op de resultpagina.

### Via main.py

* Gebruik `main.py` om de pipeline te testen en te debuggen: `python main.py`


## Toekomstige verbeteringen
-------------------------

* Implementatie van de agentic structuur
* Implementatie verschillende LLM's
* Validatie van de resultaten.

----
_dit readme-bestand is geschreven met behulp van de codeium-extensie in vscode en daarna met de hand aangepast._