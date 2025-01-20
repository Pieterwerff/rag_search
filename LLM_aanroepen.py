# Importeren van benodigde modules
import openai
from transformers import pipeline
import requests
import os
from dotenv import load_dotenv

# Omgevingsvariabelen ophalen uit het .env-bestand
load_dotenv()

# API-sleutels ophalen uit .env-bestand
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# Controleren of API-sleutels aanwezig zijn
if not openai_api_key:
    raise ValueError("OpenAI API-sleutel ontbreekt. Voeg deze toe in het .env bestand.")
if not qdrant_api_key:
    print("Qdrant API-sleutel ontbreekt. Deze is optioneel, maar voeg deze toe als nodig.")
if not anthropic_api_key:
    print("Anthropic API-sleutel ontbreekt. Voeg deze toe in het .env bestand als je de Claude API wilt gebruiken.")

def call_gpt(retrieved_text, user_query, api_key):
    """
    Roept ChatGPT aan via OpenAI API.

    Parameters:
        retrieved_text (str): De tekst van de leidraad voor context.
        user_query (str): De vraag van de gezondheidszorgprofessional.
        api_key (str): OpenAI API-sleutel.

    Returns:
        str: gegenereerde tekst van ChatGPT.
    """
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
            {"role": "user", "content": f'''Baseer je antwoord op deze tekst: {retrieved_text}\n\n
                vraag: wat is AIPA
                antwoord: AIPA staat voor artificial intelligence prediction algorithm
                vraag: Mag ik pannenkoeken?
                antwoord: Dit antwoord kan ik niet vinden in de gegeven context
                vraag: {user_query}
                antwoord:
            '''}
        ]
    )
    return response.choices[0].message.content

def call_llama(retrieved_text, user_query, model="meta-llama/Llama-2-7b", max_length=200):
    """
    Roept Llama aan via Hugging Face Transformers pipeline.

    Parameters:
        retrieved_text (str): De tekst van de leidraad voor context.
        user_query (str): De vraag van de gezondheidszorgprofessional.
        model (str): Naam van het Hugging Face model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van Llama.
    """
    llama_pipeline = pipeline("text-generation", model=model)
    prompt = f"""Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt.

    Baseer je antwoord op deze tekst: {retrieved_text}

    vraag: wat is AIPA
    antwoord: AIPA staat voor artificial intelligence prediction algorithm
    vraag: Mag ik pannenkoeken?
    antwoord: Dit antwoord kan ik niet vinden in de gegeven context
    vraag: {user_query}
    antwoord:
    """
    response = llama_pipeline(prompt, max_length=max_length)
    return response[0]['generated_text']

def call_t5(retrieved_text, user_query, model="t5-base", max_length=200):
    """
    Roept T5 aan via Hugging Face Transformers pipeline.

    Parameters:
        retrieved_text (str): De tekst van de leidraad voor context.
        user_query (str): De vraag van de gezondheidszorgprofessional.
        model (str): Naam van het Hugging Face model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van T5.
    """
    t5_pipeline = pipeline("text2text-generation", model=model)
    prompt = f"""Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt.

    Baseer je antwoord op deze tekst: {retrieved_text}

    vraag: wat is AIPA
    antwoord: AIPA staat voor artificial intelligence prediction algorithm
    vraag: Mag ik pannenkoeken?
    antwoord: Dit antwoord kan ik niet vinden in de gegeven context
    vraag: {user_query}
    antwoord:
    """
    response = t5_pipeline(prompt, max_length=max_length)
    return response[0]['generated_text']

def call_anthropic(retrieved_text, user_query, api_key, model="claude-2", max_length=200):
    """
    Roept Claude aan via de Anthropic API.

    Parameters:
        retrieved_text (str): De tekst van de leidraad voor context.
        user_query (str): De vraag van de gezondheidszorgprofessional.
        api_key (str): Anthropic API-sleutel.
        model (str): Naam van het Claude-model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van Claude.
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    prompt = f"""Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt.

    Baseer je antwoord op deze tekst: {retrieved_text}

    vraag: wat is AIPA
    antwoord: AIPA staat voor artificial intelligence prediction algorithm
    vraag: Mag ik pannenkoeken?
    antwoord: Dit antwoord kan ik niet vinden in de gegeven context
    vraag: {user_query}
    antwoord:
    """
    
    data = {
        "model": model,
        "max_tokens": max_length,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json().get("content", "")[0].get("text", "")
    else:
        return f"Anthropic API error: {response.status_code}, {response.text}"

# Test code
if __name__ == "__main__":
    # Prompt instellen en LLM kiezen
    llm_name = "chatgpt"  # Kies: "chatgpt", "llama", "t5", "anthropic"
    prompt = "Mag ik AI gebruiken voor het stellen van diagnoses bij patiënten?"
    retrieved_text = """
    Bij het gebruik van AI voor diagnose moet zorgvuldig worden gekeken naar de validatie van het systeem. 
    AI mag alleen gebruikt worden als decision support tool en niet als vervanger van de zorgprofessional. 
    De eindverantwoordelijkheid voor de diagnose ligt altijd bij de behandelend arts.
    Het systeem moet CE-gecertificeerd zijn voor diagnostisch gebruik en voldoen aan de MDR-wetgeving.
    """

    # LLM aanroepen
    try:
        if llm_name == "chatgpt":
            output = call_gpt(retrieved_text, prompt, openai_api_key)
        elif llm_name == "llama":
            output = call_llama(retrieved_text, prompt)
        elif llm_name == "t5":
            output = call_t5(retrieved_text, prompt)
        elif llm_name == "anthropic":
            output = call_anthropic(retrieved_text, prompt, api_key=anthropic_api_key)
        else:
            raise ValueError(f"Onbekende LLM gekozen: {llm_name}")
        
        print(f"Output van {llm_name}:\n{output}")

    except Exception as e:
        print(f"Er trad een fout op bij het aanroepen van {llm_name}: {e}")