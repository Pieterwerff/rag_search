# Bibliotheek
!pip install openai transformers 

# Importeren van modules
import openai
from transformers import pipeline
import requests

# Functie om ChatGPT aan te roepen via OpenAI API
def call_chatgpt(prompt, api_key):
    """
    Roept ChatGPT aan via OpenAI API.

    Parameters:
        prompt (str): Input prompt voor ChatGPT.
        api_key (str): OpenAI API-sleutel.

    Returns:
        str: gegenereerde tekst van ChatGPT.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Je bent een assistent."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Functie om Llama aan te roepen via Hugging Face Transformers
def call_llama(prompt, model="meta-llama/Llama-2-7b", max_length=200):
    """
    Roept Llama aan via Hugging Face Transformers pipeline.

    Parameters:
        prompt (str): Input prompt voor Llama.
        model (str): Naam van het Hugging Face model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van Llama.
    """
    llama_pipeline = pipeline("text-generation", model=model)
    response = llama_pipeline(prompt, max_length=max_length)
    return response[0]['generated_text']

# Functie om T5 aan te roepen via Hugging Face Transformers
def call_t5(prompt, model="t5-base", max_length=200):
    """
    Roept T5 aan via Hugging Face Transformers pipeline.

    Parameters:
        prompt (str): Input prompt voor T5.
        model (str): Naam van het Hugging Face model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van T5.
    """
    t5_pipeline = pipeline("text2text-generation", model=model)
    response = t5_pipeline(prompt, max_length=max_length)
    return response[0]['generated_text']

# Functie om Claude aan te roepen via de Anthropic API
def call_anthropic(prompt, api_key, model="claude-2", max_length=200):
    """
    Roept Claude aan via de Anthropic API.

    Parameters:
        prompt (str): Input prompt voor Claude.
        api_key (str): Anthropic API-sleutel.
        model (str): Naam van het Claude-model.
        max_length (int): Maximale lengte van de gegenereerde tekst.

    Returns:
        str: gegenereerde tekst van Claude.
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "model": model,
        "max_tokens_to_sample": max_length
    }
    response = requests.post(
        "https://api.anthropic.com/v1/complete",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json().get("completion", "")
    else:
        return f"Anthropic API error: {response.status_code}, {response.text}"

# Prompt instellen en LLM kiezen
llm_name = "chatgpt"  # Kies: "chatgpt", "llama", "t5", "anthropic"
prompt = "Leg uit wat contextueel chunking is in NLP."

# API-sleutels en instellingen
openai_api_key = "YOUR_OPENAI_API_KEY"  # Vervang door je OpenAI-sleutel
anthropic_api_key = "YOUR_ANTHROPIC_API_KEY"  # Vervang door je Anthropic-sleutel
llama_model = "meta-llama/Llama-2-7b"  # Vervang door een lokaal of ander model als nodig
t5_model = "t5-base"  # Vervang door een gewenst T5-model

# Controleer of benodigde API-sleutels aanwezig zijn
if llm_name == "chatgpt" and not openai_api_key:
    raise ValueError("OpenAI API-sleutel ontbreekt. Voeg je OpenAI API-sleutel toe.")
if llm_name == "anthropic" and not anthropic_api_key:
    raise ValueError("Anthropic API-sleutel ontbreekt. Voeg je Anthropic API-sleutel toe.")

# LLM aanroepen
try:
    if llm_name == "chatgpt":
        output = call_chatgpt(prompt, openai_api_key)
    elif llm_name == "llama":
        output = call_llama(prompt, model=llama_model)
    elif llm_name == "t5":
        output = call_t5(prompt, model=t5_model)
    elif llm_name == "anthropic":
        output = call_anthropic(prompt, api_key=anthropic_api_key)
    else:
        raise ValueError(f"Onbekende LLM gekozen: {llm_name}")
    
    # Resultaat tonen
    print(f"Output van {llm_name}:\n{output}")

except Exception as e:
    print(f"Er trad een fout op bij het aanroepen van {llm_name}: {e}")