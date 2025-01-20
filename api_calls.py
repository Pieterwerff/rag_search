from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def query_llm(retrieved_text, user_query, llm):

    """
    Queries the LLM with a user query and relevant text.

    Parameters:
        retrieved_text (str): Relevant chunk retrieved from ChromaDB.
        user_query (str): User's input question.

    Returns:
        str: Response from the LLM.
    """
    print(retrieved_text)
    if llm == 'gpt4o-mini':
        client = OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
                {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
            ]
        )
    if llm == 'gpt-3.5-turbo':
        client = OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
                {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
            ]
        )
    return response.choices[0].message.content

# Aanroepen code vanuit LLM_aanroepen.py
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
