# Importeren van benodigde modules
import openai
import requests
import os
from dotenv import load_dotenv

# Omgevingsvariabelen ophalen uit het .env-bestand
load_dotenv()

# API-sleutels ophalen uit .env-bestand
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
llama_api_key = os.getenv('LLAMA_API_KEY')

# Debugging: Controleren of API-sleutels worden geladen
print(f"OpenAI API Key geladen: {'Ja' if openai_api_key else 'Nee'}")
print(f"LLaMA API Key geladen: {'Ja' if llama_api_key else 'Nee'}")
print(f"Anthropic API Key geladen: {'Ja' if anthropic_api_key else 'Nee'}")

# Generieke functie om een LLM aan te roepen
def call_llm(llm_name, retrieved_text, user_query, max_length=200):
    """
    Roept een LLM aan op basis van de opgegeven naam.

    Parameters:
        llm_name (str): Naam van de LLM ("chatgpt", "llama", "anthropic").
        retrieved_text (str): De tekst van de leidraad voor context.
        user_query (str): De vraag van de gebruiker.
        max_length (int): Maximale lengte van het gegenereerde antwoord (indien ondersteund).

    Returns:
        str: Het gegenereerde antwoord of een foutmelding.
    """
    # Gemeenschappelijke prompt voor alle modellen
    prompt = f"""Baseer je antwoord op deze tekst: {retrieved_text}\n\n
    vraag: wat is AIPA
    antwoord: AIPA staat voor artificial intelligence prediction algorithm
    vraag: Mag ik pannenkoeken?
    antwoord: Dit antwoord kan ik niet vinden in de gegeven context
    vraag: {user_query}
    antwoord:
    """

    # Logica voor specifieke LLMs
    try:
        if llm_name == "chatgpt":
            # ChatGPT via OpenAI API
            openai.api_key = openai_api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']

        elif llm_name == "llama":
            # LLaMA via Hugging Face API
            api_url = f"https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b"
            headers = {
                "Authorization": f"Bearer {llama_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "inputs": prompt,
                "parameters": {"max_length": max_length, "return_full_text": False},
            }
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()[0]['generated_text']
            elif response.status_code == 401:
                return "LLaMA API error: Ongeldige API-sleutel of geen toegang tot het model."
            elif response.status_code == 403:
                return "LLaMA API error: Toegang tot het model geweigerd. Controleer of je de gebruiksvoorwaarden hebt geaccepteerd."
            else:
                return f"LLaMA API error: {response.status_code}, {response.text}"

        elif llm_name == "anthropic":
            # Claude via Anthropic API
            headers = {
                "x-api-key": anthropic_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": "claude-2",
                "max_tokens": max_length,
                "messages": [{"role": "user", "content": prompt}]
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

        else:
            return f"Onbekend LLM: {llm_name}"

    except Exception as e:
        return f"Er trad een fout op bij het aanroepen van {llm_name}: {e}"

# Testcode
if __name__ == "__main__":
    # Terminalinterface voor keuze van LLM
    print("Kies de LLM die je wilt gebruiken:")
    print("1. ChatGPT (OpenAI)")
    print("2. LLaMA (Hugging Face)")
    print("3. Anthropic Claude")

    choice = input("Typ het nummer van je keuze (1/2/3): ").strip()

    if choice == "1":
        llm_name = "chatgpt"
    elif choice == "2":
        llm_name = "llama"
    elif choice == "3":
        llm_name = "anthropic"
    else:
        print("Ongeldige keuze! Standaard wordt ChatGPT gebruikt.")
        llm_name = "chatgpt"

    # Testvraag en context
    prompt = "Mag ik AI gebruiken voor het stellen van diagnoses bij patiënten?"
    retrieved_text = """
    Bij het gebruik van AI voor diagnose moet zorgvuldig worden gekeken naar de validatie van het systeem. 
    AI mag alleen gebruikt worden als decision support tool en niet als vervanger van de zorgprofessional. 
    De eindverantwoordelijkheid voor de diagnose ligt altijd bij de behandelend arts.
    Het systeem moet CE-gecertificeerd zijn voor diagnostisch gebruik en voldoen aan de MDR-wetgeving.
    """

    # Aanroepen van LLM met testdata
    print(f"# Aanroepen van {llm_name}")
    output = call_llm(llm_name, retrieved_text, prompt)
    print(f"Output van {llm_name}:\n{output}")