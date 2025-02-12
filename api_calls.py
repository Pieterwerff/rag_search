from openai import OpenAI
import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def query_llm(retrieved_object, user_query, llm):

    """
    Queries the LLM with a user query and relevant text.

    Parameters:
        retrieved_object (str): Relevant chunk retrieved from ChromaDB.
        user_query (str): User's input question.

    Returns:
        str: Response from the LLM.
    """
    if llm == 'gpt4o-mini':
        client = OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."
                },
                {
                    "role": "user",
                    "content": (
                        f"Baseer je antwoord op de volgende chunks: {retrieved_object}\n\n{user_query}. "
                        "Maak hier een JSON van waarbij de bronnen waaraan wordt gerefereerd in je antwoord worden meegegeven. "
                        "Als je je antwoord baseert op een zin waar een bron in staat, geef je deze bron mee in het JSON-object. "
                        "Als de bron NIET wordt gebruikt om een antwoord op te baseren, geef je deze NIET mee aan het JSON-object.\n\n"
                        "Daarnaast geef je de id mee van de chunk waar je je antwoord op hebt gebaseerd. \n\n"
                        "Voorbeeld: \n\n"
                        "{\n"
                        "  \"antwoord\": \"Op basis van de aangeleverde tekst kan geconcludeerd worden dat er een duidelijke relatie bestaat tussen X en Y. "
                        "De tekst geeft aan dat de implementatie van Z niet alleen de efficiëntie verhoogt, maar ook de gebruikerservaring verbetert [1]. "
                        "Daarnaast wordt er gewezen op een toename in innovatie binnen de organisatie door deze aanpassingen [2]."
                        "Externe validatie in het kader van AIPA verwijst naar het proces waarbij de in fase 2 ontwikkelde AIPA-modellen worden geëvalueerd met gebruik van een dataset die niet is gebruikt voor de ontwikkeling van het model[15]. "
                        "Dit betreft zowel de evaluatie van de voorspellende waarde als de evaluatie van de (meer)waarde van het model ten opzichte van de huidige zorgpraktijk[27,28].\",\n"
                        "  \"chunks\": [\n"
                        "    {\n"
                        "      \"id\": \"be660416-1680-4ac9-9f04-9b75e162d742\", \n"
                        "      \"bronnen\": [1, 2]\n"
                        "    },\n"
                        "    {\n"
                        "      \"id\": \"09f1ce53-2848-41a0-b601-3fc9e12bdafa\", \n"
                        "      \"bronnen\": [15, 27, 28]\n"
                        "    }\n"
                        "  ]\n"
                        "}"
                    )
                }
            ]
        )
    if llm == 'gpt-3.5-turbo':
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."
                },
                {
                    "role": "user",
                    "content": (
                        f"Baseer je antwoord op de volgende chunks: {retrieved_object}\n\n{user_query}. "
                        "Maak hier een JSON van waarbij de bronnen waaraan wordt gerefereerd in je antwoord worden meegegeven. "
                        "Als je je antwoord baseert op een zin waar een bron in staat, geef je deze bron mee in het JSON-object. "
                        "Als de bron NIET wordt gebruikt om een antwoord op te baseren, geef je deze NIET mee aan het JSON-object.\n\n"
                        "Daarnaast geef je de id mee van de chunk waar je je antwoord op hebt gebaseerd. \n\n"
                        "Voorbeeld: \n\n"
                        "{\n"
                        "  \"antwoord\": \"Op basis van de aangeleverde tekst kan geconcludeerd worden dat er een duidelijke relatie bestaat tussen X en Y. "
                        "De tekst geeft aan dat de implementatie van Z niet alleen de efficiëntie verhoogt, maar ook de gebruikerservaring verbetert [1]. "
                        "Daarnaast wordt er gewezen op een toename in innovatie binnen de organisatie door deze aanpassingen [2]."
                        "Externe validatie in het kader van AIPA verwijst naar het proces waarbij de in fase 2 ontwikkelde AIPA-modellen worden geëvalueerd met gebruik van een dataset die niet is gebruikt voor de ontwikkeling van het model[15]. "
                        "Dit betreft zowel de evaluatie van de voorspellende waarde als de evaluatie van de (meer)waarde van het model ten opzichte van de huidige zorgpraktijk[27,28].\",\n"
                        "  \"chunks\": [\n"
                        "    {\n"
                        "      \"id\": \"be660416-1680-4ac9-9f04-9b75e162d742\", \n"
                        "      \"bronnen\": [1, 2]\n"
                        "    },\n"
                        "    {\n"
                        "      \"id\": \"09f1ce53-2848-41a0-b601-3fc9e12bdafa\", \n"
                        "      \"bronnen\": [15, 27, 28]\n"
                        "    }\n"
                        "  ]\n"
                        "}"
                    )
                }
            ]
        )
    return response.choices[0].message.content

def query_llm_deep_infra(retrieved_object, user_query, llm):

    endpoint = "https://api.deepinfra.com/v1/openai/chat/completions"
    key = os.getenv('DEEP_INFRA_API_KEY')
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        'model': llm, 
        #'prompt': 'Once upon a time',
        'messages': [
             {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
             {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_object}\n\n{user_query}"}
         ]
    }
    response = requests.post(endpoint, headers=headers, json=data).json()

    return response["choices"][0]["message"]["content"]



    # openai = OpenAI(
    #     api_key=os.getenv('DEEP_INFRA_API_KEY'),
    #     base_url="https://api.deepinfra.com/v1/openai",
    # )

    # #stream = True # or False

    # response = openai.chat.completions.create(
    #     model=llm,
    #     messages=[
    #         {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
    #         {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_object}\n\n{user_query}"}
    #     ]
    #     #stream=stream,
    # )

    # return response.choices[0].message.content