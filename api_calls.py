from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

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

def query_llm_deep_infra(retrieved_text, user_query, llm):

    endpoint = "https://api.deepinfra.com/v1/openai/chat/completions"
    key = os.getenv('DEEP_INFRA_API_KEY')
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        'model': llm, 
        #'prompt': 'Once upon a time',
        'messages': [
             {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
             {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
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
    #         {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
    #     ]
    #     #stream=stream,
    # )

    # return response.choices[0].message.content