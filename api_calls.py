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
