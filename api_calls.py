from openai import OpenAI
import os

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def query_llm(retrieved_text, user_query, llm):
    if llm == 'gpt':
        client = OpenAI()
    """
    Queries the LLM with a user query and relevant text.

    Parameters:
        retrieved_text (str): Relevant chunk retrieved from ChromaDB.
        user_query (str): User's input question.

    Returns:
        str: Response from the LLM.
    """
    print(retrieved_text)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Je bent een assistent die vragen over de Leidraad AI in de zorg beantwoordt."},
            {"role": "user", "content": f"Baseer je antwoord op deze tekst: {retrieved_text}\n\n{user_query}"}
        ]
    )
    return response.choices[0].message.content
