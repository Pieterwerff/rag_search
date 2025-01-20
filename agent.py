# Validation of retrieved information	
# Query pre-processing	

from openai import OpenAI
import os

OpenAI.api_key = os.getenv('OPENAI_API_KEY')


def language_agent(user_query):
    """
    Recognizes the language of the prompt from the user

    Parameters: 
    user query (Str): The query sent to Smart search tool

    Returns: 
    str: either english, dutch or other
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Je bent een taalherkenner. Je beantwoord of de taal engels of nederlands is, als de taal nederlands is reageer je: nl. Als de taal Engels is reageer je: eng. Als de taal geen van beide is reageer je: other"},
            {"role": "user", "content": f"{user_query}"}
        ]
    )
    return response.choices[0].message.content

