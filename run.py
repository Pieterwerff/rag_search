from openai import OpenAI
import os 
from dotenv import load_dotenv
import pymupdf4llm
md_text = pymupdf4llm.to_markdown("Leidraad kwaliteit AI in de zorg NL.pdf")
print(md_text)
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
                           
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Je beantwoord vragen over het document Leidraad in de zorg. Baseer je antwoord op de volgende tekst: {md_text}"},
        {
            "role": "user",
            "content": "Waar gaat het document over? "
        }
    ]
)

print(completion.choices[0].message.content)