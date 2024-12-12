from openai import OpenAI
import os 
from dotenv import load_dotenv
import pymupdf4llm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# documents = pymupdf4llm.to_markdown("Leidraad kwaliteit AI in de zorg NL.pdf")
md_text = open("leidraad.txt")
# f = open("leidraad.txt", "w")
# f.write(documents)

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()



# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": f"Je beantwoordt vragen over het document Leidraad in de zorg. Baseer je antwoord op de volgende tekst: {md_text}"},
#         {
#             "role": "user",
#             "content": "Waar staat AIPA voor? "
#         }
#     ]
# )

# print(completion.choices[0].message.content)