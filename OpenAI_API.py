import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from operator import itemgetter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

parser = StrOutputParser()

chain = model | parser

# Process the PDF
PDF_PATH = "/Users/beyzagokkaya/Documents/GitHub/rag_search/Leidraad kwaliteit AI in de zorg NL.pdf"  
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


# Define a prompt template
template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know."

Context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Test prompt with extracted context
chain = prompt | model | parser

try:
    # Use the first document's content as context
    result = chain.invoke({
        "context": documents[0].page_content[:1000],  # Limit the context size if it's too large
        "question": "What is the primary topic of the document?"
    })
    print(result)
except Exception as e:
    print(e)


