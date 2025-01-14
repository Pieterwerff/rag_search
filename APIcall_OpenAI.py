import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

def call_api(context, question):
 
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Initialize the model
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

    # Set up the output parser
    parser = StrOutputParser()

    # Define the prompt template
    template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know."

Context: {context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Combine the prompt, model, and parser into a chain
    chain = prompt | model | parser

    try:
        # Invoke the chain with the provided context and question
        result = chain.invoke({
            "context": context,
            "question": question
        })
        return result
    except Exception as e:
        return f"Error: {e}"

