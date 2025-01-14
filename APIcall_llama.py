from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

def perform_api_call(query):
   
    load_dotenv()
    Hugginface_accces_token = os.getenv("Hugginface_accces_token")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    access_token = Hugginface_accces_token
    login(token=access_token)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=True,
        temperature=0.1,
        max_new_tokens=100,
    )

    llama_pipeline = HuggingFacePipeline(pipeline=pipe)

    # Configure the QA chain
    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
    )

    retriever = None  # No embedding or retriever setup needed

    qa_chain = RetrievalQA.from_llm(
        llama_pipeline, retriever=retriever, prompt=qa_prompt
    )

    # Test QA chain with the provided query
    qa_result = qa_chain({"query": query})
    return qa_result
