from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

load_dotenv()
Hugginface_accces_token = os.getenv("Hugginface_accces_token")

# Load the PDF
loader = PyPDFLoader("/Users/beyzagokkaya/school/Smart_search/Leidraad kwaliteit AI in de zorg NL.pdf")
documents = loader.load()

# Validate the loaded documents
print(f"Number of documents: {len(documents)}")
print(documents[0].page_content[:500])  # Print first 500 characters of the first document

# Split documents into manageable chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store
db = FAISS.from_documents(chunks, embeddings)

# Print previews of documents
for i in range(3):
    print(f"Document {i} preview:")
    print(documents[i].page_content[:500])
    print("-" * 80)

# Print document lengths
for i, doc in enumerate(documents):
    print(f"Document {i} length: {len(doc.page_content)} characters")

# Perform a similarity search
query = "What is AIPA?"
results = db.similarity_search(query, k=3)
for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(result.page_content)
    print("-" * 80)

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
access_token = Hugginface_accces_token
login(token=access_token)

try:
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically uses GPU or CPU
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    raise

# Example usage of the Llama model
prompt = "Explain AIPA in simple terms: "
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_length=100)
print("Generated text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Configure the retriever
retriever = db.as_retriever()

# Setup HuggingFacePipeline for Llama
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True, 
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=100,  # max number of tokens to generate in the output
)

llama_pipeline = HuggingFacePipeline(pipeline=pipe)

# Combine retriever and pipeline for QA chain
from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
)

qa_chain = RetrievalQA.from_llm(
    llama_pipeline, retriever=retriever, prompt=qa_prompt
)
# Test QA chain
qa_query = "What is AIPA?"
qa_result = qa_chain({"query": qa_query})
print(f"Answer to query: {qa_result}")