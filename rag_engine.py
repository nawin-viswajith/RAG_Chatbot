import os
import requests
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Load GROQ key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma setup
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag_store")

def extract_chunks_from_pdf(file_path, chunk_size=1200):  
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def index_document(chunks):
    embeddings = model.encode(chunks).tolist()
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

def retrieve_relevant_chunks(query, k=6):  
    query_emb = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return results['documents'][0]

def call_groq_llama3(context, question):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert research assistant. "
                "Use the following context to answer questions clearly and in detail. "
                "Reduce guessing. Only use the context and you can generate content related to the context and query."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a thorough answer:"
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 2048  
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']
