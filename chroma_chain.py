import os
from langchain.chat_models import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = "chroma_store"

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_and_build_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=CHROMA_DB_DIR
    )
    vectordb.persist()
    return vectordb

def get_qa_chain(vectordb):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI research assistant. Use the following context from scientific papers or documents to answer the user's question.
Always respond with clarity and relevance, and cite the source when needed.

Context:
{context}

Question:
{question}

Answer (detailed, formal, and based strictly on context):
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
