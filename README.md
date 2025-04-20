# RAG Research Chatbot (GROQ + ChromaDB + Streamlit)

This project is a lightweight Retrieval-Augmented Generation (RAG) chatbot that allows you to interact with the content of your PDF documents through natural language questions. It integrates:

- LLaMA 3.3 70B (Versatile) via the GROQ API
- ChromaDB for local vector-based retrieval
- Streamlit for an interactive web-based interface

## Features

- Upload any PDF document (research paper, report, etc.)
- Ask questions in natural language
- Get context-aware answers based on relevant sections from the document
- Auto-scroll and clean, ChatGPT-style chat interface
- Easily extendable for multiple documents or persistent vector stores

## Tech Stack

| Component       | Tool/Library                              |
|------------------|--------------------------------------------|
| User Interface   | Streamlit                                  |
| Embeddings       | Sentence Transformers (MiniLM-L6-v2)       |
| Vector Store     | ChromaDB                                   |
| LLM Backend      | GROQ API with LLaMA 3.3 (70B)              |
| PDF Parsing      | PyPDF2                                     |

## File Structure

```
├── app.py             # Streamlit frontend and chat logic
├── rag_engine.py      # RAG logic including embedding, GROQ calls
├── chroma_chain.py    # Optional LangChain-based flow
├── .env               # Environment variables (GROQ API key)
├── requirements.txt   # Python dependencies
```

## Setup Instructions

1. **Clone this repository:**

```bash
git clone https://github.com/yourusername/rag-groq-chatbot.git
cd rag-groq-chatbot
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up the `.env` file with your GROQ API key:**

Create a file named `.env` in the root directory and add:

```
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your actual API key from [GROQ Console](https://console.groq.com).

## How to Run

Start the Streamlit application with:

```bash
streamlit run app.py
```

Steps:
- Upload a PDF document
- Type a question in the input box
- The model will retrieve relevant content and generate an answer based on the PDF

## Notes

- The document embeddings are stored locally in the `chroma_store/` directory.
- The GROQ API is used only for inference; your document data remains local.
- Only the top-k relevant chunks are passed to the LLaMA 3 model for each query.

## Future Improvements

- Support for multi-document sessions
- Source text highlighting and citations
- Save/export chat history
- Add support for additional LLM backends (OpenAI, HuggingFace, etc.)

## License

This project is open-source under the GPL-3.0 License.
