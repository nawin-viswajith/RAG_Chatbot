import streamlit as st
import tempfile
from rag_engine import (
    extract_chunks_from_pdf,
    index_document,
    retrieve_relevant_chunks,
    call_groq_llama3
)

# Configure Streamlit
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.markdown("<h1 class='title'>Research Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_indexed" not in st.session_state:
    st.session_state.document_indexed = False

with st.expander("📄 Upload a PDF", expanded=False):
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    
    # Store filename to detect if it's new
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None

    if uploaded_file:
        file_name = uploaded_file.name

        # Only clear if new file
        if file_name != st.session_state.last_uploaded_filename:
            st.session_state.chat_history = []
            st.session_state.last_uploaded_filename = file_name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Indexing the document..."):
            chunks = extract_chunks_from_pdf(tmp_path)
            index_document(chunks)
            st.session_state.document_indexed = True
        st.success("Document indexed successfully!")

# Chat UI
if st.session_state.document_indexed:
    # Show chat messages (chronological order)
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["text"])

    # Input box
    user_input = st.chat_input("Ask something about your document...")
    if user_input:
        # Add user query to history
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response using GROQ + Chroma
        with st.spinner("🤖 Thinking..."):
            chunks = retrieve_relevant_chunks(user_input)
            context = "\n\n".join(chunks)
            answer = call_groq_llama3(context, user_input)

        # Add bot response to history
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Auto-scroll script (optional)
        scroll_script = """
        <script>
        const chatContainer = parent.document.querySelector('.stApp .block-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """
        st.components.v1.html(scroll_script, height=0)
