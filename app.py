import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Multi-LLM imports
try:
    from langchain_groq import ChatGroq, GroqEmbeddings
    has_groq = True
except ImportError:
    has_groq = False

from models.llm import get_chat_model
from utils.rag_utils import get_rag_answer
from utils.web_search import web_search

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------
# Helper Functions
# ---------------------------

def process_uploaded_file(uploaded_file):
    """Process uploaded PDF/TXT file and create a FAISS vectorstore"""
    if uploaded_file is None:
        return None

    # Extract text
    if uploaded_file.type == "application/pdf":
        import fitz
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in pdf_document])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_chat_response(chat_model, messages, system_prompt):
    """Generate response from the selected chat model"""
    formatted_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        else:
            formatted_messages.append(AIMessage(content=msg["content"]))
    try:
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}"


# ---------------------------
# Streamlit Pages
# ---------------------------

def instructions_page():
    st.title("Customer Support Chatbot")
    st.markdown("Follow these instructions to set up and use the chatbot:")
    st.markdown("""
    ## Installation
    ```bash
    pip install -r requirements.txt
    ```
    ## API Key Setup
    Set your API keys in Streamlit Secrets or `config/config.py`:
    - `OPENAI_API_KEY`
    - `GEMINI_API_KEY`
    - `GROQ_API_KEY`
    
    ## How to Use
    1. Go to the **Chat** page.
    2. Upload a PDF/TXT knowledge base if needed.
    3. Start chatting!
    """)


def chat_page():
    st.title("🤖 Customer Support Chatbot")

    # Select provider
    provider = st.sidebar.selectbox(
        "Select AI Provider",
        ["Groq (Default)", "OpenAI", "Google Gemini"],
        index=0
    )
    response_mode = st.sidebar.radio("Response Mode:", ["Concise", "Detailed"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Upload file
    uploaded_file = st.sidebar.file_uploader(
        "Upload knowledge base (PDF/TXT/JSON)", type=["pdf", "txt", "json"]
    )
    if uploaded_file:
        st.session_state.vectorstore = process_uploaded_file(uploaded_file)
        st.sidebar.success(f"Indexed: {uploaded_file.name}")

    # Initialize chat model
    try:
        if provider.startswith("OpenAI"):
            if not OPENAI_API_KEY:
                st.error("❌ OPENAI_API_KEY not set.")
                return
            chat_model = get_chat_model("openai")
        elif provider.startswith("Google Gemini"):
            if not GEMINI_API_KEY:
                st.error("❌ GEMINI_API_KEY not set.")
                return
            chat_model = get_chat_model("gemini")
        else:  # Groq
            if not GROQ_API_KEY:
                st.error("❌ GROQ_API_KEY not set.")
                return
            chat_model = get_chat_model("groq")
    except Exception as e:
        st.error(str(e))
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build context
        context = ""
        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(prompt)
            context = "\n".join([d.page_content for d in retrieved_docs])
        else:
            context = get_rag_answer(prompt) or web_search(prompt)

        system_prompt = f"""
        You are a helpful customer support assistant.
        Use the following context to answer the user question:
        {context}
        Answer in {'short sentences' if response_mode=='Concise' else 'detailed steps'}.
        """

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(page_title="Customer Support Chatbot", layout="wide")
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.vectorstore = None
                st.rerun()
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
