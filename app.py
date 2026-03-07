import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from models.llm import get_chat_model
from utils.rag_utils import get_rag_answer
from utils.web_search import web_search

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    if uploaded_file.type == "application/pdf":
        import fitz
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in pdf_document])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def get_chat_response(chat_model, messages, system_prompt):
    formatted_messages = [SystemMessage(content=system_prompt)]

    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        else:
            formatted_messages.append(AIMessage(content=msg["content"]))

    response = chat_model.invoke(formatted_messages)
    return response.content


def instructions_page():
    st.title("Customer Support Chatbot")
    st.markdown("""
    ## How to Use
    1. Go to Chat page
    2. Upload PDF/TXT knowledge base (optional)
    3. Ask questions
    """)


def chat_page():
    st.title("🤖 Customer Support Chatbot")

    provider = st.sidebar.selectbox(
        "Select AI Provider",
        ["Groq (Default)", "OpenAI", "Google Gemini"],
        index=0
    )
    response_mode = st.sidebar.radio("Response Mode", ["Concise", "Detailed"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload knowledge base (PDF/TXT)", type=["pdf", "txt"]
    )

    if uploaded_file:
        st.session_state.vectorstore = process_uploaded_file(uploaded_file)
        st.sidebar.success(f"Indexed: {uploaded_file.name}")

    if provider.startswith("OpenAI"):
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not set")
            return
        chat_model = get_chat_model("openai")

    elif provider.startswith("Google Gemini"):
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not set")
            return
        chat_model = get_chat_model("gemini")

    else:
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not set")
            return
        chat_model = get_chat_model("groq")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        context = ""

        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(prompt)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
        else:
            context = get_rag_answer(prompt) or web_search(prompt)

        system_prompt = f"""
        You are a helpful customer support assistant.
        Use the context below to answer the question.

        Context:
        {context}

        Answer in {'short sentences' if response_mode == 'Concise' else 'detailed steps'}.
        """

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="Customer Support Chatbot", layout="wide")

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Chat", "Instructions"])

        if page == "Chat" and st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.rerun()

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
