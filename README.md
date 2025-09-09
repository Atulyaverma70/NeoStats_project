# 🤖 Customer Support Chatbot (RAG + Multi-Provider)

An AI-powered customer support chatbot built with **Streamlit** and **LangChain**.  
It can:

- Answer questions directly using pre-loaded knowledge (FAQs, manuals, troubleshooting docs).
- Accept uploaded PDF/TXT/JSON files and index them on the fly.
- Choose between **Groq (Llama-3.1)**, **OpenAI GPT-3.5**, or **Google Gemini** models.
- Fall back to Bing web search if no internal answer is found.

---

## 🚀 Features

- **Retrieval-Augmented Generation (RAG)**: pulls from `knowledge_docs` or uploaded files.
- **Multi-model Support**: Groq (default), OpenAI, Gemini.
- **Embeddings**: uses GroqEmbeddings if available; falls back to OpenAIEmbeddings.
- **Vector Store**: FAISS for fast similarity search.
- **Web Search**: optional Bing Search integration.
- **Streamlit UI**: chat history, file upload, provider selector.

---

## 📂 Project Structure

.
├── app.py # Streamlit main app
├── models/
│ └── llm.py # Functions to get Groq/OpenAI/Gemini chat models
├── utils/
│ ├── rag_utils.py # Load knowledge_docs & retrieve answers
│ └── web_search.py # Bing search fallback
├── embeddings.py # Precompute embeddings for local docs
├── config/
│ └── config.py # API keys here
├── knowledge_docs/ # Your manuals, FAQs, troubleshooting docs
│ ├── manuals/
│ │ └── user_manual.pdf
│ ├── faqs/
│ │ ├── faqs.txt
│ │ └── faq.json
│ └── troubleshooting/
│ └── troubleshooting.txt
└── requirements.txt

yaml
Copy code

---

## 🔑 Setup API Keys

Edit `config/config.py` and set:

```python
OPENAI_API_KEY = "your_openai_key"
GROQ_API_KEY = "your_groq_key"
GEMINI_API_KEY = "your_gemini_key"
BING_API_KEY = "your_bing_key"  # optional, for web_search
🛠 Installation
bash
Copy code
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
If you plan to use GroqEmbeddings, install its package:

bash
Copy code
pip install langchain_groq
▶️ Running the App
bash
Copy code
streamlit run app.py
Then open the provided localhost URL in your browser.

📝 Usage
1. Direct Knowledge Base (pre-loaded)
Put your manuals, FAQs, troubleshooting files under knowledge_docs.

The chatbot automatically loads these and can answer directly.

2. Upload Your Own File
In the sidebar, upload a PDF/TXT/JSON file.

The file will be chunked, embedded (Groq/OpenAI), and indexed on-the-fly.

The chatbot will answer using the new knowledge base.

3. Model Selection
Sidebar dropdown lets you choose Groq (default), OpenAI, or Google Gemini.

Responses can be “Concise” or “Detailed.”

4. Web Search Fallback
If no internal context found, Bing search snippets are used.

📜 Precomputing Embeddings (Optional)
You can precompute all knowledge_docs embeddings and store them in a pickle:

bash
Copy code
python embeddings.py
This creates vector_store.pkl which can be loaded for faster startup.

⚙️ How It Works (Flow)
app.py launches Streamlit UI.

User selects model & uploads file (or uses pre-loaded docs).

process_uploaded_file() → splits & embeds docs → stores in FAISS.

Chat input → builds a context (from FAISS / rag_utils / web_search).

get_chat_model() in models/llm.py returns correct model wrapper.

get_chat_response() sends system+user messages to model → returns answer.

Streamlit displays the conversation.
