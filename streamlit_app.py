import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from io import BytesIO

# --------- Basic chatbot (rule-based) ---------
def basic_reply(text: str) -> str:
    """Super-simple local chatbot to avoid any API calls."""
    t = (text or "").lower()
    if "hello" in t or "hi" in t:
        return "Hello! How can I help you today?"
    if "help" in t:
        return "I can answer questions, or you can switch to the Groq-powered mode for smarter replies."
    if "thanks" in t or "thank you" in t:
        return "You're welcome! 😊"
    return "Sorry, I only know a few canned replies. Try the Groq-powered mode for better answers."


# --------- Groq client wrapper ---------
class GroqChatClient:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages: list) -> str:
        """
        messages: list[{'role': 'system'|'user'|'assistant', 'content': str}]
        Returns assistant text.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return resp.choices[0].message.content


# --------- PDF RAG (TF-IDF) + Groq ---------
class DocumentChat:
    """
    Extracts text from PDFs, chunks it, TF-IDF retrieves relevant pieces,
    then asks the Groq model with the retrieved context.
    """

    def __init__(self, groq_client: GroqChatClient | None = None, chunk_size: int = 1000):
        self.groq_client = groq_client
        self.chunk_size = chunk_size
        self.chunks = []
        self.vectorizer = None
        self._matrix = None

    def add_pdf(self, file_like):
        """
        file_like: a stream-like object (e.g., Streamlit UploadedFile)
        """
        content = file_like.read()
        reader = PdfReader(BytesIO(content))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        full_text = "\n".join(texts)
        self._chunk_and_store(full_text)

    def _chunk_and_store(self, text: str):
        txt = text.replace("\n", " ")
        for i in range(0, len(txt), self.chunk_size):
            chunk = txt[i: i + self.chunk_size].strip()
            if chunk:
                self.chunks.append(chunk)

        # build / rebuild vectorizer & matrix
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english").fit(self.chunks)
            self._matrix = self.vectorizer.transform(self.chunks)

    def _retrieve(self,_

# Load .env for local dev
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

st.set_page_config(page_title="Groq Chatbot", layout="centered")
st.title("📚 Groq Chatbot — 3 Modes")

mode = st.sidebar.radio(
    "Choose mode",
    ["Basic chatbot", "Chatbot aware (Groq)", "Chatbot with documents (PDF)"]
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

# Initialize Groq client only when needed
groq_client = None
if mode != "Basic chatbot" and GROQ_API_KEY:
    groq_client = GroqChatClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)

# File uploader for PDF mode
uploaded_files = None
if mode == "Chatbot with documents (PDF)":
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

# Display conversation history
if st.session_state.history:
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

# Input area
user_input = st.text_input("Your message", key="input")

send = st.button("Send")
if send:
    if not user_input:
        st.warning("Please type a message.")
    else:
        st.session_state.history.append(("user", user_input))

        if mode == "Basic chatbot":
            reply = basic_reply(user_input)
            st.session_state.history.append(("assistant", reply))

        elif mode == "Chatbot aware (Groq)":
            if not groq_client:
                st.error("GROQ_API_KEY not set in environment. See .env.example")
            else:
                # Build message list from session history (simple trim to last 12 turns)
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                for r, m in st.session_state.history[-12:]:
                    role = "user" if r == "user" else "assistant"
                    messages.append({"role": role, "content": m})

                try:
                    resp = groq_client.chat(messages)
                    st.session_state.history.append(("assistant", resp))
                except Exception as e:
                    st.session_state.history.append(("assistant", f"[Error calling Groq] {e}"))

        elif mode == "Chatbot with documents (PDF)":
            if not uploaded_files:
                st.warning("Please upload at least one PDF file for this mode.")
            else:
                docchat = DocumentChat(groq_client=groq_client)
                for pf in uploaded_files:
                    docchat.add_pdf(pf)
                try:
                    answer = docchat.ask(user_input)
                    st.session_state.history.append(("assistant", answer))
                except Exception as e:
                    st.session_state.history.append(("assistant", f"[Error] {e}"))

        # Clear input and refresh UI
        st.session_state.input = ""
        st.rerun()
