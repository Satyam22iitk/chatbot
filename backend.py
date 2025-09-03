import os
from groq import Groq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from io import BytesIO
import pdfplumber  # for better text extraction


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


class GroqChatClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages: list) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return resp.choices[0].message.content


class DocumentChat:
    """Extracts text from PDFs, chunks it, TF-IDF retrieves relevant pieces, then asks the Groq model."""

    def __init__(self, groq_client: GroqChatClient | None = None, chunk_size: int = 1000):
        self.groq_client = groq_client
        self.chunk_size = chunk_size
        self.chunks = []
        self.vectorizer = None
        self._matrix = None

    def add_pdf(self, file_like):
        """Read text from a PDF (works for Streamlit's UploadedFile)."""
        content = file_like.read()
        reader = PdfReader(BytesIO(content))
        texts = []

        # Try PyPDF2 first
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)

        full_text = "\n".join(texts).strip()

        # If PyPDF2 fails, try pdfplumber
        if not full_text:
            with pdfplumber.open(BytesIO(content)) as pdf:
                texts = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(texts).strip()

        if not full_text:
            full_text = "[Could not extract any text from this PDF]"

        self._chunk_and_store(full_text)

    def _chunk_and_store(self, text: str):
        txt = text.replace("\n", " ")
        for i in range(0, len(txt), self.chunk_size):
            chunk = txt[i: i + self.chunk_size].strip()
            if chunk:
                self.chunks.append(chunk)

        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english").fit(self.chunks)
            self._matrix = self.vectorizer.transform(self.chunks)

    def _retrieve(self, query: str, top_k: int = 3):
        if self._matrix is not None and self.vectorizer is not None:
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
            idxs = np.argsort(sims)[::-1][:top_k]
            return [self.chunks[i] for i in idxs if sims[i] > 0]
        return []

    def ask(self, question: str) -> str:
        context_chunks = self._retrieve(question, top_k=4)
        context = "\n\n".join(context_chunks)

        if not context:
            return "No relevant content found in the uploaded PDFs."

        system_prompt = (
            "You are a helpful assistant. Use the provided document snippets to "
            "answer the user's question. If the answer is not in the snippets, say you don't know."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {question}"},
        ]

        if self.groq_client is None:
            return "[Local fallback] I found these document snippets:\n\n" + context[:2000]

        return self.groq_client.chat(messages)
