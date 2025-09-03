import os
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

    def _retrieve(self, query: str, top_k: int = 3):
        if self._matrix is not None and self.vectorizer is not None:
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
            idxs = np.argsort(sims)[::-1][:top_k]
            return [self.chunks[i] for i in idxs if sims[i] > 0]
        return []

    def ask(self, question: str) -> str:
        # Retrieve top chunks
        context_chunks = self._retrieve(question, top_k=4)
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a helpful assistant. Use the provided document snippets to "
            "answer the user's question. If the answer is not in the snippets, say you don't know."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {question}"},
        ]

        if self.groq_client is None:
            # Local safety fallback
            if context:
                return "[Local fallback] I found these document snippets:\n\n" + context[:2000]
            return "[Local fallback] No Groq client configured and no documents found."

        # Call Groq
        return self.groq_client.chat(messages)
