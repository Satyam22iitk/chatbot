import os

self.chunks = []
self.vectorizer = None
self._matrix = None

def add_pdf(self, file_like):
    # file_like is a stream (UploadedFile from Streamlit)
    content = file_like.read()
    reader = PdfReader(BytesIO(content))
    full = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        full.append(t)
    text = "\n".join(full)
    self._chunk_and_store(text)

def _chunk_and_store(self, text: str):
    text = text.replace("\n", " ")
    # naive chunking by characters
    for i in range(0, len(text), self.chunk_size):
        chunk = text[i : i + self.chunk_size].strip()
        if chunk:
            self.chunks.append(chunk)
    # (re)build vectorizer
    if self.chunks:
        self.vectorizer = TfidfVectorizer(stop_words="english").fit(self.chunks)
        self._matrix = self.vectorizer.transform(self.chunks)

def _retrieve(self, query: str, top_k: int = 3):
    if not self._matrix is None and self.vectorizer:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in idxs if sims[i] > 0]
    return []

def ask(self, question: str) -> str:
    # retrieve
    context_chunks = self._retrieve(question, top_k=4)
    context = "\n\n".join(context_chunks)

    system_prompt = "You are a helpful assistant. Use the provided document snippets to answer the user's question. If the answer is not in the snippets, say you don't know."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {question}"},
    ]

    if self.groq_client is None:
        # fallback: reply that Groq client not available and show retrieved text
        if context:
            return "[Local fallback] I found these document snippets:\n\n" + context[:2000]
        return "[Local fallback] No Groq client configured and no documents found."

    # call Groq
    return self.groq_client.chat(messages)
