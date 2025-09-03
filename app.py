# filename: app_streamlit_hf_inference.py
import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import json
from typing import List

load_dotenv()

# ---------------------------
# Helper: Hugging Face Inference API
# ---------------------------
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{}"

def query_hf_model(repo_id: str, token: str, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7):
    """
    Calls the HF Inference API using requests and normalizes the response.
    Works for many model types (text-generation, text2text).
    """
    url = HF_INFERENCE_URL.format(repo_id)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        # sometimes specifying task helps for models without pipeline tags:
        # "task": "text-generation"
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Network error calling HF Inference API: {e}")

    # If HF returns non-JSON content-type (text/plain) try to decode it
    content_type = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        # try to surface helpful message
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Hugging Face API Error {resp.status_code}: {err}")

    # Response bodies vary by model:
    # - many text-generation models return [{"generated_text": "..."}]
    # - some return plain text
    # - some return {"error": "..."}
    try:
        data = resp.json()
    except ValueError:
        # plain text fallback
        return resp.text.strip()

    # If list and contains generated_text
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        # Some models return 'summary_text' or 'content'
        for key in ("generated_text", "summary_text", "content", "text"):
            if key in data[0]:
                return data[0][key].strip()
        # as fallback, join available keys
        return " ".join(str(v) for v in data[0].values()).strip()

    if isinstance(data, dict):
        # If the model returned a dict with 'generated_text' or 'error'
        if "generated_text" in data:
            return data["generated_text"].strip()
        if "error" in data:
            raise RuntimeError(f"Hugging Face API returned error: {data['error']}")
        # Otherwise stringify
        return json.dumps(data)

    # final fallback
    return str(data)

# ---------------------------
# Streamlit app
# ---------------------------

st.set_page_config(page_title="Multi-Feature Chatbot (HF Inference)", page_icon="🤖", layout="wide")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
if "feature" not in st.session_state:
    st.session_state.feature = "Basic Chatbot"
if "model_name" not in st.session_state:
    st.session_state.model_name = "google/flan-t5-base"

def set_api_key():
    if st.session_state.api_key_input:
        st.session_state.api_key = st.session_state.api_key_input
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.api_key
        st.success("API key set successfully!")

# PDF processing (extract text, chunk, embed, store)
def process_pdf(pdf_files) -> bool:
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {getattr(pdf_file, 'name', 'uploaded file')}: {e}")
            return False

    if not text.strip():
        st.error("No text could be extracted from the PDF(s).")
        return False

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    st.session_state.chunks = chunks

    # embeddings - requires HF token if model is gated
    token = st.session_state.api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=token)
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_processed = True
        return True
    except Exception as e:
        st.error(f"Error creating embeddings / vectorstore: {e}")
        return False

def pdf_question_answer(prompt: str) -> str:
    if not st.session_state.vectorstore:
        return "PDF not processed. Please upload and process PDFs in the sidebar."

    # retrieve top-k
    try:
        docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
    except Exception as e:
        return f"Error during similarity search: {e}"

    # build context from retrieved docs
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    system = (
        "You are a helpful assistant. Use the context below to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know' or provide best-effort with uncertainties.\n\n"
    )
    final_prompt = f"{system}Context:\n{context}\n\nUser: {prompt}\nAssistant:"
    try:
        token = st.session_state.api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        resp = query_hf_model(st.session_state.model_name, token, final_prompt, max_new_tokens=256, temperature=0.2)
        return resp
    except Exception as e:
        return f"Error generating response from HF Inference API: {e}"

def basic_chat(prompt: str) -> str:
    token = st.session_state.api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    try:
        return query_hf_model(st.session_state.model_name, token, prompt, max_new_tokens=256, temperature=0.7)
    except Exception as e:
        return f"Error generating response: {e}"

def context_aware_chat(prompt: str, history: List[dict]) -> str:
    # build conversation history text
    conv = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    full_prompt = f"Conversation so far:\n{conv}\n\nUser: {prompt}\nAssistant:"
    return basic_chat(full_prompt)

# UI
st.title("🤖 Multi-Feature Chatbot (Hugging Face Inference API)")
st.markdown("Use HF Inference endpoint directly to avoid pipeline-tag and text/plain parsing issues.")

with st.sidebar:
    st.header("Configuration")
    st.text_input("Hugging Face API Key", value=st.session_state.api_key, type="password", key="api_key_input", help="Paste your Hugging Face token here.")
    st.button("Set API Key", on_click=set_api_key)

    st.session_state.model_name = st.selectbox(
        "Select Model",
        [
            "google/flan-t5-base",
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill",
            "tiiuae/falcon-7b-instruct",
            "HuggingFaceH4/zephyr-7b-beta"
        ],
        index=0
    )

    st.session_state.feature = st.radio("Feature:", ["Basic Chatbot", "Context-Aware Chatbot", "PDF Document Chatbot"], index=0)

    if st.session_state.feature == "PDF Document Chatbot":
        st.subheader("Upload PDF(s)")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs and building vectorstore..."):
                ok = process_pdf(uploaded_files)
                if ok:
                    st.success("PDFs processed and vectorstore created.")

    st.markdown("---")
    st.info("Note: Hugging Face Inference API has rate limits and model-specific behavior. Keep prompt sizes moderate.")

# show chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not (st.session_state.api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        with st.chat_message("assistant"):
            st.error("Please set your Hugging Face API key first in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                if st.session_state.feature == "Basic Chatbot":
                    reply = basic_chat(prompt)
                elif st.session_state.feature == "Context-Aware Chatbot":
                    history = st.session_state.messages[:-1]
                    reply = context_aware_chat(prompt, history)
                else:  # PDF
                    reply = pdf_question_answer(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
