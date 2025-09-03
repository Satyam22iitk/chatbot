import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Multi-Feature Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "feature" not in st.session_state:
    st.session_state.feature = "basic"

# Set Hugging Face API key
def set_api_key():
    if st.session_state.api_key_input:
        st.session_state.api_key = st.session_state.api_key_input
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.api_key
        st.success("API key set successfully!")

# Process uploaded PDF files
def process_pdf(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    # Create conversation chain
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",   # smaller, stable model
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    st.session_state.pdf_processed = True

# Basic chatbot function
def basic_chatbot(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            st.error(f"Hugging Face API Error {response.status_code}: {response.text[:200]}")
            return {}
        try:
            return response.json()
        except Exception:
            st.error("Response was not valid JSON: " + response.text[:200])
            return {}
    
    output = query({"inputs": prompt})
    if isinstance(output, list) and len(output) > 0:
        return output[0].get("generated_text", "No response generated.")
    elif isinstance(output, dict):
        return output.get("generated_text", "No response generated.")
    return "Sorry, I could not process that request."

# Context-aware chatbot function
def context_aware_chatbot(prompt, conversation_history):
    full_prompt = f"Conversation history:\n{conversation_history}\n\nUser: {prompt}\nAssistant:"
    
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            st.error(f"Hugging Face API Error {response.status_code}: {response.text[:200]}")
            return {}
        try:
            return response.json()
        except Exception:
            st.error("Response was not valid JSON: " + response.text[:200])
            return {}
    
    output = query({
        "inputs": full_prompt,
        "parameters": {
            "max_length": 1000,
            "temperature": 0.7,
            "do_sample": True
        }
    })
    
    if isinstance(output, list) and len(output) > 0:
        return output[0].get("generated_text", "No response generated.")
    elif isinstance(output, dict):
        return output.get("generated_text", "No response generated.")
    return "Sorry, I could not process that request."

# PDF chatbot function
def pdf_chatbot(prompt):
    if st.session_state.conversation_chain:
        response = st.session_state.conversation_chain({"question": prompt})
        return response["answer"]
    else:
        return "PDF processing not completed. Please upload and process PDF files first."

# Main application
def main():
    st.title("🤖 Multi-Feature Chatbot")
    st.markdown("Chatbot with three features: Basic, Context-Aware, and PDF Document Chat")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.text_input("Hugging Face API Key", type="password", key="api_key_input")
        st.button("Set API Key", on_click=set_api_key)
        
        st.session_state.feature = st.radio(
            "Select Chatbot Feature:",
            ["Basic Chatbot", "Context-Aware Chatbot", "PDF Document Chatbot"],
            index=0
        )
        
        if st.session_state.feature == "PDF Document Chatbot":
            st.subheader("Upload PDF Documents")
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            if uploaded_files and st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    process_pdf(uploaded_files)
                    st.success("PDFs processed successfully!")
        
        st.markdown("---")
        st.info("""
        **Features:**
        - **Basic Chatbot:** Simple question-answering
        - **Context-Aware Chatbot:** Remembers conversation history
        - **PDF Document Chatbot:** Answers questions from uploaded PDFs
        """)

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if not st.session_state.api_key:
            with st.chat_message("assistant"):
                st.error("Please set your Hugging Face API key in the sidebar to use the chatbot.")
            return
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.feature == "Basic Chatbot":
                    response = basic_chatbot(prompt)
                elif st.session_state.feature == "Context-Aware Chatbot":
                    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])
                    response = context_aware_chatbot(prompt, history)
                else:
                    response = pdf_chatbot(prompt) if st.session_state.pdf_processed else "Please upload and process PDF files first."
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
