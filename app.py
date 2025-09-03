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
import json

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
        repo_id="google/flan-t5-xxl",
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
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    output = query({
        "inputs": prompt,
    })
    
    return output[0]['generated_text'] if isinstance(output, list) else output.get('generated_text', 'Sorry, I could not process that request.')

# Context-aware chatbot function
def context_aware_chatbot(prompt, conversation_history):
    # Combine conversation history with new prompt
    full_prompt = f"Conversation history:\n{conversation_history}\n\nUser: {prompt}\nAssistant:"
    
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    output = query({
        "inputs": full_prompt,
        "parameters": {
            "max_length": 1000,
            "temperature": 0.7,
            "do_sample": True
        }
    })
    
    return output[0]['generated_text'] if isinstance(output, list) else output.get('generated_text', 'Sorry, I could not process that request.')

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
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        st.text_input(
            "Hugging Face API Key",
            type="password",
            key="api_key_input",
            help="Enter your Hugging Face API key to use the chatbot features"
        )
        st.button("Set API Key", on_click=set_api_key)
        
        # Feature selection
        st.session_state.feature = st.radio(
            "Select Chatbot Feature:",
            ["Basic Chatbot", "Context-Aware Chatbot", "PDF Document Chatbot"],
            index=0
        )
        
        # PDF upload for document chatbot
        if st.session_state.feature == "PDF Document Chatbot":
            st.subheader("Upload PDF Documents")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True
            )
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
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if API key is set
        if not st.session_state.api_key:
            with st.chat_message("assistant"):
                st.error("Please set your Hugging Face API key in the sidebar to use the chatbot.")
            return
        
        # Generate assistant response based on selected feature
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.feature == "Basic Chatbot":
                    response = basic_chatbot(prompt)
                elif st.session_state.feature == "Context-Aware Chatbot":
                    # Extract conversation history
                    conversation_history = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]]
                    )
                    response = context_aware_chatbot(prompt, conversation_history)
                else:  # PDF Document Chatbot
                    if not st.session_state.pdf_processed:
                        response = "Please upload and process PDF files first using the sidebar."
                    else:
                        response = pdf_chatbot(prompt)
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()