import os
import streamlit as st
import requests
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import tempfile
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Multi-Feature Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
def init_session_state():
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
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ""
    if "available_models" not in st.session_state:
        st.session_state.available_models = {
            "Basic Chatbot": "google/flan-t5-base",
            "Context-Aware Chatbot": "microsoft/DialoGPT-medium",
            "PDF Document Chatbot": "google/flan-t5-base"
        }

init_session_state()

# Set Hugging Face API key
def set_api_key():
    if st.session_state.api_key_input:
        st.session_state.api_key = st.session_state.api_key_input
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.api_key
        st.success("API key set successfully!")
        
        # Test the API key
        test_api_key()

# Test the API key
def test_api_key():
    try:
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
        response = requests.get("https://huggingface.co/api/models", headers=headers)
        if response.status_code == 200:
            st.sidebar.success("API key is valid!")
        else:
            st.sidebar.error(f"API key validation failed: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Error testing API key: {str(e)}")

# Process uploaded PDF files
def process_pdf(pdf_files):
    try:
        text = ""
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        if not text.strip():
            st.error("No text could be extracted from the PDF files. Please try different files.")
            return False
        
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
            repo_id=st.session_state.available_models["PDF Document Chatbot"],
            model_kwargs={"temperature": 0.5, "max_length": 512},
            huggingfacehub_api_token=st.session_state.api_key
        )
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        st.session_state.pdf_processed = True
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

# Basic chatbot function
def basic_chatbot(prompt):
    try:
        model_name = st.session_state.available_models["Basic Chatbot"]
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        # Handle different response formats
        if response.status_code == 200:
            try:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    else:
                        return str(result[0])
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text']
                else:
                    return str(result)
            except json.JSONDecodeError:
                return response.text
        elif response.status_code == 404:
            return f"Model not found (404). Please try a different model in the settings."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Context-aware chatbot function
def context_aware_chatbot(prompt):
    try:
        # Update conversation history
        st.session_state.conversation_history += f"User: {prompt}\n"
        
        model_name = st.session_state.available_models["Context-Aware Chatbot"]
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
        
        # Use the conversation history as context
        full_prompt = f"{st.session_state.conversation_history}Assistant:"
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json={
                "inputs": full_prompt,
                "parameters": {
                    "max_length": 1000,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        assistant_response = result[0]['generated_text']
                    else:
                        assistant_response = str(result[0])
                elif isinstance(result, dict) and 'generated_text' in result:
                    assistant_response = result['generated_text']
                else:
                    assistant_response = str(result)
                
                # Extract only the new response (remove the history)
                if "Assistant:" in assistant_response:
                    assistant_response = assistant_response.split("Assistant:")[-1].strip()
                
                # Update conversation history
                st.session_state.conversation_history += f"Assistant: {assistant_response}\n"
                
                return assistant_response
            except json.JSONDecodeError:
                return response.text
        elif response.status_code == 404:
            return f"Model not found (404). Please try a different model in the settings."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# PDF chatbot function
def pdf_chatbot(prompt):
    if not st.session_state.pdf_processed:
        return "PDF processing not completed. Please upload and process PDF files first."
    
    try:
        response = st.session_state.conversation_chain({"question": prompt})
        return response.get("answer", "No answer could be generated.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Clear conversation history
def clear_chat():
    st.session_state.messages = []
    st.session_state.conversation_history = ""

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
            help="Enter your Hugging Face API key to use the chatbot features",
            value=st.session_state.api_key
        )
        st.button("Set API Key", on_click=set_api_key)
        
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Selection")
        st.session_state.available_models["Basic Chatbot"] = st.text_input(
            "Basic Chatbot Model",
            value=st.session_state.available_models["Basic Chatbot"],
            help="Enter the model ID for the basic chatbot (e.g., google/flan-t5-base)"
        )
        
        st.session_state.available_models["Context-Aware Chatbot"] = st.text_input(
            "Context-Aware Chatbot Model",
            value=st.session_state.available_models["Context-Aware Chatbot"],
            help="Enter the model ID for the context-aware chatbot (e.g., microsoft/DialoGPT-medium)"
        )
        
        st.session_state.available_models["PDF Document Chatbot"] = st.text_input(
            "PDF Chatbot Model",
            value=st.session_state.available_models["PDF Document Chatbot"],
            help="Enter the model ID for the PDF chatbot (e.g., google/flan-t5-base)"
        )
        
        st.markdown("---")
        
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
            if uploaded_files:
                if st.button("Process PDFs"):
                    with st.spinner("Processing PDFs..."):
                        if process_pdf(uploaded_files):
                            st.success("PDFs processed successfully!")
            
            if st.session_state.pdf_processed:
                st.success("PDF documents are ready for questioning!")
        
        st.markdown("---")
        st.button("Clear Chat History", on_click=clear_chat)
        
        st.markdown("---")
        st.info("""
        **Features:**
        - **Basic Chatbot:** Simple question-answering
        - **Context-Aware Chatbot:** Remembers conversation history
        - **PDF Document Chatbot:** Answers questions from uploaded PDFs
        
        **Note:** You need a Hugging Face API key to use these features.
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
                    response = context_aware_chatbot(prompt)
                else:  # PDF Document Chatbot
                    response = pdf_chatbot(prompt)
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
