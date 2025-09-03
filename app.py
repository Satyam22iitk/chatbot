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
    st.session_state.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
if "feature" not in st.session_state:
    st.session_state.feature = "basic"
if "model_name" not in st.session_state:
    st.session_state.model_name = "google/flan-t5-base"

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
            text += page.extract_text() or ""  # Handle None return
    
    if not text.strip():
        st.error("No text could be extracted from the PDF(s).")
        return False
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    try:
        # Create embeddings
        embeddings = HuggingFaceEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        # Create conversation chain
        llm = HuggingFaceHub(
            repo_id=st.session_state.model_name,
            model_kwargs={"temperature": 0.5, "max_length": 512}
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

# Generic chatbot function using HuggingFaceHub
def huggingface_api_chatbot(prompt, conversation_history=None):
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{st.session_state.model_name}"
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}

        if conversation_history:
            full_prompt = f"Conversation history:\n{conversation_history}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt

        payload = {"inputs": full_prompt}
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return f"HF API Error {response.status_code}: {response.text[:200]}"

        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "No response generated.")
            elif isinstance(data, dict):
                return data.get("generated_text", "No response generated.")
            else:
                return str(data)
        except Exception:
            return response.text[:500]  # fallback if plain text
    except Exception as e:
        return f"Error generating response: {str(e)}"



# PDF chatbot function
def pdf_chatbot(prompt):
    if st.session_state.conversation_chain:
        try:
            response = st.session_state.conversation_chain({"question": prompt})
            return response["answer"]
        except Exception as e:
            return f"Error generating response: {str(e)}"
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
            value=st.session_state.api_key,
            type="password",
            key="api_key_input",
            help="Enter your Hugging Face API key to use the chatbot features"
        )
        st.button("Set API Key", on_click=set_api_key)
        
        # Model selection
        st.session_state.model_name = st.selectbox(
            "Select Model",
            [
                "google/flan-t5-base",
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
                "tiiuae/falcon-7b-instruct",
                "HuggingFaceH4/zephyr-7b-beta"
            ],
            index=0,
            help="Select a model compatible with Hugging Face Hub"
        )
        
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
                    if process_pdf(uploaded_files):
                        st.success("PDFs processed successfully!")
        
        st.markdown("---")
        st.info("""
        **Features:**
        - **Basic Chatbot:** Simple question-answering
        - **Context-Aware Chatbot:** Remembers conversation history
        - **PDF Document Chatbot:** Answers questions from uploaded PDFs
        
        **Note:** Some models may take longer to respond or may not be available in all regions.
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
                    response = huggingface_hub_chatbot(prompt)
                elif st.session_state.feature == "Context-Aware Chatbot":
                    # Extract conversation history
                    conversation_history = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]]
                    )
                    response = huggingface_hub_chatbot(prompt, conversation_history)
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
