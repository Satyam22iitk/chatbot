import streamlit as st
import groq
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize Groq client
def initialize_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
    
    if api_key:
        try:
            return groq.Client(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            return None
    else:
        st.info("Please enter your Groq API key to continue.")
        return None

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Basic chatbot function
def basic_chatbot(client, prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # Using a simpler, faster model
            max_tokens=150,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Context-aware chatbot function
def context_aware_chatbot(client, prompt, chat_history):
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add chat history to context
        for message in chat_history:
            messages.append({"role": "user", "content": message["user"]})
            messages.append({"role": "assistant", "content": message["assistant"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Document-aware chatbot function
def document_aware_chatbot(client, prompt, document_text):
    try:
        # Combine document text with prompt
        enhanced_prompt = f"Based on the following document:\n\n{document_text}\n\nAnswer this question: {prompt}"
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=200,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Main application
def main():
    st.title("🤖 Groq-Powered Chatbot")
    st.write("Select a chatbot mode from the sidebar to get started!")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "mode" not in st.session_state:
        st.session_state.mode = "Basic Chatbot"
    
    # Sidebar for mode selection
    st.sidebar.title("Settings")
    mode = st.sidebar.radio(
        "Select Chatbot Mode:",
        ["Basic Chatbot", "Context-Aware Chatbot", "Document Chatbot"]
    )
    st.session_state.mode = mode
    
    # Initialize Groq client
    client = initialize_groq_client()
    
    # Document upload section (only for document mode)
    if mode == "Document Chatbot":
        st.sidebar.subheader("Upload Document")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                st.session_state.document_text = extract_text_from_pdf(uploaded_file)
            
            if st.session_state.document_text:
                st.sidebar.success("PDF processed successfully!")
                # Show a preview of the document
                with st.sidebar.expander("Document Preview"):
                    st.text(st.session_state.document_text[:500] + "..." if len(st.session_state.document_text) > 500 else st.session_state.document_text)
            else:
                st.sidebar.error("Failed to extract text from PDF.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["assistant"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not client:
            st.error("Please enter a valid Groq API key in the sidebar to continue.")
            return
        
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response based on mode
        with st.spinner("Thinking..."):
            if mode == "Basic Chatbot":
                response = basic_chatbot(client, prompt)
            elif mode == "Context-Aware Chatbot":
                response = context_aware_chatbot(client, prompt, st.session_state.chat_history)
            elif mode == "Document Chatbot":
                if not st.session_state.document_text:
                    response = "Please upload a PDF document first to use this mode."
                else:
                    response = document_aware_chatbot(client, prompt, st.session_state.document_text)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "user": prompt,
            "assistant": response
        })

if __name__ == "__main__":
    main()
