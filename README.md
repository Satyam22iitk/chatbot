# Multi-Feature Chatbot

A Streamlit-based chatbot application with three different features powered by Hugging Face models.

## Features

1. **Basic Chatbot**: Simple question-answering using the FLAN-T5 model
2. **Context-Aware Chatbot**: Remembers conversation history using DialoGPT
3. **PDF Document Chatbot**: Processes and answers questions from uploaded PDF documents

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Get a Hugging Face API key from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Run the application: `streamlit run app.py`

## Usage

1. Enter your Hugging Face API key in the sidebar
2. Select the chatbot feature you want to use
3. For the PDF Document Chatbot, upload and process your PDF files first
4. Start chatting with the bot!

## Deployment

To deploy on Streamlit Community Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Add your Hugging Face API key as a secret in the advanced settings

## File Structure
