import os
import streamlit as st
from dotenv import load_dotenv
from backend import GroqChatClient, basic_reply, DocumentChat

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

st.set_page_config(page_title="Groq Chatbot", layout="centered")
st.title("📚 Groq Chatbot — 3 Modes")

mode = st.sidebar.radio(
    "Choose mode",
    ["Basic chatbot", "Chatbot aware (Groq)", "Chatbot with documents (PDF)"]
)

if "history" not in st.session_state:
    st.session_state.history = []

if "docchat" not in st.session_state:
    st.session_state.docchat = None  # persist DocumentChat across turns

groq_client = None
if mode != "Basic chatbot" and GROQ_API_KEY:
    groq_client = GroqChatClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)

uploaded_files = None
if mode == "Chatbot with documents (PDF)":
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.session_state.docchat is None:
        st.session_state.docchat = DocumentChat(groq_client=groq_client)
        for pf in uploaded_files:
            st.session_state.docchat.add_pdf(pf)

# Show chat history
if st.session_state.history:
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

user_input = st.text_input("Your message", key="user_message")
send = st.button("Send")

if send:
    if not user_input:
        st.warning("Please type a message.")
    else:
        st.session_state.history.append(("user", user_input))

        if mode == "Basic chatbot":
            reply = basic_reply(user_input)
            st.session_state.history.append(("assistant", reply))

        elif mode == "Chatbot aware (Groq)":
            if not groq_client:
                st.error("GROQ_API_KEY not set in environment.")
            else:
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                for r, m in st.session_state.history[-12:]:
                    role = "user" if r == "user" else "assistant"
                    messages.append({"role": role, "content": m})
                try:
                    resp = groq_client.chat(messages)
                    st.session_state.history.append(("assistant", resp))
                except Exception as e:
                    st.session_state.history.append(("assistant", f"[Error calling Groq] {e}"))

        elif mode == "Chatbot with documents (PDF)":
            if not uploaded_files and not st.session_state.docchat:
                st.warning("Please upload at least one PDF file.")
            else:
                try:
                    answer = st.session_state.docchat.ask(user_input)
                    st.session_state.history.append(("assistant", answer))
                except Exception as e:
                    st.session_state.history.append(("assistant", f"[Error] {e}"))

        st.rerun()
