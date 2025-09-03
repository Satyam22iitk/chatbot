import os
import streamlit as st
from dotenv import load_dotenv
from backend import GroqChatClient, basic_reply, DocumentChat

# Load .env for local dev
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
    st.session_state.history = []  # list of (role, text)

# Initialize Groq client only when needed
groq_client = None
if mode != "Basic chatbot" and GROQ_API_KEY:
    groq_client = GroqChatClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)

# File uploader for PDF mode
uploaded_files = None
if mode == "Chatbot with documents (PDF)":
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

# Display conversation history
if st.session_state.history:
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

# Input area
# Input area
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
                st.error("GROQ_API_KEY not set in environment. See .env.example")
            else:
                # Build message list from session history (simple trim to last 12 turns)
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
            if not uploaded_files:
                st.warning("Please upload at least one PDF file for this mode.")
            else:
                docchat = DocumentChat(groq_client=groq_client)
        
                # Save uploaded files temporarily
                import tempfile
                for pf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pf.read())
                        tmp_path = tmp_file.name
                    docchat.add_pdf(tmp_path)
        
                try:
                    answer = docchat.ask(user_input)
                    st.session_state.history.append(("assistant", answer))
                except Exception as e:
                    st.session_state.history.append(("assistant", f"[Error] {e}"))


        # Clear input and refresh UI
        st.session_state.input_counter = 0
        st.rerun()
