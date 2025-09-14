import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants - Load from environment variables for security
WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
BEARER_TOKEN = os.getenv("N8N_BEARER_TOKEN", "")

def generate_session_id():
    return str(uuid.uuid4())

def send_message_to_llm(session_id, message):
    # Validate that required environment variables are set
    if not WEBHOOK_URL or not BEARER_TOKEN:
        return "Error: Missing required environment variables. Please set N8N_WEBHOOK_URL and N8N_BEARER_TOKEN."
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "sessionId": session_id,
        "chatInput": message
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json().get("output", "No output received")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"

def main():
    st.title("Chat with LLM")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get LLM response
        llm_response = send_message_to_llm(st.session_state.session_id, user_input)

        # Add LLM response to chat history
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
        with st.chat_message("assistant"):
            st.write(llm_response)

if __name__ == "__main__":
    main()