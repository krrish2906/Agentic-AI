# --------------------- IMPORTS ---------------------
import streamlit as st
from langgraph_backend_sqlite import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ----------------- UTILITY FUNCTIONS -----------------
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# ----------------- SESSION SETUP -----------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ----------------- SIDEBAR -----------------
st.sidebar.title("ChatBot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temporary_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temporary_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temporary_messages


# -------------------- MAIN UI ---------------------
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

    # Get response from chatbot
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        ai_message = st.write_stream(ai_only_stream())
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
