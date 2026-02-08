import streamlit as st
from connect_memory_with_llm import build_qa_chain

st.set_page_config(
    page_title="Medical Chatbot 🩺",
    page_icon="🩺",
)

st.title("🩺 Medical Chatbot")
st.caption("Get guidance for your health")

@st.cache_resource
def load_chain():
    return build_qa_chain()

qa_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(user_input)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

st.warning(
    "⚠️ This chatbot is for educational purposes only and not a substitute for professional medical advice."
)
