import streamlit as st
from RAG_script import query_rag_system, load_vectorstore
from dotenv import load_dotenv
import os
load_dotenv()
# Page config
st.set_page_config(page_title="AI Support Assistant", layout="wide")

st.title("Customer Support AI Assistant")

# Load vectorstore once
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

vectorstore = get_vectorstore()

# User input
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):

        result = query_rag_system(user_query, vectorstore, k=3, verbose=False)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Confidence")
        st.write(result["confidence"])

        # Expandable debug info
        with st.expander("Show retrieved chunks"):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                st.write(f"Chunk {i} (Score: {chunk['score']:.4f})")
                st.write(chunk["content"])
                st.markdown("---")