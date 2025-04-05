
#importing the important libraries

import streamlit as st
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import numpy as np
import faiss
import os
import tempfile

# Streamlit app title
st.set_page_config(page_title="AI Document Chatbot", layout="wide")
st.title("📄 AI Document Chatbot")

# Initialize session state variables

if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []


st.sidebar.header("🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Asking the users to upload files for real time query
st.sidebar.header("📁 Upload DOCX File")
uploaded_file = st.sidebar.file_uploader("Choose a .docx file", type=["docx"])
if uploaded_file and st.session_state.openai_client:
    with st.spinner("Processing your file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            docx_path = tmp.name
        pdf_path = "temp.pdf"

        
        doc = Document(docx_path)
        pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = [Paragraph(p.text) for p in doc.paragraphs if p.text.strip()]
        pdf.build(story)

        # Pdfplumber to extract text
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        # Chunker and Embedding model
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.chunks = splitter.split_text(text)

        embeddings = st.session_state.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=st.session_state.chunks
        )
        emb_array = np.array([e.embedding for e in embeddings.data]).astype("float32")
        st.session_state.faiss_index = faiss.IndexFlatL2(emb_array.shape[1])
        st.session_state.faiss_index.add(emb_array)

        os.remove(docx_path)
        os.remove(pdf_path)
    st.success("✅ File processed successfully!")

# Retrieve chunks based on query
def retrieve_chunks(query, top_k=3):
    emb = st.session_state.openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding
    emb = np.array([emb]).astype('float32')
    _, idxs = st.session_state.faiss_index.search(emb, top_k)
    return [st.session_state.chunks[i] for i in idxs[0]]

# Generating response
def generate_response(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Based on the context below, answer the query:\n\nContext:\n{context}\n\nQuery: {query}"
    response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Main chat section
st.subheader("💬 Chat Interface")
if st.session_state.faiss_index is None:
    st.info("👈 Upload a DOCX file and provide your API key to start chatting.")
else:
    query = st.chat_input("Ask something about the document...")
    if query:
        with st.spinner("Generating response..."):
            chunks = retrieve_chunks(query)
            answer = generate_response(query, chunks)
            st.session_state.chat_history.append({"query": query, "response": answer})

            # Saving the last 5 sessions 
            if len(st.session_state.chat_history) > 5:
                st.session_state.chat_history.pop(0)

        # Display full chat history
    for chat in st.session_state.chat_history[::-1]:
        with st.chat_message("user"):
            st.markdown(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["response"])

            # Feedback Input using Radio Button
            feedback = st.radio(
                "Was this answer helpful?",
                ["👍 Yes", "👎 No"],
                key=f"fb_{chat['query']}",
                horizontal=True
            )

            feedback_comment = ""
            if feedback == "👎 No":
                feedback_comment = st.text_input(
                    "Tell us what went wrong:",
                    key=f"comment_{chat['query']}"
                )

            # Saving the feedbacks
            if not any(f['query'] == chat['query'] for f in st.session_state.feedback_log):
                st.session_state.feedback_log.append({
                    "query": chat["query"],
                    "response": chat["response"],
                    "feedback": feedback,
                    "comment": feedback_comment
                })

            # Keep only last 5 feedbacks
            st.session_state.feedback_log = st.session_state.feedback_log[-5:]

# Feedback summary display (last 5)
if st.session_state.feedback_log:
    st.sidebar.subheader("📝 Recent Feedback")
    for i, fb in enumerate(st.session_state.feedback_log[::-1]):
        st.sidebar.markdown(f"**{i+1}. Query:** {fb['query']}")
        st.sidebar.markdown(f"- **Feedback:** {fb['feedback']}")
        if fb["comment"]:
            st.sidebar.markdown(f"- **Note:** {fb['comment']}")
        st.sidebar.markdown("---")
