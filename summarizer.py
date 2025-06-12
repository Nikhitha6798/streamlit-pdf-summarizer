import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load models once at startup
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return summarizer, qa_model

summarizer, qa_model = load_pipelines()

# Extract text from PDF
def extract_text(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

st.title("Free PDF Summarizer & Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("Text extracted from PDF!")

    if st.button("Summarize"):
        if len(text) > 1000:
            text = text[:1000]  
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)

    question = st.text_input("Ask a question based on the PDF content:")
    if question:
        answer = qa_model(question=question, context=text[:2000])  # limit for performance
        st.subheader("🤖 Answer")
        st.write(answer['answer'])