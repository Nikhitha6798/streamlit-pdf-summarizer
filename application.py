
import streamlit as st
import fitz # PyMuPDF for PDF handling
from io import BytesIO
import openai 
import os

from dotenv import load_dotenv 
load_dotenv()

# Your OpenAI key 
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

st.title("📄 PDF Summarizer & Q&A")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully.")

    # Extract PDF text
    text = ""
    with fitz.open(stream=BytesIO(uploaded_file.read()), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    st.subheader("📖 Extracted Text Preview")
    st.write(text)

    # 🔘 Summarize Button
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You summarize documents."},
                    {"role": "user", "content": f"Summarize this document:\n{text}"}
                ],
                max_tokens=500
            )
        
            summary = response.choices[0].message.content.strip()
            st.markdown("### 📌 Summary")
            st.write(summary)

    # ❓ Question Input
    st.markdown("### ❓ Ask a Question About the Document")
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Answering..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using the document only."},
                    {"role": "user", "content": f"Document:\n{text}\n\nQuestion: {question}"}
                ],
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()
            st.markdown("### 🗣️ Answer:")
            st.write(answer)
