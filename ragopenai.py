import streamlit as st
import fitz  # PyMuPDF for PDF handling
from io import BytesIO
import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

st.title("PDF Summarizer & Q&A with RAG + OpenAI")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully.")

    # Extract PDF text
    text = ""
    with fitz.open(stream=BytesIO(uploaded_file.read()), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    st.subheader("Extracted Text Preview")
    st.write(text[:2000] + "..." if len(text) > 2000 else text)  # Preview first 2k chars

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Setup RetrievalQA chain using OpenAI chat model
    chat = ChatOpenAI(temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

    # Summarize button
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            # Run QA chain with prompt to summarize document
            summary = qa.run("Provide a concise summary of the document.")
            st.markdown("### Summary")
            st.write(summary)

    # Ask a question
    st.markdown("### ❓ Ask a Question About the Document")
    question = st.text_input("Enter your question here:")

    if question:
        with st.spinner("Searching for answer..."):
            answer = qa.run(question)
            st.markdown("### 🗣️ Answer:")
            st.write(answer)
