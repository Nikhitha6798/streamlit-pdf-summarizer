# 📄 PDF Summarizer & Q&A with LLMs

This project is an interactive Streamlit application that allows users to upload PDF files, extract their content, and either summarize the text or ask questions based on the document. It supports different backends, including **OpenAI GPT**, **HuggingFace Transformers**, and **RAG (Retrieval-Augmented Generation)** with OpenAI.

---

## 🚀 Features

- Upload a PDF and view the extracted content.
- Summarize documents using different models.
- Ask context-aware questions based on the PDF.
- Multiple backend options:
  - OpenAI GPT (LLM)
  - HuggingFace Transformers
  - Retrieval-Augmented Generation (RAG) with OpenAI

---

## 🧠 Model Variants & File Roles

| File Name        | Model Used               | Description                                                                 |
|------------------|--------------------------|-----------------------------------------------------------------------------|
| `application.py` | 💡 OpenAI GPT (LLM)      | Streamlit app using OpenAI's GPT API for summarizing and Q&A.              |
| `summarizer.py`  | 🤗 HuggingFace Transformers | Uses local or hosted transformer models to generate summaries.             |
| `ragopenai.py`   | 🔁 OpenAI + RAG          | Combines OpenAI's embeddings with vector retrieval (RAG) for contextual Q&A.|

Each script reflects a different approach to NLP-based summarization and question-answering, giving you flexibility to compare performance and accuracy.

---

## 📂 File Structure

streamlit_pdf_summarizer/
│
├── application.py # OpenAI GPT-based summarization and Q&A
├── summarizer.py # HuggingFace Transformers summarization
├── ragopenai.py # OpenAI-based RAG pipeline
├── utils/ # Helper functions (optional)
├── requirements.txt # Python dependencies
├── .env # Environment variables (NOT committed to Git)
└── README.md # You're here!

---


## 🔐 Environment Setup

1. **Create a `.env` file** (keep it private):
   ```env
   OPENAI_API_KEY=your_openai_key
   OPENAI_ORG_ID=your_org_id

  OR 

  If you want to set environment variables temporarily for your terminal session (not recommended for permanent usage), you can run:
 
  export OPENAI_API_KEY="xxx"
  
  export OPENAI_ORG_ID="xxx"
   
  This sets environment variables only for that terminal session.

---

2. Install Dependencies
   
   pip install -r requirements.txt
   
---

3. Run the application
   streamlit run application.py (or)
   streamlit run summarizer.py (or)
   streamlit run ragopenai.py

---

### 🧪 Technologies Used

Python

Streamlit

OpenAI GPT-3.5

HuggingFace Transformers

PyMuPDF (for PDF text extraction)

RAG (Vector Search + LLM for contextual Q&A)

---

📌 Notes

Make sure your .env file is added to .gitignore.

GitHub push protection will block attempts to push API keys—keep your secrets safe!

You can switch between model variants to compare their behavior and accuracy.



