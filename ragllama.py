#pip install streamlit llama-index transformers sentence-transformers faiss-cpu PyMuPDF
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import torch

# LlamaIndex & LangChain & Transformers
from llama_index.core import VectorStoreIndex, Document, Settings
# Updated import path for LangchainEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
# Updated import path for LangChainLLM
from llama_index.llms.langchain import LangChainLLM
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set Streamlit page configuration for better layout
st.set_page_config(layout="wide", page_title="PDF Q&A with Local RAG")

# --- Model Loading Section ---
# Use st.cache_resource to load models once and share them across reruns
@st.cache_resource
def load_llm_and_tokenizer():
    """Loads the Mistral-7B-Instruct LLM and its tokenizer."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", # Automatically determines where to load the model (GPU if available)
            torch_dtype=torch.float16, # Use float16 for reduced memory usage, especially on GPUs
            # You might need to adjust trust_remote_code=True for some models, but usually not needed for Mistral
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading Mistral LLM: {str(e)}")
        st.info("Please ensure you have enough memory (GPU/RAM) to load this model.")
        st.stop() # Stop the app if model loading fails

tokenizer, model = load_llm_and_tokenizer()

@st.cache_resource
def setup_llm_pipeline(tokenizer, model):
    """Sets up the HuggingFace text generation pipeline and wraps it for LlamaIndex."""
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            # Ensure the pipeline doesn't return the prompt in the output
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id, # Set end-of-sequence token
        )
        # Wrap the HuggingFace pipeline with LangChain's HuggingFacePipeline
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        # Wrap the LangChain LLM with LlamaIndex's LangChainLLM
        return LangChainLLM(llm=hf_llm) # Use the corrected LangChainLLM
    except Exception as e:
        st.error(f"❌ Error setting up LLM pipeline: {str(e)}")
        st.stop()

llm = setup_llm_pipeline(tokenizer, model)

@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace embedding model."""
    try:
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        return embed_model
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {str(e)}")
        st.stop() # Stop the app if embedding model loading fails

embed_model = load_embedding_model()

# Configure global LlamaIndex settings
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 512 # Define a chunk size for better indexing
Settings.chunk_overlap = 20 # Define overlap for better context

# --- PDF Text Extraction ---
def extract_text_from_pdf(file):
    """
    Extracts text content from an uploaded PDF file.
    Handles scanned PDFs by checking if text is found.
    """
    with st.spinner("Extracting text from PDF..."):
        try:
            pdf_stream = BytesIO(file.read())
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = " ".join(page.get_text() for page in doc if page.get_text().strip())
            if not text:
                st.warning("⚠️ No extractable text found. This PDF might contain scanned images. Text cannot be read.")
                return ""
            return text
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            return ""

# --- Index Building ---
def build_index_from_text(text):
    """
    Builds a VectorStoreIndex from extracted text.
    Uses LlamaIndex to create an in-memory index for RAG.
    """
    with st.spinner("Building index from text..."):
        try:
            documents = [Document(text=text)]
            index = VectorStoreIndex.from_documents(documents)
            return index
        except Exception as e:
            st.error(f"❌ Error building index: {str(e)}")
            return None

# --- Querying the Index ---
def query_index(index, question):
    """
    Queries the built LlamaIndex for an answer to a given question.
    """
    with st.spinner("Searching for answer..."):
        try:
            # Configure the query engine for better responses
            query_engine = index.as_query_engine(
                response_mode="tree_summarize", # Good for complex questions
                similarity_top_k=3 # Retrieve top 3 relevant chunks
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            st.error(f"❌ Error querying index: {str(e)}")
            return "An error occurred while trying to find an answer. Please try again."

# --- Streamlit Application UI ---
st.title("📄 PDF Q&A with RAG (Free & Local)")
st.markdown(
    """
    Upload a PDF document to build a local knowledge base and ask questions about its content.
    This app uses **LlamaIndex** for Retrieval-Augmented Generation (RAG)
    and **Mistral-7B-Instruct** for answering, running entirely on your machine.
    """
)
st.markdown("---")

# Initialize session state for the index
if "index" not in st.session_state:
    st.session_state.index = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Check if a new file is uploaded or if index is not built yet
    if st.session_state.index is None or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name # Store file name to detect new upload
        text = extract_text_from_pdf(uploaded_file)
        if text:
            st.session_state.index = build_index_from_text(text)
            if st.session_state.index:
                st.success("✅ PDF indexed. You can now ask questions!")
            else:
                st.error("Failed to build index. Please check the PDF content.")
    else:
        st.info(f"✅ PDF '{uploaded_file.name}' is already indexed. Ask a question!")

    # Question input and answer display
    if st.session_state.index:
        question = st.text_input("Enter your question about the PDF content:")
        if question:
            answer = query_index(st.session_state.index, question)
            st.subheader("🤖 Answer")
            st.write(answer)
    else:
        st.warning("Upload a PDF to begin asking questions.")