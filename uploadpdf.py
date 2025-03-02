import streamlit as st
import fitz  # PyMuPDF
import cohere
from pinecone import Pinecone
import os
import time
from typing import List, Dict, Any

# Set up page configuration
st.set_page_config(page_title="PDF Question Answering with RAG", layout="wide")

# Initialize session state variables to persist data across reruns
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# API keys - in production, these should be stored in environment variables
@st.cache_data
def get_api_keys():
    return {
        "COHERE_API_KEY": 'swin5myqIchDwCgtXe4Kui6GtJQoDmFDxbDUZZOr',
        "PINECONE_API_KEY": 'pcsk_3dHMW6_KEyM5RhL3CARr4sY2Db6qtut9oXttAEHtweh97R8tf4zmCVRysQXTHU3BWATGS7',
        "PINECONE_ENVIRONMENT": 'test',
        "PINECONE_INDEX_NAME": 'test'
    }

API_KEYS = get_api_keys()

# Initialize clients with error handling
@st.cache_resource
def initialize_clients():
    try:
        # Initialize Cohere client
        co = cohere.Client(api_key=API_KEYS["COHERE_API_KEY"])
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=API_KEYS["PINECONE_API_KEY"])
        
        # Check if index exists
        index_list = [idx["name"] for idx in pc.list_indexes()]
        
        # Create index if it doesn't exist
        if API_KEYS["PINECONE_INDEX_NAME"] not in index_list:
            pc.create_index(
                name=API_KEYS["PINECONE_INDEX_NAME"],
                dimension=1024,  # Dimension for Cohere embed-english-v3.0
                metric="cosine",
                spec="serverless"  # Use serverless for easy startup
            )
            # Wait for index to be ready
            time.sleep(10)
        
        # Connect to the index
        index = pc.Index(API_KEYS["PINECONE_INDEX_NAME"])
        
        return {"cohere": co, "pinecone": pc, "index": index}
    
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None

# Check if API keys are provided
if not all(API_KEYS.values()):
    st.error("Please provide all required API keys.")
else:
    clients = initialize_clients()
    if not clients:
        st.stop()

# Function to extract text from PDF with chunking
def extract_text_from_pdf(pdf_file, chunk_size=1000, chunk_overlap=200):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", "Untitled"),
            "author": doc.metadata.get("author", "Unknown"),
            "file_name": pdf_file.name,
            "total_pages": len(doc)
        }
        
        # Extract full text
        full_text = ""
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            full_text += f"Page {page_num+1}: {text}\n\n"
        
        # Create chunks with overlap
        chunks = []
        words = full_text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:  # Ensure we don't add empty chunks
                chunks.append(chunk)
        
        return {"text": full_text, "chunks": chunks, "metadata": metadata}
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to generate embeddings using Cohere
def generate_embedding(text: str) -> List[float]:
    try:
        response = clients["cohere"].embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings[0]
    except Exception as e:
        st.error(f"Failed to generate embedding: {str(e)}")
        return None

# Function to store document chunks in Pinecone
def store_text_in_pinecone(chunks: List[str], metadata: Dict[str, Any]) -> bool:
    try:
        batch_size = 100  # Process in batches to avoid timeouts
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            vectors = []
            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"{metadata['file_name']}_chunk_{i+j}"
                embedding = generate_embedding(chunk)
                
                if embedding:
                    # Include both the chunk text and document metadata
                    chunk_metadata = {
                        "text": chunk,
                        "page": chunk.split("\n")[0] if chunk.startswith("Page") else "Unknown",
                        "document_title": metadata["title"],
                        "file_name": metadata["file_name"]
                    }
                    vectors.append((chunk_id, embedding, chunk_metadata))
            
            if vectors:
                clients["index"].upsert(vectors=vectors)
                
        return True
    
    except Exception as e:
        st.error(f"Error storing chunks in Pinecone: {str(e)}")
        return False

# Function to retrieve relevant chunks from Pinecone
def query_pinecone(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        # Generate embedding for the question
        query_vector = generate_embedding(question)
        
        if not query_vector:
            return []
        
        # Query Pinecone
        results = clients["index"].query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return []

# Function to generate answer with Cohere
def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    try:
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk.metadata["text"] for chunk in relevant_chunks])
        
        # Construct prompt for Cohere
        prompt = f"""You are a helpful assistant answering questions based on the provided document.
        
Document Context:
{context}

Question: {question}

Please provide a comprehensive answer based ONLY on the information in the document. If the answer cannot be found in the document, say "I don't have enough information to answer this question based on the provided document."
"""
        
        # Generate response using Cohere's Generate endpoint
        response = clients["cohere"].chat(
            message=prompt,
            model="command-r-plus",
            temperature=0.2,
            max_tokens=800
        )
        
        return response.text
    
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating your answer."

# Streamlit UI
st.title("📄 PDF Question Answering with RAG")
st.subheader("Upload a PDF, ask questions, and get AI-powered answers")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API key input fields (hidden in production)
    if st.checkbox("Show API Settings", False):
        st.warning("In production, use environment variables instead of entering keys here.")
        API_KEYS["COHERE_API_KEY"] = st.text_input("Cohere API Key", value=API_KEYS["COHERE_API_KEY"], type="password")
        API_KEYS["PINECONE_API_KEY"] = st.text_input("Pinecone API Key", value=API_KEYS["PINECONE_API_KEY"], type="password")
        API_KEYS["PINECONE_INDEX_NAME"] = st.text_input("Pinecone Index Name", value=API_KEYS["PINECONE_INDEX_NAME"])
        
        if st.button("Re-initialize Clients"):
            st.cache_resource.clear()
            clients = initialize_clients()
    
    # RAG settings
    st.subheader("RAG Settings")
    chunk_size = st.slider("Chunk Size (words)", 300, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap (words)", 50, 500, 200)
    top_k = st.slider("Number of Retrieved Chunks", 3, 10, 5)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        file_hash = hash(uploaded_file.name + str(uploaded_file.size))
        
        # Only process the file if it hasn't been processed before
        if file_hash not in st.session_state.processed_files:
            with st.spinner("Processing PDF, extracting text, and creating embeddings..."):
                pdf_data = extract_text_from_pdf(uploaded_file, chunk_size, chunk_overlap)
                
                if pdf_data:
                    st.write(f"📊 Document: **{pdf_data['metadata']['title']}**")
                    st.write(f"📄 Pages: {pdf_data['metadata']['total_pages']}")
                    st.write(f"🧩 Chunks created: {len(pdf_data['chunks'])}")
                    
                    # Store chunks in Pinecone
                    success = store_text_in_pinecone(pdf_data['chunks'], pdf_data['metadata'])
                    
                    if success:
                        st.success("PDF processed and indexed successfully!")
                        st.session_state.processed_files.add(file_hash)
                    else:
                        st.error("Failed to index the PDF. Please try again.")
        else:
            st.success(f"'{uploaded_file.name}' is already processed and indexed!")

with col2:
    st.header("Ask Questions")
    question = st.text_input("Ask a question about the uploaded PDF document:")
    
    if question:
        with st.spinner("Searching for relevant information and generating answer..."):
            # Retrieve relevant chunks
            results = query_pinecone(question, top_k)
            
            if results:
                # Generate answer
                answer = generate_answer(question, results)
                
                # Display answer
                st.markdown("### 💡 Answer:")
                st.markdown(answer)
                
                # Option to show source chunks
                with st.expander("View Source Chunks"):
                    for i, result in enumerate(results):
                        st.markdown(f"**Source {i+1}** (Relevance: {result.score:.2f})")
                        st.markdown(result.metadata["text"])
                        st.markdown("---")
            else:
                st.warning("No relevant information found in the document. Try rephrasing your question or upload a relevant document.")