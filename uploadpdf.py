    
import streamlit as st
import fitz  # PyMuPDF
import cohere
from pinecone import Pinecone
import os
import time
from typing import List, Dict, Any
import uuid

@st.cache_data
def get_api_keys():
    return {
        "COHERE_API_KEY": 'swin5myqIchDwCgtXe4Kui6GtJQoDmFDxbDUZZOr',
        "PINECONE_API_KEY": 'pcsk_3dHMW6_KEyM5RhL3CARr4sY2Db6qtut9oXttAEHtweh97R8tf4zmCVRysQXTHU3BWATGS7',
        "PINECONE_ENVIRONMENT": 'test',
        "PINECONE_INDEX_NAME": 'test'
    }
# Set up page configuration
st.set_page_config(page_title="PDF Question Answering with RAG", layout="wide")

# Initialize session state variables to persist data across reruns
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # Changed to dict to store document metadata



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
def store_text_in_pinecone(chunks: List[str], metadata: Dict[str, Any], document_id: str) -> bool:
    try:
        batch_size = 100  # Process in batches to avoid timeouts
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            vectors = []
            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"{document_id}_chunk_{i+j}"
                embedding = generate_embedding(chunk)
                
                if embedding:
                    # Include both the chunk text and document metadata
                    chunk_metadata = {
                        "text": chunk,
                        "page": chunk.split("\n")[0] if chunk.startswith("Page") else "Unknown",
                        "document_title": metadata["title"],
                        "file_name": metadata["file_name"],
                        "document_id": document_id,  # Add document_id to metadata for filtering
                        "custom_name": metadata["custom_name"]  # Add custom name for display
                    }
                    vectors.append((chunk_id, embedding, chunk_metadata))
            
            if vectors:
                clients["index"].upsert(vectors=vectors)
                
        return True
    
    except Exception as e:
        st.error(f"Error storing chunks in Pinecone: {str(e)}")
        return False

# Function to delete document from Pinecone by document_id
def delete_document_from_pinecone(document_id: str) -> bool:
    try:
        # Delete all vectors with the specified document_id prefix
        clients["index"].delete(filter={"document_id": {"$eq": document_id}})
        return True
    except Exception as e:
        st.error(f"Error deleting document from Pinecone: {str(e)}")
        return False

# Function to retrieve relevant chunks from Pinecone with document filtering
def query_pinecone(question: str, top_k: int = 5, document_ids: List[str] = None) -> List[Dict[str, Any]]:
    try:
        # Generate embedding for the question
        query_vector = generate_embedding(question)
        
        if not query_vector:
            return []
        
        # Prepare filter if document_ids are provided
        filter_query = None
        if document_ids and len(document_ids) > 0:
            filter_query = {"document_id": {"$in": document_ids}}
        
        # Query Pinecone
        results = clients["index"].query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_query
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
        
        # Get source information
        sources = set([f"{chunk.metadata.get('custom_name', chunk.metadata.get('document_title', 'Unknown document'))}" 
                     for chunk in relevant_chunks])
        
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
        
        # Append source information
        answer = response.text
        sources_text = f"\n\nSources: {', '.join(sources)}"
        
        return answer + sources_text
    
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
    
    # Document management
    st.subheader("Document Management")
    if st.session_state.processed_files:
        if st.button("Clear All Documents"):
            for doc_id in st.session_state.processed_files:
                delete_document_from_pinecone(doc_id)
            st.session_state.processed_files = {}
            st.success("All documents have been removed from the index.")
    
    # List processed documents
    if st.session_state.processed_files:
        st.write("📚 Indexed Documents:")
        for doc_id, doc_info in st.session_state.processed_files.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📑 {doc_info['custom_name']}")
            with col2:
                if st.button("🗑️", key=f"delete_{doc_id}", help="Delete this document"):
                    if delete_document_from_pinecone(doc_id):
                        del st.session_state.processed_files[doc_id]
                        st.rerun()

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    
    # Document upload section
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    # Custom document name input
    custom_name = st.text_input("Document Name (for easy reference)", 
                               placeholder="Enter a name for this document")
    
    if uploaded_file is not None and custom_name:
        if st.button("Process Document"):
            # Generate a unique ID for the document
            document_id = str(uuid.uuid4())
            
            with st.spinner("Processing PDF, extracting text, and creating embeddings..."):
                pdf_data = extract_text_from_pdf(uploaded_file, chunk_size, chunk_overlap)
                
                if pdf_data:
                    # Add custom name to metadata
                    pdf_data['metadata']['custom_name'] = custom_name
                    
                    st.write(f"📊 Document: **{custom_name}**")
                    st.write(f"📄 Pages: {pdf_data['metadata']['total_pages']}")
                    st.write(f"🧩 Chunks created: {len(pdf_data['chunks'])}")
                    
                    # Store chunks in Pinecone with document_id
                    success = store_text_in_pinecone(pdf_data['chunks'], pdf_data['metadata'], document_id)
                    
                    if success:
                        # Store document info in session state
                        st.session_state.processed_files[document_id] = {
                            "file_name": uploaded_file.name,
                            "custom_name": custom_name,
                            "title": pdf_data['metadata']['title'],
                            "chunks": len(pdf_data['chunks']),
                            "pages": pdf_data['metadata']['total_pages']
                        }
                        
                        st.success(f"PDF '{custom_name}' processed and indexed successfully!")
                    else:
                        st.error("Failed to index the PDF. Please try again.")
    elif uploaded_file is not None and not custom_name:
        st.warning("Please enter a document name before processing.")

with col2:
    st.header("Ask Questions")
    
    # Document selection for query
    selected_documents = []
    if st.session_state.processed_files:
        document_options = {doc_info["custom_name"]: doc_id 
                          for doc_id, doc_info in st.session_state.processed_files.items()}
        
        # Allow selecting specific documents or all documents
        query_option = st.radio(
            "Search scope:",
            ["All Documents", "Selected Documents"],
            index=1  # Default to Selected Documents
        )
        
        if query_option == "Selected Documents":
            selected_doc_names = st.multiselect(
                "Select documents to search:",
                options=list(document_options.keys()),
                default=list(document_options.keys())[:1] if document_options else None
            )
            
            # Map selected names to document IDs
            selected_documents = [document_options[name] for name in selected_doc_names]
        
    # Question input
    question = st.text_input("Ask a question about the uploaded PDF document:")
    
    if question:
        if not st.session_state.processed_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Searching for relevant information and generating answer..."):
                # If "All Documents" is selected or no documents are selected, don't filter
                filter_doc_ids = None
                if query_option == "Selected Documents" and selected_documents:
                    filter_doc_ids = selected_documents
                
                # Retrieve relevant chunks
                results = query_pinecone(question, top_k, filter_doc_ids)
                
                if results:
                    # Generate answer
                    answer = generate_answer(question, results)
                    
                    # Display answer
                    st.markdown("### 💡 Answer:")
                    st.markdown(answer)
                    
                    # Option to show source chunks
                    with st.expander("View Source Chunks"):
                        for i, result in enumerate(results):
                            source_name = result.metadata.get("custom_name", result.metadata.get("document_title", "Unknown"))
                            st.markdown(f"**Source {i+1}** (from **{source_name}**, Relevance: {result.score:.2f})")
                            st.markdown(result.metadata["text"])
                            st.markdown("---")
                else:
                    if query_option == "Selected Documents" and selected_documents:
                        st.warning("No relevant information found in the selected documents. Try selecting different documents or rephrasing your question.")
                    else:
                        st.warning("No relevant information found. Try rephrasing your question or upload relevant documents.")