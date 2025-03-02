import streamlit as st
import fitz  # PyMuPDF
import cohere
from pinecone import Pinecone
import os
import time
import uuid
import base64
from typing import List, Dict, Any
from io import BytesIO

# Set up page configuration
st.set_page_config(page_title="PDF Question Answering with RAG", layout="wide")

# Initialize session state variables to persist data across reruns
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # Store document metadata
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None
if "pdf_display_page" not in st.session_state:
    st.session_state.pdf_display_page = 0
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "highlighted_text" not in st.session_state:
    st.session_state.highlighted_text = []  # Store text to highlight
if "last_query" not in st.session_state:
    st.session_state.last_query = None

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

# Function to extract text from PDF with chunking and store page text mapping
def extract_text_from_pdf(pdf_file, chunk_size=1000, chunk_overlap=200):
    try:
        # Save the PDF bytes for display
        pdf_bytes = pdf_file.getvalue()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", "Untitled"),
            "author": doc.metadata.get("author", "Unknown"),
            "file_name": pdf_file.name,
            "total_pages": len(doc)
        }
        
        # Extract full text and store page content separately
        full_text = ""
        page_texts = {}  # Store text for each page
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            page_header = f"Page {page_num+1}: "
            full_text += page_header + text + "\n\n"
            
            # Store raw page text without header for highlighting
            page_texts[page_num] = text
        
        # Create chunks with overlap
        chunks = []
        current_page = 0
        page_starts = [0]  # Track starting indices of pages
        
        # Build page start positions
        for page_num in range(len(doc)):
            page_starts.append(len(full_text))
            full_text += f"Page {page_num+1}: " + doc[page_num].get_text("text") + "\n\n"
        
        # Create chunks with page metadata
        words = full_text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            
            # Find which page this chunk starts in
            chunk_start = len(" ".join(words[:i]))
            for page_num in range(len(page_starts)-1):
                if page_starts[page_num] <= chunk_start < page_starts[page_num+1]:
                    current_page = page_num
                    break
            
            chunks.append({
                "text": chunk_text,
                "start_page": current_page
            })
            
        return {
            "text": full_text,
            "chunks": chunks,  # Now chunks have text and start_page
            "metadata": metadata,
            "pdf_bytes": pdf_bytes,
            "page_texts": page_texts
        }
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to display PDF with highlighted text
def display_pdf(pdf_bytes, page=0, highlighted_text=None):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # Ensure page is within bounds
        page = int(max(0, min(page, total_pages - 1)))
        
        # Get the current page
        pdf_page = doc[page]
        
        # Add highlights if specified for this page
        if highlighted_text:
            for text in highlighted_text:
                text_instances = pdf_page.search_for(text)
                for inst in text_instances:
                    highlight = pdf_page.add_highlight_annot(inst)
                    highlight.update()
        
        # Render page to an image (PNG)
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
        img_bytes = pix.tobytes("png")
        
        # Display the image
        st.image(img_bytes, caption=f"Page {page + 1} of {total_pages}", use_column_width=True)
        
        # Return total pages for navigation
        return total_pages
    
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        return 0

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
def store_text_in_pinecone(chunks: List[Dict], metadata: Dict[str, Any], document_id: str, page_texts: Dict[int, str]) -> bool:
    try:
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            vectors = []
            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"{document_id}_chunk_{i+j}"
                embedding = generate_embedding(chunk["text"])
                
                if embedding:
                    chunk_metadata = {
                        "text": chunk["text"],
                        "document_title": metadata["title"],
                        "file_name": metadata["file_name"],
                        "document_id": document_id,
                        "custom_name": metadata["custom_name"],
                        "start_page": chunk["start_page"]  # Use tracked page number
                    }
                    vectors.append((chunk_id, embedding, chunk_metadata))
            
            if vectors:
                clients["index"].upsert(vectors=vectors)
                
        metadata["page_texts"] = page_texts
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

# Extract key sentences from chunk text
def extract_key_sentences(text, max_sentences=3):
    # Simple sentence extraction based on periods
    sentences = []
    for part in text.split("\n"):
        for sentence in part.split(". "):
            clean_sentence = sentence.strip()
            if clean_sentence and len(clean_sentence) > 20:  # Avoid very short fragments
                sentences.append(clean_sentence)
    
    # Return up to max_sentences
    return sentences[:max_sentences]

# Function to generate answer with Cohere
def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    try:
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk.metadata["text"] for chunk in relevant_chunks])
        
        # Get source information
        sources = set([f"{chunk.metadata.get('custom_name', chunk.metadata.get('document_title', 'Unknown document'))}" 
                     for chunk in relevant_chunks])
        
        # Extract page numbers for navigation and text for highlighting
        page_info = {}
        highlights = {}
        
        highlights = {}
        page_info = {}
        
        for chunk in relevant_chunks:
            doc_id = chunk.metadata.get("document_id")
            page_num = chunk.metadata.get("start_page", 0)
            
            # Store highlight information
            if doc_id not in highlights:
                highlights[doc_id] = {}
            if page_num not in highlights[doc_id]:
                highlights[doc_id][page_num] = []
            
            # Extract key sentences from chunk text
            key_sentences = extract_key_sentences(chunk.metadata["text"])
            highlights[doc_id][page_num].extend(key_sentences)
            
            # Store page info for navigation
            if doc_id not in page_info:
                page_info[doc_id] = {
                    "name": chunk.metadata.get("custom_name"),
                    "pages": set()
                }
            page_info[doc_id]["pages"].add(page_num)
        
        # Convert page sets to sorted lists
        for doc_id in page_info:
            page_info[doc_id]["pages"] = sorted(page_info[doc_id]["pages"])
        
        # Construct prompt for Cohere
        prompt = f"""You are a helpful assistant answering questions based on the provided document.
        
Document Context:
{context}

Question: {question}

Please provide a comprehensive answer based ONLY on the information in the document. If the answer cannot be found in the document, say "I don't have enough information to answer this question based on the provided document." Don't mention the specific page numbers in your answer.
"""
        
        # Generate response using Cohere's Generate endpoint
        response = clients["cohere"].chat(
            message=prompt,
            model="command-r-plus",
            temperature=0.2,
            max_tokens=800
        )
        
        answer = response.text
        
        # Add sources with page references
        sources_text = f"\n\nSources: {', '.join(sources)}"
        
        # If we're looking at a specific document and found page references
        if len(page_info) == 1 and st.session_state.current_pdf is not None:
            doc_id, info = list(page_info.items())[0]
            if doc_id == st.session_state.current_pdf:
                page_refs = ", ".join([f"page {p+1}" for p in sorted(info["pages"])])
                sources_text += f" ({page_refs})"
        
        return answer + sources_text, page_info, highlights
    
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating your answer.", {}, {}

# Sidebar for document management
with st.sidebar:
    st.title("📚 Document Manager")
    
    # Upload section
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    custom_name = st.text_input("Document Name", placeholder="Enter a name for this document")
    
    if uploaded_file is not None and custom_name:
        if st.button("Process Document"):
            # Generate a unique ID for the document
            document_id = str(uuid.uuid4())
            
            with st.spinner("Processing PDF..."):
                pdf_data = extract_text_from_pdf(uploaded_file)
                
                if pdf_data:
                    # Add custom name to metadata
                    pdf_data['metadata']['custom_name'] = custom_name
                    
                    # Store chunks in Pinecone with document_id
                    success = store_text_in_pinecone(
                        pdf_data['chunks'], 
                        pdf_data['metadata'], 
                        document_id,
                        pdf_data['page_texts']
                    )
                    
                    if success:
                        # Store document info in session state
                        st.session_state.processed_files[document_id] = {
                            "file_name": uploaded_file.name,
                            "custom_name": custom_name,
                            "title": pdf_data['metadata']['title'],
                            "chunks": len(pdf_data['chunks']),
                            "pages": pdf_data['metadata']['total_pages'],
                            "pdf_bytes": pdf_data['pdf_bytes'],
                            "page_texts": pdf_data['page_texts']  # Store page texts
                        }
                        
                        # Set as current PDF
                        st.session_state.current_pdf = document_id
                        st.session_state.current_pdf_name = custom_name
                        st.session_state.pdf_bytes = pdf_data['pdf_bytes']
                        st.session_state.pdf_display_page = 0
                        st.session_state.highlighted_text = []  # Clear any highlights
                        
                        st.success(f"PDF '{custom_name}' processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to index the PDF. Please try again.")
    elif uploaded_file is not None and not custom_name:
        st.warning("Please enter a document name before processing.")
    
    # Document management
    st.header("Your Documents")
    
    if st.session_state.processed_files:
        for doc_id, doc_info in st.session_state.processed_files.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                doc_name = doc_info["custom_name"]
                if st.button(f"📄 {doc_name}", key=f"open_{doc_id}"):
                    st.session_state.current_pdf = doc_id
                    st.session_state.current_pdf_name = doc_name
                    st.session_state.pdf_bytes = doc_info["pdf_bytes"]
                    st.session_state.pdf_display_page = 0
                    st.session_state.highlighted_text = []  # Clear any highlights
                    st.rerun()
            
            with col2:
                # View/active indicator
                if st.session_state.current_pdf == doc_id:
                    st.markdown("📌 Active")
            
            with col3:
                # Delete button
                if st.button("🗑️", key=f"delete_{doc_id}"):
                    if delete_document_from_pinecone(doc_id):
                        # If deleting the current PDF, clear it
                        if st.session_state.current_pdf == doc_id:
                            st.session_state.current_pdf = None
                            st.session_state.current_pdf_name = None
                            st.session_state.pdf_bytes = None
                            st.session_state.highlighted_text = []
                        
                        del st.session_state.processed_files[doc_id]
                        st.rerun()
    else:
        st.info("No documents uploaded yet. Upload a PDF to get started.")
    
    # Document clearing
    if st.session_state.processed_files and st.button("Clear All Documents"):
        for doc_id in list(st.session_state.processed_files.keys()):
            delete_document_from_pinecone(doc_id)
        
        st.session_state.processed_files = {}
        st.session_state.current_pdf = None
        st.session_state.current_pdf_name = None
        st.session_state.pdf_bytes = None
        st.session_state.highlighted_text = []
        st.rerun()
    
    # Settings section
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size (words)", 300, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap (words)", 50, 500, 200)
    top_k = st.slider("Results to Retrieve", 3, 10, 5)
    highlight_enabled = st.checkbox("Enable Text Highlighting", value=True)

# Main content area with two columns
col1, col2 = st.columns([1, 1])

# Left column: PDF Viewer
with col1:
    st.title("📑 PDF Viewer")
    
    if st.session_state.current_pdf and st.session_state.pdf_bytes:
        # Display document name
        st.header(st.session_state.current_pdf_name)
        
        # Get current page highlights
        current_highlights = []
        if highlight_enabled and st.session_state.current_pdf in st.session_state.highlighted_text:
            page_highlights = st.session_state.highlighted_text[st.session_state.current_pdf]
            if st.session_state.pdf_display_page in page_highlights:
                current_highlights = page_highlights[st.session_state.pdf_display_page]
        
        # PDF display with highlights
        total_pages = display_pdf(
            st.session_state.pdf_bytes, 
            st.session_state.pdf_display_page,
            current_highlights if highlight_enabled else None
        )
        
        # Page navigation
        if total_pages > 1:
            col_prev, col_page, col_next = st.columns([1, 3, 1])
            
            with col_prev:
                if st.button("◀️ Previous") and st.session_state.pdf_display_page > 0:
                    st.session_state.pdf_display_page -= 1
                    st.rerun()
            
            with col_page:
                page_num = st.select_slider(
                    "Page Navigation",
                    options=list(range(1, total_pages + 1)),
                    value=st.session_state.pdf_display_page + 1
                )
                if page_num - 1 != st.session_state.pdf_display_page:
                    st.session_state.pdf_display_page = page_num - 1
                    st.rerun()
            
            with col_next:
                if st.button("Next ▶️") and st.session_state.pdf_display_page < total_pages - 1:
                    st.session_state.pdf_display_page += 1
                    st.rerun()
    else:
        st.info("Select or upload a document to view it here.")

# Right column: Q&A Interface
with col2:
    st.title("❓ Ask Questions")
    
    # Document selection for query
    if st.session_state.processed_files:
        st.subheader("Search Documents")
        
        # Options for search scope
        query_option = st.radio(
            "Search scope:",
            ["Current Document", "Selected Documents", "All Documents"],
            index=0 if st.session_state.current_pdf else 1
        )
        
        # Initialize selected documents list
        selected_documents = []
        
        if query_option == "Current Document":
            if st.session_state.current_pdf:
                selected_documents = [st.session_state.current_pdf]
                st.info(f"Searching in: {st.session_state.current_pdf_name}")
            else:
                st.warning("No document is currently open. Please select a document first.")
                query_option = "Selected Documents"  # Fall back to selected documents
        
        if query_option == "Selected Documents":
            # Create a mapping of document names to IDs
            document_options = {
                doc_info["custom_name"]: doc_id 
                for doc_id, doc_info in st.session_state.processed_files.items()
            }
            
            # Select documents to search
            selected_doc_names = st.multiselect(
                "Select documents to search:",
                options=list(document_options.keys()),
                default=[st.session_state.current_pdf_name] if st.session_state.current_pdf_name in document_options.values() else None
            )
            
            # Map selected names to document IDs
            selected_documents = [document_options[name] for name in selected_doc_names]
        
        # Question input
        question = st.text_input("Ask a question about the document(s):")
        
        if question:
            with st.spinner("Generating answer..."):
                # If "All Documents" is selected, don't filter by document ID
                filter_doc_ids = None
                if query_option != "All Documents":
                    filter_doc_ids = selected_documents
                
                if not filter_doc_ids and query_option != "All Documents":
                    st.warning("Please select at least one document to search.")
                else:
                    # Store the current query
                    st.session_state.last_query = question
                    
                    # Retrieve relevant chunks
                    results = query_pinecone(question, top_k, filter_doc_ids)
                    
                    if results:
                        # Generate answer
                        answer, page_info, highlights = generate_answer(question, results)
                        
                        # Store highlights for display
                        if highlight_enabled:
                            st.session_state.highlighted_text = highlights
                            
                            # Automatically show the first highlighted page if we're in current document mode
                            if len(page_info) == 1:
                                doc_id = list(page_info.keys())[0]
                                if doc_id == st.session_state.current_pdf:
                                    first_page = list(page_info[doc_id]["pages"])[0]
                                    st.session_state.pdf_display_page = first_page
                
                # If answer references different document, switch to it
                            elif len(page_info) > 0:
                                doc_id = list(page_info.keys())[0]
                                if doc_id in st.session_state.processed_files:
                                    st.session_state.current_pdf = doc_id
                                    st.session_state.current_pdf_name = page_info[doc_id]["name"]
                                    st.session_state.pdf_bytes = st.session_state.processed_files[doc_id]["pdf_bytes"]
                                    st.session_state.pdf_display_page = list(page_info[doc_id]["pages"])[0]
                        
                        # Display answer
                        st.markdown("### 💡 Answer:")
                        st.markdown(answer)
                        
                        # Page navigation suggestions if we have page info for the current document
                        if st.session_state.current_pdf in page_info and query_option == "Current Document":
                            st.subheader("Jump to Referenced Pages")
                            page_buttons = st.columns(min(5, len(page_info[st.session_state.current_pdf]["pages"])))
                            
                            for i, page_num in enumerate(sorted(page_info[st.session_state.current_pdf]["pages"])):
                                with page_buttons[i % 5]:
                                    if st.button(f"Page {page_num + 1}", key=f"jump_to_{page_num}"):
                                        st.session_state.pdf_display_page = page_num
                                        st.rerun()
                        
                        # Option to show source chunks
                        with st.expander("View Source Chunks"):
                            for i, result in enumerate(results):
                                source_name = result.metadata.get("custom_name", result.metadata.get("document_title", "Unknown"))
                                page_text = result.metadata.get("page", "Unknown page")
                                st.markdown(f"**Source {i+1}** from **{source_name}** ({page_text})")
                                st.markdown(result.metadata["text"])
                                st.markdown("---")
                    else:
                        if query_option != "All Documents":
                            st.warning("No relevant information found in the selected document(s). Try expanding your search or rephrasing your question.")
                        else:
                            st.warning("No relevant information found. Try rephrasing your question.")
    else:
        st.info("Please upload a document first to start asking questions.")