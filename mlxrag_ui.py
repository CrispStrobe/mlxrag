import streamlit as st
import threading

# Global lock for MPS-critical sections
mps_lock = threading.Lock()

# Set page configuration
st.set_page_config(
    page_title="mlxrag v.0.1",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import tempfile
import shutil
import time
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add the current directory to the path so we can import the module
sys.path.append('.')

# Try to import from the main script
try:
    with mps_lock:
        from mlxrag import (
            DocumentProcessor, QdrantManager, TextExtractor, 
            check_qdrant_available, get_files_to_process, download_model_files,
            DEFAULT_COLLECTION, DEFAULT_MODEL, DEFAULT_WEIGHTS_PATH,
            DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL,
            MLX_EMBEDDING_REGISTRY, HAS_MLX_EMBEDDING_MODELS, HAS_MLX
        )
    MODULE_LOADED = True
except Exception as e:
    st.error(f"Error importing module: {str(e)}")
    MODULE_LOADED = False

# Define CSS for styling with improved colors for visibility
st.markdown("""
<style>
    .result-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 2px;
        border-radius: 3px;
        font-weight: bold;
        color: #000;
    }
    .info-box {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        border: 1px solid #bbdefb;
    }
    .filename {
        font-weight: bold;
        font-size: 16px;
        color: #2196f3;
    }
    .meta-info {
        color: #666;
        font-size: 14px;
        margin-top: 5px;
    }
    .preview-text {
        background-color: #fff;
        color: #000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #eee;
        margin-top: 10px;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    .stButton button {
        width: 100%;
    }
    .search-header {
        margin-bottom: 20px;
    }
    .pagination-btn {
        margin: 0 5px;
    }
    .page-info {
        text-align: center;
        margin: 10px 0;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

if not MODULE_LOADED:
    st.error("Failed to load the required modules. Please make sure the their python file is in the same directory.")
    st.stop()

# Create a wrapper for critical GPU-related functions

def mps_safe_call(fn, *args, **kwargs):
    with mps_lock:
        return fn(*args, **kwargs)

# We use `mps_safe_call` around anything using MLX or PyTorch models
# For example: mps_safe_call(qdrant.insert_embeddings, results)
# Or wrap: results = mps_safe_call(processor.process_file, file_path)

# Initialize session state variables
if 'qdrant_manager' not in st.session_state:
    st.session_state.qdrant_manager = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_initialized' not in st.session_state:
    st.session_state.search_initialized = False
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'search_limit' not in st.session_state:
    st.session_state.search_limit = 10
if 'context_size' not in st.session_state:
    st.session_state.context_size = 500
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'total_results' not in st.session_state:
    st.session_state.total_results = 0

# Main app with tabs
tab1, tab2 = st.tabs(["Index Documents", "Search Documents"])

# Tab 1: Index Documents
with tab1:
    st.title("Document Indexer")
    
    # Connection settings (sidebar)
    with st.sidebar:
        st.title("Settings")
        
        # Connection type
        connection_type = st.radio("Connection Type", ["Local Storage", "Remote Server"])
        
        if connection_type == "Local Storage":
            host = "localhost"
            port = 6333
            # Use default local storage path (will be set by Qdrant manager)
            storage_path = os.path.join(os.getcwd(), "qdrant_storage")
            st.info(f"Using local storage at: {storage_path}")
            
            # Option to clear storage
            if st.button("Clear Storage"):
                try:
                    # Create directory path
                    collections_dir = os.path.join(storage_path, "collections")
                    if os.path.exists(collections_dir):
                        shutil.rmtree(collections_dir)
                        os.makedirs(collections_dir, exist_ok=True)
                        st.success("Storage cleared successfully!")
                    else:
                        os.makedirs(collections_dir, exist_ok=True)
                        st.info("Created storage directory.")
                except Exception as e:
                    st.error(f"Error clearing storage: {e}")
        else:
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=6333, min_value=1, max_value=65535)
            storage_path = None
        
        # Collection settings
        collection_name = st.text_input("Collection Name", DEFAULT_COLLECTION)
        recreate_collection = st.checkbox("Recreate Collection if Exists")
        
        # Model settings
        st.subheader("Model Settings")
        use_mlx_models = st.checkbox("Use MLX Embedding Models", value=HAS_MLX_EMBEDDING_MODELS)
        
        if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
            dense_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if not v.get("lm_head")]
            sparse_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if v.get("lm_head")]
            
            dense_model = st.selectbox("Dense Model", dense_models, 
                                     index=dense_models.index(DEFAULT_DENSE_MODEL) if DEFAULT_DENSE_MODEL in dense_models else 0)
            sparse_model = st.selectbox("Sparse Model", sparse_models,
                                       index=sparse_models.index(DEFAULT_SPARSE_MODEL) if DEFAULT_SPARSE_MODEL in sparse_models else 0)
        else:
            dense_model = DEFAULT_DENSE_MODEL
            sparse_model = DEFAULT_SPARSE_MODEL
        
        # Advanced settings
        st.session_state.show_debug = st.checkbox("Show Debug Info", False)
            
    # Check connection button
    if st.button("Check Connection"):
        try:
            qdrant_available = mps_safe_call(check_qdrant_available, host, port, verbose=True)
            if qdrant_available:
                st.success("✅ Qdrant is available!")
                
                # Test connection with Qdrant Manager directly
                qdrant = mps_safe_call(QdrantManager,
                    host=host,
                    port=port,
                    collection_name=collection_name,
                    storage_path=storage_path if connection_type == "Local Storage" else None,
                    verbose=st.session_state.show_debug,
                    dense_model_id=dense_model,
                    sparse_model_id=sparse_model
                )
                
                # List collections
                collections = mps_safe_call(lambda: qdrant.client.get_collections().collections)

                collection_names = [c.name for c in collections]
                
                if collection_name in collection_names:
                    st.success(f"Collection '{collection_name}' exists!")
                    
                    # Get collection info
                    collection_info = mps_safe_call(qdrant.get_collection_info)
                    if "points_count" in collection_info:
                        st.info(f"Collection has {collection_info['points_count']} points")
                else:
                    st.info(f"Collection '{collection_name}' does not exist yet. It will be created during indexing.")
            else:
                st.error("❌ Qdrant is not available.")
        except Exception as e:
            st.error(f"Error checking connection: {str(e)}")
    
    # File upload option
    upload_option = st.radio("Upload Method", ["Upload Files", "Specify Directory"])
    
    if upload_option == "Upload Files":
        uploaded_files = st.file_uploader("Upload documents", 
                                          accept_multiple_files=True, 
                                          type=["txt", "md", "pdf", "json", "csv", "html"])
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} files.")
            
            # Create a temp directory for uploaded files if needed
            if 'upload_dir' not in st.session_state:
                st.session_state.upload_dir = os.path.join("qdrant_uploads")
                os.makedirs(st.session_state.upload_dir, exist_ok=True)
            
            # Button to index uploaded files
            if st.button("Index Uploaded Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded files to temp directory
                saved_files = []
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / (len(uploaded_files) * 2)  # First half for saving
                    progress_bar.progress(progress)
                    status_text.text(f"Saving file {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Save the file
                    file_path = os.path.join(st.session_state.upload_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_files.append(file_path)
                
                # Initialize Qdrant and processor
                try:
                    # Initialize Qdrant manager
                    qdrant = mps_safe_call(QdrantManager,
                        host=host,
                        port=port,
                        collection_name=collection_name,
                        storage_path=storage_path if connection_type == "Local Storage" else None,
                        verbose=st.session_state.show_debug,
                        dense_model_id=dense_model,
                        sparse_model_id=sparse_model
                    )
                    
                    # Create collection
                    mps_safe_call(qdrant.create_collection, recreate=recreate_collection)
                    
                    # Initialize document processor
                    processor_args = {
                        "model_name": DEFAULT_MODEL,
                        "weights_path": DEFAULT_WEIGHTS_PATH,
                        "verbose": st.session_state.show_debug
                    }
                    
                    if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
                        processor_args.update({
                            "dense_model": dense_model,
                            "sparse_model": sparse_model,
                            "use_mlx_embedding": True
                        })
                    
                    processor = mps_safe_call(DocumentProcessor, **processor_args)
                    
                    # Process and index each file
                    total_chunks = 0
                    successful_files = 0
                    
                    for i, file_path in enumerate(saved_files):
                        progress = 0.5 + (i + 1) / (len(saved_files) * 2)  # Second half for indexing
                        progress_bar.progress(progress)
                        status_text.text(f"Indexing file {i+1}/{len(saved_files)}: {os.path.basename(file_path)}")
                        
                        try:
                            # Choose processing method based on model type
                            if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
                                results = mps_safe_call(processor.process_file_with_sparse, file_path)
                                if results:
                                    mps_safe_call(qdrant.insert_embeddings_with_sparse, results)
                                    total_chunks += len(results)
                                    successful_files += 1
                            else:
                                results = mps_safe_call(processor.process_file, file_path)
                                if results:
                                    mps_safe_call(qdrant.insert_embeddings, results)
                                    total_chunks += len(results)
                                    successful_files += 1
                                    
                        except Exception as e:
                            st.error(f"Error processing {file_path}: {str(e)}")
                            continue
                    
                    # Update session state
                    st.session_state.qdrant_manager = qdrant
                    st.session_state.processor = processor
                    st.session_state.search_initialized = True
                    
                    # Show success message
                    progress_bar.progress(1.0)
                    status_text.text("Indexing complete!")
                    st.success(f"Successfully indexed {successful_files}/{len(saved_files)} files with {total_chunks} chunks.")
                    
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
                    import traceback
                    if st.session_state.show_debug:
                        st.error(traceback.format_exc())
    
    else:  # Directory indexing
        directory_path = st.text_input("Directory Path", "")
        include_patterns = st.text_input("Include Patterns (space separated)", "*.txt *.pdf *.md *.csv *.json")
        
        if directory_path and os.path.isdir(directory_path):
            if st.button("Index Directory"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Get files to process
                    include_patterns_list = include_patterns.split() if include_patterns else None
                    files = get_files_to_process(
                        directory_path,
                        include_patterns=include_patterns_list,
                        verbose=st.session_state.show_debug
                    )
                    
                    if not files:
                        st.warning("No matching files found in directory.")
                    else:
                        status_text.text(f"Found {len(files)} files to process.")
                        
                        # Initialize Qdrant manager
                        qdrant = mps_safe_call(QdrantManager,
                            host=host,
                            port=port,
                            collection_name=collection_name,
                            storage_path=storage_path if connection_type == "Local Storage" else None,
                            verbose=st.session_state.show_debug,
                            dense_model_id=dense_model,
                            sparse_model_id=sparse_model
                        )
                        
                        # Create collection
                        mps_safe_call(qdrant.create_collection, recreate=recreate_collection)
                        
                        # Initialize document processor
                        processor_args = {
                            "model_name": DEFAULT_MODEL,
                            "weights_path": DEFAULT_WEIGHTS_PATH,
                            "verbose": st.session_state.show_debug
                        }
                        
                        if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
                            processor_args.update({
                                "dense_model": dense_model,
                                "sparse_model": sparse_model,
                                "use_mlx_embedding": True
                            })
                        
                        processor = mps_safe_call(DocumentProcessor, **processor_args)
                        
                        # Process and index each file
                        total_chunks = 0
                        successful_files = 0
                        
                        for i, file_path in enumerate(files):
                            progress = (i + 1) / len(files)
                            progress_bar.progress(progress)
                            status_text.text(f"Indexing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                            
                            try:
                                # Choose processing method based on model type
                                if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
                                    results = mps_safe_call(processor.process_file_with_sparse, file_path)
                                    if results:
                                        mps_safe_call(qdrant.insert_embeddings_with_sparse, results)
                                        total_chunks += len(results)
                                        successful_files += 1
                                else:
                                    results = mps_safe_call(processor.process_file, file_path)
                                    if results:
                                        mps_safe_call(qdrant.insert_embeddings, results)
                                        total_chunks += len(results)
                                        successful_files += 1
                                        
                            except Exception as e:
                                st.error(f"Error processing {file_path}: {str(e)}")
                                continue
                        
                        # Update session state
                        st.session_state.qdrant_manager = qdrant
                        st.session_state.processor = processor
                        st.session_state.search_initialized = True
                        
                        # Show success message
                        progress_bar.progress(1.0)
                        status_text.text("Indexing complete!")
                        st.success(f"Successfully indexed {successful_files}/{len(files)} files with {total_chunks} chunks.")
                
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
                    import traceback
                    if st.session_state.show_debug:
                        st.error(traceback.format_exc())
        
        elif directory_path:
            st.error(f"Directory '{directory_path}' not found.")

# Tab 2: Search
with tab2:
    st.title("Document Search")
    
    # Connection settings in sidebar
    with st.sidebar:
        st.title("Search Settings")
        
        st.session_state.search_limit = st.number_input("Results Per Page", 5, 50, 10)
        max_results = st.number_input("Max Total Results", 10, 1000, 100)

        st.session_state.context_size = st.number_input("Context Size", 200, 2000, 500)
        prefetch_limit = st.number_input("Prefetch Limit for Hybrid Search", 10, 200, 50)
        fusion_type = st.selectbox("Fusion Strategy for Hybrid Search", ["rrf", "dbsf"])
        rerank = st.checkbox("Apply Reranking", True)
        sort_by_score = st.checkbox("Sort Results by Score", value=True)
    
    # Initialize search if needed
    if not st.session_state.search_initialized:
        st.info("No active connection. Please initialize search to continue.")
        
        if st.button("Initialize Search"):
            try:
                # Initialize Qdrant manager
                qdrant = mps_safe_call(QdrantManager,
                    host=host,
                    port=port,
                    collection_name=collection_name,
                    storage_path=storage_path if connection_type == "Local Storage" else None,
                    verbose=st.session_state.show_debug,
                    dense_model_id=dense_model,
                    sparse_model_id=sparse_model
                )
                
                # Check if collection exists
                collections = mps_safe_call(lambda: qdrant.client.get_collections().collections)

                collection_exists = any(c.name == collection_name for c in collections)
                
                if not collection_exists:
                    st.warning(f"Collection '{collection_name}' doesn't exist. Creating it now...")
                    mps_safe_call(qdrant.create_collection, recreate=False)
                
                # Initialize document processor
                processor_args = {
                    "model_name": DEFAULT_MODEL,
                    "weights_path": DEFAULT_WEIGHTS_PATH,
                    "verbose": st.session_state.show_debug
                }
                
                if use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
                    processor_args.update({
                        "dense_model": dense_model,
                        "sparse_model": sparse_model,
                        "use_mlx_embedding": True
                    })
                
                processor = mps_safe_call(DocumentProcessor, **processor_args)
                
                # Update session state
                st.session_state.qdrant_manager = qdrant
                st.session_state.processor = processor
                st.session_state.search_initialized = True
                
                # Get collection info
                try:
                    collection_info = mps_safe_call(qdrant.get_collection_info)
                    if "points_count" in collection_info:
                        points_count = collection_info["points_count"]
                        if points_count > 0:
                            st.success(f"✅ Connected to collection '{collection_name}' with {points_count} points.")
                        else:
                            st.warning(f"Connected to collection '{collection_name}', but it's empty. Please index documents.")
                    else:
                        st.success(f"✅ Connected to collection '{collection_name}'.")
                except Exception as e:
                    st.warning(f"Connected but couldn't get collection info: {e}")
                
                # Force refresh to update UI
                st.rerun()
                
            except Exception as e:
                st.error(f"Error initializing search: {str(e)}")
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
    
    # Active search UI
    else:
        # Search interface
        st.write("### Enter your search query")
        
        # Create search form with text input and button side by side
        col1, col2 = st.columns([3, 1])

        def handle_search_submitted():
            st.session_state.search_triggered = True
            st.session_state.current_page = 1
            st.session_state.last_query = st.session_state.search_text
        
        with col1:
            search_text = st.text_input("Search Query", key="search_text", on_change=handle_search_submitted)
        
        with col2:
            search_type = st.selectbox("Search Type", ["hybrid", "vector", "sparse", "keyword"])
        
        # Search button - now we check both button click and if Enter was pressed
        search_clicked = st.button("Search")

        # This flag will only be set once when Enter is pressed
        if "search_triggered" not in st.session_state:
            st.session_state.search_triggered = False

        if search_clicked or st.session_state.search_triggered:
            st.session_state.search_triggered = False

            query = st.session_state.search_text
            if not query:
                st.warning("Please enter a search query.")
            else:
                try:
                    with st.spinner("Searching..."):
                        # Get search results - ask for more results for pagination
                        total_results_to_fetch = max_results  # Get as many results for pagination
                        
                        results = mps_safe_call(st.session_state.qdrant_manager.search,
                            query=query,
                            search_type=search_type,
                            limit=total_results_to_fetch,
                            processor=st.session_state.processor,
                            prefetch_limit=prefetch_limit,
                            fusion_type=fusion_type,
                            relevance_tuning=True,
                            context_size=st.session_state.context_size,
                            rerank=rerank
                        )

                        # Optional sorting by score
                        if sort_by_score:
                            results['results'].sort(key=lambda x: x['score'], reverse=True)

                        # Re-assign rank after sorting
                        for i, r in enumerate(results['results']):
                            r['rank'] = i + 1
                        
                        # Store results in session state
                        st.session_state.search_results = results
                        
                        # Store total results count for pagination
                        if "count" in results:
                            st.session_state.total_results = results["count"]
                        else:
                            st.session_state.total_results = 0
                        
                        # Store last successful query
                        st.session_state.last_query = query
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
                    # Check if it's a collection not found error and provide specific help
                    if "Collection" in str(e) and "does not exist" in str(e):
                        st.error("The collection doesn't exist. Try initializing search again or check your collection name.")
                        
                        # Reset the initialization flag to force re-initialization
                        st.session_state.search_initialized = False
                    
                    if st.session_state.show_debug:
                        import traceback
                        st.error(traceback.format_exc())
        
        # Display search results
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            if "error" in results:
                st.error(f"Search error: {results['error']}")
            else:
                # Search info
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>Query:</strong> {results['query']}</p>
                    <p><strong>Search type:</strong> {results['search_type']}</p>
                    <p><strong>Total results:</strong> {results['count']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if results['count'] == 0:
                    st.warning("No results found for your query.")
                else:
                    # Pagination logic
                    total_pages = (results['count'] + st.session_state.search_limit - 1) // st.session_state.search_limit
                    
                    # Make sure current page is within bounds
                    if st.session_state.current_page < 1:
                        st.session_state.current_page = 1
                    if st.session_state.current_page > total_pages:
                        st.session_state.current_page = total_pages
                    
                    # Calculate start and end indices for current page
                    start_idx = (st.session_state.current_page - 1) * st.session_state.search_limit
                    end_idx = min(start_idx + st.session_state.search_limit, results['count'])
                    
                    # Display page info
                    st.markdown(f"""
                    <div class="page-info">
                        Showing results {start_idx + 1}-{end_idx} of {results['count']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get current page results
                    page_results = results['results'][start_idx:end_idx]
                    
                    # Display each result
                    for result in page_results:
                        # Format the preview by replacing markdown ** with HTML span for highlighting
                        preview = result['preview']
                        preview = preview.replace('**', '<span class="highlight">')
                        preview = preview.replace('**', '</span>')
                        
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="filename">{result['file_name']} - Rank: {result['rank']}, Score: {result['score']:.4f}</div>
                            <div class="meta-info">Path: {result['file_path']}</div>
                            <div class="meta-info">Chunk: {result['chunk_index']} ({result['chunk_size']['characters']} chars, {result['chunk_size']['words']} words)</div>
                            <div class="preview-text">{preview}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expand to show full text
                        with st.expander("Show bigger context"):
                            #st.text_area("Full context", result['text'], height=300)
                            st.text_area("Full Context", result["big_context"], height=500)
                            
                            # Add button to open file in default application
                            file_path = result['file_path']
                            if os.path.exists(file_path):
                                if st.button(f"Open File", key=f"open_{result['id']}"):
                                    try:
                                        import subprocess
                                        import platform
                                        
                                        # Determine the operating system and open file accordingly
                                        system = platform.system()
                                        if system == 'Darwin':  # macOS
                                            subprocess.run(['open', file_path])
                                        elif system == 'Windows':
                                            subprocess.run(['start', file_path], shell=True)
                                        else:  # Linux and others
                                            subprocess.run(['xdg-open', file_path])
                                            
                                        st.success(f"Opening {file_path}")
                                    except Exception as e:
                                        st.error(f"Error opening file: {e}")
                    
                    # Pagination controls
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
                    
                    with col1:
                        if st.session_state.current_page > 1:
                            if st.button("First"):
                                st.session_state.current_page = 1
                                st.rerun()
                    
                    with col2:
                        if st.session_state.current_page > 1:
                            if st.button("Previous"):
                                st.session_state.current_page -= 1
                                st.rerun()
                    
                    with col3:
                        # Create a centered container for page numbers
                        st.markdown(f"""
                        <div class="centered">
                            <span>Page {st.session_state.current_page} of {total_pages}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        if st.session_state.current_page < total_pages:
                            if st.button("Next"):
                                st.session_state.current_page += 1
                                st.rerun()
                    
                    with col5:
                        if st.session_state.current_page < total_pages:
                            if st.button("Last"):
                                st.session_state.current_page = total_pages
                                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Qdrant Document Search powered by MLX and SPLADE</p>
    </div>
    """,
    unsafe_allow_html=True
)
