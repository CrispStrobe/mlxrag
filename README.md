# RAG Index and Search with MLX and SPLADE Support

A python based document indexing and search tool that combines dense and sparse vector embeddings for different search methods (vector, hybrid) using Qdrant as the vector database backend and streamlit for browser frontend.

## Overview

Provides an integrating framework for document indexing and retrieval making use of:

- **Dense Embeddings**: For semantic similarity using MLX and transformer models (BERT, BGE, E5, etc)
- **Sparse Embeddings**: Using SPLADE for lexical search / token-based retrieval with term expansion
- **Hybrid Search**: Combines dense and sparse approaches for optimal results
- **MLX Acceleration**: Uses Apple's MLX for efficient embedding generation on Apple Silicon
- **Document Processing**: Support for TXT, MD, HTML, PDF, JSON, CSV files, chunking with token limit enforcement, basic Metadata extraction and payload storage
 
A streamlit-based frontend user interface enables indexing and searching via browser.

## Requirements

- Python 3.7+
- Qdrant (either local or remote server)
- Optional but recommended:
  - MLX (for Apple Silicon acceleration)
  - mlx_embedding_models (for enhanced model support)
  - PyTorch (fallback when MLX is not available)
  - psutil (for memory monitoring)
  - tqdm (for progress bars)
  - streamlit

## Installation

```bash
# Basic installation
pip install qdrant-client transformers

# Recommended additional packages
pip install mlx mlx_embedding_models PyPDF2 tqdm psutil torch streamlit
```

## Usage

### Browser frontend

A streamlit-based search interface is available to simplify interactive exploration and retrieval from indexed documents.

```bash
streamlit run mlxrag_ui.py
```

### Indexing Documents

```bash
python mlxrag.py index documents_directory \
  --use-mlx-models \
  --dense-model bge-small \
  --sparse-model distilbert-splade \
  --collection my_documents
```

### Searching Documents

```bash
# Hybrid search (both dense and sparse)
python mlxrag.py search "your search query" \
  --search-type hybrid \
  --use-mlx-models

# Sparse-only search for lexical matching
python mlxrag.py search "exact terms to match" \
  --search-type sparse

# Dense vector search for semantic similarity
python mlxrag.py search "semantic concept" \
  --search-type vector
```

### List Available Models

```bash
python mlxrag.py list-models
```

## Command Line Arguments

### Global Arguments

- `--verbose`: Enable detailed logging
- `--storage-path`: Path for local Qdrant storage

### Index Command Arguments

- `directory`: Directory containing documents to index
- `--include`: File patterns to include (e.g., "*.pdf *.txt")
- `--limit`: Maximum number of files to index
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)
- `--collection`: Collection name
- `--model`: Fallback model name
- `--weights`: Path to model weights
- `--recreate`: Recreate collection if it exists
- `--dense-model`: MLX dense embedding model name
- `--sparse-model`: MLX sparse embedding model name
- `--top-k`: Top-k tokens for sparse vectors
- `--custom-repo-id`: Custom model repository ID
- `--custom-ndim`: Custom embedding dimension
- `--custom-pooling`: Custom pooling strategy (mean, first, max)
- `--custom-normalize`: Normalize embeddings
- `--custom-max-length`: Custom max sequence length

### Search Command Arguments

- `query`: Search query
- `--search-type`: Type of search (hybrid, vector, sparse, keyword)
- `--limit`: Maximum number of results
- `--prefetch-limit`: Prefetch limit for hybrid search
- `--fusion`: Fusion strategy (rrf, dbsf)
- `--relevance-tuning`: Apply relevance tuning
- `--context-size`: Size of context window for preview
- `--score-threshold`: Minimum score threshold for results
- `--debug`: Show detailed debug information
- `--no-color`: Disable colored output

## Vector Types and Models

### Dense Vector Models

- **bge-micro**: 3 layers, 384-dim
- **gte-tiny**: 6 layers, 384-dim
- **minilm-l6**: 6 layers, 384-dim
- **bge-small**: 12 layers, 384-dim
- **bge-base**: 12 layers, 768-dim
- **bge-large**: 24 layers, 1024-dim
- **snowflake-lg**: 24 layers, 1024-dim

### Sparse Vector Models (SPLADE)

- **distilbert-splade**: 6 layers
- **neuralcherche-sparse-embed**: 6 layers
- **opensearch**: 6 layers
- **naver-splade-distilbert**: 6 layers

## Examples

### Indexing Pipeline

```bash
# Index a directory with both dense and sparse embeddings
python mlxrag.py index ~/documents \
  --include "*.pdf *.txt *.md" \
  --use-mlx-models \
  --dense-model bge-small \
  --sparse-model distilbert-splade \
  --collection docs_collection \
  --verbose
```

### Advanced Search with Hybrid Approach

```bash
# Run a hybrid search with relevance tuning
python mlxrag.py search "quantum computing applications" \
  --search-type hybrid \
  --use-mlx-models \
  --dense-model bge-small \
  --sparse-model distilbert-splade \
  --collection docs_collection \
  --limit 15 \
  --prefetch-limit 50 \
  --fusion rrf \
  --context-size 400 \
  --relevance-tuning
```

### Using a Custom Model

```bash
# Index using a custom model from Hugging Face
python mlxrag.py index ~/documents \
  --use-mlx-models \
  --custom-repo-id "my-org/my-custom-model" \
  --custom-ndim 768 \
  --custom-pooling first \
  --collection custom_collection
```

## How It Works

1. **Document Processing**:
   - Files are extracted based on their format
   - Text is split into chunks respecting token limits
   - Each chunk is processed for embedding

2. **Vector Generation**:
   - Dense vectors capture semantic meaning
   - Sparse vectors capture lexical information with term expansion
   - Vectors are optimized for storage efficiency

3. **Indexing**:
   - Vectors are stored in Qdrant with appropriate configuration
   - Metadata is preserved for filtering and display

4. **Search**:
   - Queries are processed similarly to documents
   - Various search strategies can be employed
   - Results are ranked and formatted for display

## Sparse Vector Implementation

This implementation uses SPLADE (Sparse Lexical and Expansion Model) for generating sparse vectors, which provides several advantages over simple bag-of-words approaches:

- **Term Expansion**: Includes semantically related terms not in the original text
- **Learned Weights**: Assigns importance to terms based on context
- **Efficient Storage**: Only non-zero values are stored
- **Interpretable Results**: Each dimension corresponds to a specific token

## License

MIT

## Credits

This tool builds upon several open-source projects:
- Qdrant for vector storage and search
- MLX for efficient embedding computation (esp. mlx-examples, where we modified the BERT approach: https://github.com/CrispStrobe/mlx-examples/tree/main/bert)
- SPLADE for sparse vector generation
- Transformers for model loading and tokenization
- mlx_embedding_models
