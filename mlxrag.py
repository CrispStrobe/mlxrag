#!/usr/bin/env python3
"""
Enhanced Qdrant Indexer with MLX and SPLADE support.

This script combines document indexing and search capabilities with flexible model selection:
- MLX embedding models from registry or custom models
- MLX for efficient dense embedding generation
- SPLADE for sparse embedding generation
- Qdrant for vector storage and retrieval with multiple model support
"""

import argparse
import fnmatch
import os
import time
import json
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import tempfile
import requests
from collections import Counter
import re
import numpy as np

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Install with: pip install psutil")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

# Import Qdrant in dual-mode (HTTP + local support)
try:
    from qdrant_client import QdrantClient

    # HTTP-specific models
    from qdrant_client import models as qmodels
    from qdrant_client.http.models import Distance as HTTPDistance, VectorParams as HTTPVectorParams, MatchText
    from qdrant_client.http.exceptions import UnexpectedResponse

    # Local (non-HTTP) models
    from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, SparseVector
    from qdrant_client.models import models as qdrant_models

    qdrant_available = True
except ImportError:
    qdrant_available = False
    print("Qdrant not available. Install with: pip install qdrant-client")

# Try to import MLX - handle if not available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not available. Will use PyTorch for embeddings if available.")
    HAS_MLX = False

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Install with: pip install transformers")

# Try to import MLX embedding models
try:
    from mlx_embedding_models.embedding import EmbeddingModel, SpladeModel
    HAS_MLX_EMBEDDING_MODELS = True
except ImportError:
    HAS_MLX_EMBEDDING_MODELS = False
    print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")

# Try to import PyTorch
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    if not HAS_MLX:
        print("Warning: Neither MLX nor PyTorch is available. At least one is required for embedding generation.")

# Constants
DEFAULT_COLLECTION = "documents"
DEFAULT_MODEL = "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx"
DEFAULT_WEIGHTS_PATH = "weights/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_WEIGHTS_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_CONFIG_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/config.json"
MODEL_TOKENIZER_MAP = {
    "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# Default MLX embedding models
DEFAULT_DENSE_MODEL = "bge-small"
DEFAULT_SPARSE_MODEL = "distilbert-splade"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".json", ".csv"}
CHUNK_SIZE = 512  # Max tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
VECTOR_SIZE = 384  # Default vector size for the model


# MLX Embedding Models Registry (copy from mlx_embedding_models for reference)
MLX_EMBEDDING_REGISTRY = {
    # 3 layers, 384-dim
    "bge-micro": {
        "repo": "TaylorAI/bge-micro-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True, 
        "ndim": 384,
    },
    # 6 layers, 384-dim
    "gte-tiny": {
        "repo": "TaylorAI/gte-tiny",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "minilm-l6": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "snowflake-xs": {
        "repo": "Snowflake/snowflake-arctic-embed-xs",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 384-dim
    "minilm-l12": {
        "repo": "sentence-transformers/all-MiniLM-L12-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "bge-small": {
        "repo": "BAAI/bge-small-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first", # cls token, not pooler output
        "normalize": True,
        "ndim": 384,
    },
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 768-dim
    "bge-base": {
        "repo": "BAAI/bge-base-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1": {
        "repo": "nomic-ai/nomic-embed-text-v1",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1.5": {
        "repo": "nomic-ai/nomic-embed-text-v1.5",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
        "apply_ln": True,
    },
    # 24 layers, 1024-dim
    "bge-large": {
        "repo": "BAAI/bge-large-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024,
    },
    "snowflake-lg": {
        'repo': 'Snowflake/snowflake-arctic-embed-l',
        'max_length': 512,
        'pooling_strategy': 'first',
        'normalize': True,
        'ndim': 1024,
    },
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "max_length": 8192,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    "mixedbread-large": {
        "repo": 'mixedbread-ai/mxbai-embed-large-v1',
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    # SPARSE MODELS #
    "distilbert-splade": {
        "repo": "raphaelsty/distilbert-splade",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "neuralcherche-sparse-embed": {
        "repo": "raphaelsty/neural-cherche-sparse-embed",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "opensearch": {
        "repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "bert-base-uncased": { # mainly here as a baseline
        "repo": "bert-base-uncased",
        "max_length": 512,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "naver-splade-distilbert": {
        "repo": "naver/splade-v3-distilbert",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    }
}

# Add our custom model to the registry
MLX_EMBEDDING_REGISTRY["cstr-paraphrase-multilingual"] = {
    "repo": "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx",
    "max_length": 512,
    "pooling_strategy": "mean",
    "normalize": True,
    "ndim": 384,
}


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)
        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        self.layers = [TransformerEncoderLayer(dims, num_heads, mlp_dims) for _ in range(num_layers)]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array = None) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        token_types = self.token_type_embeddings(token_type_ids)
        embeddings = position + words + token_types
        return self.norm(embeddings)


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        pooled = mx.tanh(self.pooler(y[:, 0]))  # CLS token output
        return y, pooled


class MLXEmbeddingProvider:
    """Provider for dense and sparse embedding models from mlx_embedding_models or custom models"""
    
    def __init__(self, 
                dense_model_name: str = DEFAULT_DENSE_MODEL, 
                sparse_model_name: str = DEFAULT_SPARSE_MODEL,
                custom_repo_id: str = None,
                custom_ndim: int = None,
                custom_pooling: str = "mean",
                custom_normalize: bool = True,
                custom_max_length: int = 512,
                top_k: int = 64,
                batch_size: int = 16,
                verbose: bool = False):
        """
        Initialize MLXEmbeddingProvider with specified dense and sparse models.
        """
        self.verbose = verbose
        self.dense_model = None
        self.sparse_model = None
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.custom_repo_id = custom_repo_id
        self.custom_ndim = custom_ndim
        self.custom_pooling = custom_pooling
        self.custom_normalize = custom_normalize
        self.custom_max_length = custom_max_length
        self.batch_size = batch_size
        self.top_k = top_k
        
        # Use default dimension from registry, can be overridden for custom models
        if dense_model_name in MLX_EMBEDDING_REGISTRY:
            self.ndim = MLX_EMBEDDING_REGISTRY[dense_model_name]["ndim"]
        else:
            self.ndim = custom_ndim or 384  # Default if not specified
        
        if not HAS_MLX_EMBEDDING_MODELS:
            if verbose:
                print("mlx_embedding_models not available, skipping model loading")
            return
        
        try:
            # Load dense model
            if verbose:
                print(f"Loading dense embedding model: {dense_model_name}")
            
            # Handle custom model if specified
            if custom_repo_id:
                if verbose:
                    print(f"Using custom model repo: {custom_repo_id}")
                # Create model with custom parameters
                try:
                    self.dense_model = EmbeddingModel.from_pretrained(
                        custom_repo_id,
                        pooling_strategy=custom_pooling,
                        normalize=custom_normalize,
                        max_length=custom_max_length
                    )
                    if custom_ndim:
                        self.ndim = custom_ndim
                    else:
                        # Try to get dimension from model config, safely
                        try:
                            if hasattr(self.dense_model.model, 'config') and hasattr(self.dense_model.model.config, 'hidden_size'):
                                self.ndim = self.dense_model.model.config.hidden_size
                        except Exception as e:
                            if verbose:
                                print(f"Could not determine dimension from model: {e}")
                                print(f"Using default dimension: {self.ndim}")
                except Exception as e:
                    print(f"Error loading custom model {custom_repo_id}: {e}")
                    self.dense_model = None
            else:
                # Load from registry
                try:
                    self.dense_model = EmbeddingModel.from_registry(dense_model_name)
                    # Safely get dimension from model
                    if hasattr(self.dense_model.model, 'config') and hasattr(self.dense_model.model.config, 'hidden_size'):
                        self.ndim = self.dense_model.model.config.hidden_size
                except Exception as e:
                    print(f"Error loading model {dense_model_name} from registry: {e}")
                    self.dense_model = None
                    
            if self.dense_model and verbose:
                print(f"Loaded dense model with dimension: {self.ndim}")
            
            # Load sparse model
            if verbose:
                print(f"Loading sparse embedding model: {sparse_model_name}")
            try:
                self.sparse_model = SpladeModel.from_registry(sparse_model_name, top_k=top_k)
                if verbose:
                    print(f"Loaded SPLADE model with top-k: {top_k}")
            except Exception as e:
                print(f"Error loading SPLADE model: {e}")
                self.sparse_model = None
        except Exception as e:
            print(f"Error loading MLX embedding models: {e}")

    def debug_splade_model(self, text: str):
        """
        Debug the SPLADE model to understand what's happening with _sort_inputs
        """
        print("\n=== SPLADE Model Debugging ===")
        
        if self.sparse_model is None:
            print("No SPLADE model available for debugging")
            return
            
        print(f"SPLADE model type: {type(self.sparse_model)}")
        print(f"SPLADE model attributes: {dir(self.sparse_model)}")
        
        # Check the _sort_inputs method
        if hasattr(self.sparse_model, '_sort_inputs'):
            print(f"\n_sort_inputs method exists: {self.sparse_model._sort_inputs}")
            
            # Let's try to understand what it's doing
            print("\nInvestigating _sort_inputs internals:")
            import inspect
            if inspect.ismethod(self.sparse_model._sort_inputs):
                try:
                    source = inspect.getsource(self.sparse_model._sort_inputs)
                    print(f"Source code for _sort_inputs:\n{source}")
                except Exception as e:
                    print(f"Could not get source: {e}")
        else:
            print("No _sort_inputs method found directly on SPLADE model")
            
            # It might be inherited from a parent class
            if hasattr(self.sparse_model, '__class__') and hasattr(self.sparse_model.__class__, '__mro__'):
                print("\nChecking parent classes for _sort_inputs:")
                for cls in self.sparse_model.__class__.__mro__:
                    if hasattr(cls, '_sort_inputs'):
                        print(f"Found _sort_inputs in parent class: {cls.__name__}")
                        try:
                            source = inspect.getsource(cls._sort_inputs)
                            print(f"Source code for _sort_inputs in {cls.__name__}:\n{source}")
                        except Exception as e:
                            print(f"Could not get source from {cls.__name__}: {e}")
        
        # Try to understand what's happening in the tokenization step
        print("\nInvestigating tokenization and sort_inputs:")
        try:
            print(f"Input text: '{text}'")
            tokens = self.sparse_model._tokenize([text], min_length=self.sparse_model.min_query_length)
            print(f"Tokens type: {type(tokens)}")
            print(f"Tokens content: {tokens}")
            
            # Now try to see what _sort_inputs returns without unpacking
            try:
                sort_result = self.sparse_model._sort_inputs(tokens)
                print(f"\nResult from _sort_inputs (without unpacking):")
                print(f"Type: {type(sort_result)}")
                print(f"Value: {sort_result}")
                
                # If it's a sequence, check its length
                if hasattr(sort_result, '__len__'):
                    print(f"Length: {len(sort_result)}")
                    
                    # If it has multiple items, print each one
                    if len(sort_result) > 0:
                        print("Items:")
                        for i, item in enumerate(sort_result):
                            print(f"Item {i}: {type(item)} - {item}")
                
            except ValueError as ve:
                print(f"Original error reproduced: {ve}")
                # Try to get the actual return value without unpacking
                import sys
                try:
                    def mock_sort_inputs(tokens):
                        try:
                            result = self.sparse_model._sort_inputs(tokens)
                            print(f"CAPTURED RESULT: {result}")
                            return result
                        except Exception as e:
                            print(f"Error in _sort_inputs: {e}")
                            return None
                    
                    orig_sort_inputs = self.sparse_model._sort_inputs
                    self.sparse_model._sort_inputs = mock_sort_inputs
                    
                    # Try calling encode directly
                    try:
                        result = self.sparse_model.encode([text], batch_size=1)
                        print(f"Result from encode with mock: {result}")
                    except Exception as e:
                        print(f"Error in encode with mock: {e}")
                    
                    # Restore original method
                    self.sparse_model._sort_inputs = orig_sort_inputs
                except Exception as e:
                    print(f"Error in mock approach: {e}")
                
        except Exception as e:
            print(f"Error during tokenization: {e}")
        
        print("\n=== End of SPLADE Debugging ===")

    
    def get_dense_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get dense embeddings for text(s).
        
        Args:
            text: String or list of strings to encode
        
        Returns:
            Numpy array of embeddings with shape (batch_size, dimension)
        """
        if self.dense_model is None:
            raise RuntimeError("Dense model not loaded")
        
        # Handle input type
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Generate embeddings
        embeddings = self.dense_model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress=self.verbose
        )
        
        # Return single embedding if input was a string
        if isinstance(text, str):
            return embeddings[0]
        return embeddings
    
    def generate_fallback_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a simple fallback sparse vector using term frequencies.
        
        Args:
            text: Text to encode
            
        Returns:
            Tuple of (indices, values) for sparse representation
        """
        from collections import Counter
        import re
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 'that', 'it', 'with', 'for'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # Count terms
        counter = Counter(tokens)
        
        # Create sparse vector
        indices = []
        values = []
        
        for term, count in counter.items():
            term_index = hash(term) % 100000
            term_value = count / max(1, len(tokens))
            indices.append(term_index)
            values.append(term_value)
        
        # Handle empty case
        if not indices:
            return [0], [0.0]
            
        return indices, values
    
    def get_sparse_embedding_direct(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Get sparse embedding by directly working with the tokenizer and model,
        bypassing the problematic _sort_inputs method.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        try:
            # Check if we have a SPLADE model
            if self.sparse_model is None:
                raise ValueError("No SPLADE model available")
            
            # Check if the model has the required components
            if not hasattr(self.sparse_model, 'model') or not hasattr(self.sparse_model, 'tokenizer'):
                raise ValueError("SPLADE model is missing required components")
            
            # Manually tokenize the text
            inputs = self.sparse_model.tokenizer(
                [text],
                padding='max_length',
                truncation=True,
                max_length=self.sparse_model.max_length,
                return_tensors='np'
            )
            
            # Prepare batch for model
            batch = {}
            for k, v in inputs.items():
                # Convert numpy arrays to MLX arrays
                import mlx.core as mx
                batch[k] = mx.array(v)
            
            # Run the model manually
            mlm_output, _ = self.sparse_model.model(**batch)
            
            # Apply max pooling over sequence length dimension (dim=1)
            # Multiply by attention mask to ignore padding
            embs = mx.max(mlm_output * mx.expand_dims(batch["attention_mask"], -1), axis=1)
            
            # Apply SPLADE log(1+ReLU(x)) transformation
            embs = mx.log(1 + mx.maximum(embs, 0))
            
            # Apply top-k if needed
            if self.sparse_model.top_k > 0 and hasattr(self.sparse_model, '_create_sparse_embedding'):
                embs = self.sparse_model._create_sparse_embedding(embs, self.sparse_model.top_k)
            
            # Convert to numpy
            import numpy as np
            sparse_embs = np.array(mx.eval(embs), copy=False)
            
            # Extract nonzero components (the first result in the batch)
            embedding = sparse_embs[0]
            
            # Convert to sparse representation
            indices = []
            values = []
            
            for idx in range(len(embedding)):
                value = float(embedding[idx])
                if value > 0:
                    indices.append(idx)
                    values.append(value)
            
            # If empty, return default
            if not indices:
                return [0], [0.0]
            
            return indices, values
            
        except Exception as e:
            if self.verbose:
                print(f"Error in direct SPLADE processing: {e}")
            return self.generate_fallback_sparse_vector(text)

    def get_sparse_embedding_patched(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Get sparse embedding with a patched implementation of _sort_inputs.
        This replaces the problematic method with a simpler version that returns
        exactly what the encode method expects.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        try:
            # Check if we have a SPLADE model
            if self.sparse_model is None:
                raise ValueError("No SPLADE model available")
            
            # Define a simple _sort_inputs implementation
            def simple_sort_inputs(tokens):
                # Just return tokens and a range of indices
                import numpy as np
                return tokens, np.arange(len(tokens["input_ids"]))
            
            # Temporarily replace the problematic method
            original_sort_inputs = None
            if hasattr(self.sparse_model, '_sort_inputs'):
                original_sort_inputs = self.sparse_model._sort_inputs
                self.sparse_model._sort_inputs = simple_sort_inputs
            
            try:
                # Try to encode with our patched method
                sparse_embedding = self.sparse_model.encode([text], batch_size=1)
                
                # Process the result as before
                if isinstance(sparse_embedding, list):
                    if len(sparse_embedding) > 0:
                        sparse_embedding = sparse_embedding[0]
                    else:
                        return self.generate_fallback_sparse_vector(text)
                
                # Extract non-zero indices and values
                indices = []
                values = []
                
                # Handle array-like embeddings
                if hasattr(sparse_embedding, "shape") and hasattr(sparse_embedding, "__getitem__") and hasattr(sparse_embedding, "__len__"):
                    for idx in range(len(sparse_embedding)):
                        value = float(sparse_embedding[idx])
                        if value > 0:
                            indices.append(idx)
                            values.append(value)
                # Handle tuple format
                elif isinstance(sparse_embedding, tuple) and len(sparse_embedding) == 2:
                    indices, values = sparse_embedding
                else:
                    if self.verbose:
                        print(f"Unknown sparse embedding format: {type(sparse_embedding)}")
                    return self.generate_fallback_sparse_vector(text)
                
                # Handle empty results
                if not indices:
                    return [0], [0.0]
                
                return indices, values
                
            finally:
                # Restore the original method
                if original_sort_inputs is not None:
                    self.sparse_model._sort_inputs = original_sort_inputs
            
        except Exception as e:
            if self.verbose:
                print(f"Error in patched SPLADE processing: {e}")
            return self.generate_fallback_sparse_vector(text)
        
    def get_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Get sparse embedding for text as indices and values, with efficient
        token length handling and no warnings.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        # Handle empty text
        if not text.strip():
            return [0], [0.0]
            
        # If no SPLADE model, use fallback immediately
        if self.sparse_model is None:
            return self.generate_fallback_sparse_vector(text)
        
        # Patch tokenizer to silence warnings 
        # (Do this each time to ensure it's applied)
        if hasattr(self, 'sparse_model') and hasattr(self.sparse_model, 'tokenizer'):
            try:
                original_encode = None
                if hasattr(self.sparse_model.tokenizer, 'encode'):
                    original_encode = self.sparse_model.tokenizer.encode
                    
                    # Create a patched version that automatically truncates
                    def patched_encode(text, *args, **kwargs):
                        # If truncation not explicitly set, force it on
                        if 'truncation' not in kwargs:
                            kwargs['truncation'] = True
                            # Only set max_length if not already provided
                            if 'max_length' not in kwargs and hasattr(self.sparse_model, 'max_length'):
                                kwargs['max_length'] = self.sparse_model.max_length
                                
                        # Call original function with modified args
                        return original_encode(text, *args, **kwargs)
                    
                    # Apply the patch
                    self.sparse_model.tokenizer.encode = patched_encode
            except:
                pass
        
        # ===== TOKEN LENGTH HANDLING =====
        if hasattr(self.sparse_model, 'tokenizer') and hasattr(self.sparse_model, 'max_length'):
            try:
                # Check token length without warning
                tokens = self.sparse_model.tokenizer.encode(text, add_special_tokens=True, truncation=False)
                
                max_len = self.sparse_model.max_length
                
                if len(tokens) > max_len:
                    # Only log detailed info in verbose mode
                    if self.verbose:
                        print(f"\nTruncating text from {len(tokens)} tokens to {max_len}")
                    
                    # Intelligent truncation:
                    # 1. First try to respect sentence boundaries if possible
                    try:
                        # Simple sentence splitting 
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        
                        # Build up text until we approach the limit
                        truncated_text = ""
                        current_len = 0
                        safe_max = max_len - 5  # Leave some room for special tokens
                        
                        for sentence in sentences:
                            # Check how many new tokens this sentence would add
                            sent_tokens = self.sparse_model.tokenizer.encode(sentence, add_special_tokens=False)
                            
                            # Stop if adding this sentence would exceed the limit
                            if current_len + len(sent_tokens) > safe_max:
                                break
                                
                            truncated_text += sentence + " "
                            current_len += len(sent_tokens)
                        
                        # If we managed to include at least some text, use it
                        if truncated_text and len(truncated_text) > len(text) / 10:  # At least 10% of original
                            text = truncated_text
                        else:
                            # Fall back to simple truncation
                            tokens = tokens[:max_len]
                            text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                    except:
                        # Fall back to simple truncation if sentence splitting fails
                        tokens = tokens[:max_len]
                        text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                    
                    # Verify truncation worked (silently)
                    try:
                        new_tokens = self.sparse_model.tokenizer.encode(text, add_special_tokens=True)
                        
                        if len(new_tokens) > max_len:
                            # If still too long, do direct token truncation
                            tokens = new_tokens[:max_len]
                            text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                    except:
                        # Last resort: character-based truncation
                        char_ratio = max_len / len(tokens)
                        char_limit = int(len(text) * char_ratio * 0.9)  # 10% safety margin
                        text = text[:char_limit]
            except Exception as e:
                if self.verbose:
                    print(f"Token handling error (will continue): {e}")
        
        # ===== SPLADE ENCODING WITH PATCHED _sort_inputs =====        
        # Monkey patch the _sort_inputs method
        original_sort_inputs = None
        if hasattr(self.sparse_model, '_sort_inputs'):
            original_sort_inputs = self.sparse_model._sort_inputs
            
            def patched_sort_inputs(tokens):
                result = original_sort_inputs(tokens)
                if isinstance(result, tuple) and len(result) == 3:
                    return result[0], result[1]
                return result
                
            self.sparse_model._sort_inputs = patched_sort_inputs
        
        try:
            # Encode with our patched method
            try:
                # Disable tqdm for cleaner output
                import tqdm.auto
                original_tqdm = tqdm.auto.tqdm
                
                # Create a dummy tqdm that does nothing
                class DummyTQDM:
                    def __init__(self, *args, **kwargs):
                        self.total = kwargs.get('total', 0)
                        self.n = 0
                    def update(self, n=1): self.n += n
                    def close(self): pass
                    def __enter__(self): return self
                    def __exit__(self, *args, **kwargs): pass
                    def set_postfix(self, *args, **kwargs): pass
                
                # Replace tqdm with our dummy version
                tqdm.auto.tqdm = DummyTQDM
                
                # Now encode with warnings suppressed
                sparse_embedding = self.sparse_model.encode([text], batch_size=1, show_progress=False)
                
                # Restore original tqdm
                tqdm.auto.tqdm = original_tqdm
                
            except Exception as e:
                if self.verbose:
                    print(f"Error encoding with SPLADE: {e}")
                return self.generate_fallback_sparse_vector(text)
            
            # Process the result
            if isinstance(sparse_embedding, list):
                if len(sparse_embedding) > 0:
                    sparse_embedding = sparse_embedding[0]
                else:
                    return self.generate_fallback_sparse_vector(text)
            elif sparse_embedding is None:
                return self.generate_fallback_sparse_vector(text)
            
            # Extract non-zero indices and values efficiently
            indices = []
            values = []
            
            if hasattr(sparse_embedding, "shape"):
                import numpy as np
                
                # Flatten if needed
                if len(sparse_embedding.shape) > 1:
                    sparse_embedding = np.ravel(sparse_embedding)
                
                # Get non-zero indices directly
                nonzero_indices = np.nonzero(sparse_embedding > 0)[0]
                for idx in nonzero_indices:
                    try:
                        value = float(sparse_embedding[idx])
                        indices.append(int(idx))
                        values.append(value)
                    except:
                        continue
            elif isinstance(sparse_embedding, tuple) and len(sparse_embedding) == 2:
                indices, values = sparse_embedding
            else:
                if self.verbose:
                    print(f"Unknown sparse embedding format: {type(sparse_embedding)}")
                return self.generate_fallback_sparse_vector(text)
            
            # Handle empty results
            if not indices:
                return [0], [0.0]
            
            return indices, values
            
        except Exception as e:
            if self.verbose:
                print(f"SPLADE model error: {e}. Using fallback sparse encoding.")
            return self.generate_fallback_sparse_vector(text)
        finally:
            # Restore the original methods
            if original_sort_inputs is not None:
                self.sparse_model._sort_inputs = original_sort_inputs
                
            # Restore original tokenizer.encode if we patched it
            if hasattr(self.sparse_model, 'tokenizer') and hasattr(self.sparse_model.tokenizer, 'encode'):
                if 'original_encode' in locals() and original_encode is not None:
                    self.sparse_model.tokenizer.encode = original_encode


    def get_sparse_embeddings_batch(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """
        Get sparse embeddings for multiple texts without nested progress bars.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            List of (indices, values) tuples for sparse vector representation
        """
        # Handle empty input
        if not texts:
            return []
        
        # If no SPLADE model, use individual fallback immediately
        if self.sparse_model is None:
            return [self.generate_fallback_sparse_vector(text) for text in texts]
        
        # Process individually but without showing nested progress bars
        results = []
        for text in texts:
            result = self.get_sparse_embedding(text)
            results.append(result)
        
        return results

    def patch_splade_model(self):
        """
        Apply a direct patch to the SPLADE model to fix the unpacking error.
        Call this once after initialization if needed.
        """
        if not hasattr(self, 'sparse_model') or self.sparse_model is None:
            return
            
        # Save original method
        if not hasattr(self, '_original_splade_encode'):
            self._original_splade_encode = self.sparse_model.encode
        
        # Define patched method
        def patched_encode(sentences, batch_size=16, show_progress=True, **kwargs):
            # First patch the _sort_inputs method
            original_sort_inputs = self.sparse_model._sort_inputs
            
            def fixed_sort_inputs(tokens):
                result = original_sort_inputs(tokens)
                if isinstance(result, tuple) and len(result) == 3:
                    return result[0], result[1]  # Return only the first two values
                return result
                
            self.sparse_model._sort_inputs = fixed_sort_inputs
            
            try:
                # Call original with patched _sort_inputs
                return self._original_splade_encode(sentences, batch_size, show_progress, **kwargs)
            finally:
                # Restore original method
                self.sparse_model._sort_inputs = original_sort_inputs
        
        # Apply the patch
        self.sparse_model.encode = patched_encode
        
        if self.verbose:
            print("SPLADE model patched to fix unpacking error")
        
    def direct_sparse_encoding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Directly compute SPLADE vectors without using the problematic encode method.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        try:
            import mlx.core as mx
            import numpy as np
            
            # Check if we have a model
            if not hasattr(self, 'sparse_model') or self.sparse_model is None:
                raise ValueError("No SPLADE model available")
                
            # Tokenize using the model's tokenizer
            tokens = self.sparse_model.tokenizer([text], 
                                            padding="max_length", 
                                            truncation=True, 
                                            max_length=self.sparse_model.max_length,
                                            return_tensors="np")
            
            # Convert to MLX arrays
            mlx_inputs = {k: mx.array(v) for k, v in tokens.items()}
            
            # Run the model
            mlm_output, _ = self.sparse_model.model(**mlx_inputs)
            
            # Apply max pooling and SPLADE transformation
            attention_mask = mlx_inputs.get("attention_mask")
            embs = mx.max(mlm_output * mx.expand_dims(attention_mask, -1), axis=1)
            embs = mx.log(1 + mx.maximum(embs, 0))
            
            # Apply top-k if needed
            if hasattr(self.sparse_model, "top_k") and self.sparse_model.top_k > 0:
                if hasattr(self.sparse_model, "_create_sparse_embedding"):
                    embs = self.sparse_model._create_sparse_embedding(embs, self.sparse_model.top_k)
            
            # Convert to numpy and extract indices/values
            embs_np = np.array(mx.eval(embs))[0]  # First element in batch
            
            # Convert to sparse representation
            indices = []
            values = []
            
            for idx in range(len(embs_np)):
                value = float(embs_np[idx])
                if value > 0:
                    indices.append(idx)
                    values.append(value)
            
            # If no positive values, return a default
            if not indices:
                return [0], [0.0]
                
            return indices, values
            
        except Exception as e:
            if self.verbose:
                print(f"Error in direct SPLADE encoding: {e}")
            return self.generate_fallback_sparse_vector(text)

def get_size_str(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download_file(url, local_path, verbose=False, retries=3):
    """Download a file with progress bar and retry support"""
    import time

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(retries):
        try:
            if verbose:
                print(f"Downloading from {url} to {local_path} (attempt {attempt+1})")

            with requests.get(url, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code} - {response.reason}")

                file_size = int(response.headers.get('Content-Length', 0))
                chunk_size = 8192

                with open(local_path, 'wb') as f:
                    if verbose and HAS_TQDM:
                        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(local_path))
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                progress.update(len(chunk))
                        progress.close()
                    else:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)

            # Check for LFS placeholder (small files starting with version info)
            if os.path.getsize(local_path) < 1000:
                with open(local_path, 'rb') as f:
                    head = f.read(200)
                    if b"oid sha256" in head:
                        raise Exception("Downloaded LFS pointer file instead of real model. Check if LFS is enabled.")
            return  # Success

        except Exception as e:
            if verbose:
                print(f"Download failed: {e}")
            time.sleep(1)

    raise Exception(f"Failed to download file after {retries} attempts: {url}")


def download_model_files(model_name, weights_path, verbose=False):
    """Download model weights and config from HuggingFace"""
    weights_dir = os.path.dirname(weights_path)
    if not os.path.exists(weights_dir):
        if verbose:
            print(f"Creating weights directory: {weights_dir}")
        os.makedirs(weights_dir, exist_ok=True)

    # Check if we need to download weights
    if not os.path.exists(weights_path):
        if verbose:
            print(f"Model weights not found at {weights_path}. Downloading...")
        
        # If model is hosted on HuggingFace, download direct link
        if model_name.startswith(("cstr/", "sentence-transformers/")):
            weights_url = DEFAULT_WEIGHTS_URL
            download_file(weights_url, weights_path, verbose)
        else:
            # For other models, we'd need to convert them first (not implemented here)
            raise ValueError(f"Direct download only supported for specific models. Please download weights manually for {model_name}")
    elif verbose:
        print(f"Model weights already exist at: {weights_path}")
    
    # Download config if needed
    config_path = os.path.join(os.path.dirname(weights_path), "config.json")
    if not os.path.exists(config_path):
        if verbose:
            print(f"Downloading model config to {config_path}")
        download_file(DEFAULT_CONFIG_URL, config_path, verbose)
    elif verbose:
        print(f"Model config already exists at: {config_path}")


class MLXModel:
    """Wrapper for MLX model operations using custom BertModel"""
    
    def __init__(self, weights_path, verbose=False):
        self.weights_path = weights_path
        self.verbose = verbose
        self.config = None
        self.model = None
        self.loaded = False

    def load(self):
        """Load config and weights into MLX model"""
        if self.loaded:
            return
            
        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(self.weights_path), "config.json")
            if self.verbose:
                print(f"Loading config from {config_path}")
                
            # Check if there's a local config
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    
                # Create a config object manually to avoid HuggingFace lookup
                class Config:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                
                config = Config(**config_dict)
                
                if self.verbose:
                    print(f"Loaded local config from {config_path}")
            else:
                # If no config available, raise error
                raise ValueError(f"No config file found at {config_path}")
            
            if self.verbose:
                print(f"Creating model with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
                
            # Create the model
            self.model = BertModel(config)
            
            if self.verbose:
                print(f"Loading weights from {self.weights_path}")
            
            # Load weights directly
            weights = np.load(self.weights_path)
            params = {k: mx.array(v) for k, v in weights.items()}
            
            # Use MLX's standard method for updating parameters
            self.model.update(params)
            
            if self.verbose:
                print("Weights loaded successfully")
                print("Running dummy forward pass tests...")

                dummy_input = {
                    "input_ids": mx.array([[101, 102]]),         # [CLS] [SEP]
                    "attention_mask": mx.array([[1, 1]])
                }

                try:
                    _, pooled = self.model(**dummy_input)
                    mx.eval(pooled)
                    print("✅ Dummy forward pass successful. Output shape:", pooled.shape)
                except Exception as e:
                    print("❌ Forward pass failed:", str(e))
                
            self.config = config
            self.loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def get_embedding(self, tokenized_input) -> np.ndarray:
        """Compute embedding from tokenized input"""
        input_ids = mx.array(tokenized_input["input_ids"])
        attention_mask = mx.array(tokenized_input["attention_mask"])
        
        # Forward pass through the model
        _, pooled_output = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        
        # Convert to numpy array
        return mx.eval(pooled_output)[0]


class TextExtractor:
    """Extract text from various file formats"""
    
    @staticmethod
    def extract_from_file(file_path: str, verbose: bool = False) -> str:
        """Extract text from a file based on its extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if verbose:
            print(f"Extracting text from: {file_path} (format: {ext})")
            
        if ext == ".txt" or ext == ".md" or ext == ".html":
            return TextExtractor._extract_from_text_file(file_path)
        elif ext == ".pdf":
            return TextExtractor._extract_from_pdf(file_path)
        elif ext == ".json":
            return TextExtractor._extract_from_json(file_path)
        elif ext == ".csv":
            return TextExtractor._extract_from_csv(file_path)
        else:
            if verbose:
                print(f"Unsupported file format: {ext}. Skipping.")
            return ""
    
    @staticmethod
    def _extract_from_text_file(file_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                return ""
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            return text
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_json(file_path: str) -> str:
        """Extract text from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error extracting text from JSON {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_csv(file_path: str) -> str:
        """Extract text from a CSV file"""
        try:
            import csv
            text = ""
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    text += " ".join(row) + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from CSV {file_path}: {str(e)}")
            return ""

class ChunkPreparation:
    """text chunking to strictly respect token limits"""
    
    @staticmethod
    def prepare_chunks(text: str, tokenizer: Any, max_length: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks suitable for embedding,
        strictly enforcing token limits.
        
        Args:
            text: Text to split into chunks
            tokenizer: Tokenizer to use for token counting
            max_length: Maximum tokens per chunk (including special tokens)
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # CRITICAL: Adjust for special tokens
        special_tokens_count = 2  # Most BERT-based models add [CLS] and [SEP]
        effective_max_length = max_length - special_tokens_count
        
        chunks = []
        
        # For very long texts, first split by paragraphs 
        if len(text) > 100000:
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            
            # Process each paragraph separately
            for i, paragraph in enumerate(paragraphs):
                paragraph_chunks = ChunkPreparation._chunk_text(
                    paragraph, tokenizer, effective_max_length, overlap
                )
                
                # Add paragraph index to chunks for reference
                for chunk in paragraph_chunks:
                    chunk["paragraph_idx"] = i
                    chunks.append(chunk)
                    
            return ChunkPreparation._verify_chunks(chunks, tokenizer, max_length)
        
        # For shorter texts, process the entire text
        chunks = ChunkPreparation._chunk_text(text, tokenizer, effective_max_length, overlap)
        return ChunkPreparation._verify_chunks(chunks, tokenizer, max_length)
    
    @staticmethod
    def _chunk_text(text: str, tokenizer: Any, max_length: int = 510, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into chunks with improved token limit enforcement
        
        Args:
            text: Text to chunk
            tokenizer: Tokenizer for counting tokens
            max_length: Maximum tokens per chunk (excluding special tokens)
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        # Split into sentences for more natural chunking
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = []
        
        for sentence in sentences:
            # Get tokens for this sentence (without special tokens)
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            
            # If a single sentence is too long, split it by words
            if len(sentence_tokens) > max_length:
                # Process any accumulated sentences first
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                    
                    chunks.append({
                        "text": chunk_text,
                        "token_count": len(chunk_tokens),
                        "start_idx": 0,
                        "end_idx": 0
                    })
                    
                    current_chunk_sentences = []
                    current_chunk_tokens = []
                
                # Split the long sentence into word-level chunks
                words = sentence.split()
                current_piece = []
                current_piece_tokens = []
                
                for word in words:
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    
                    # If adding this word would exceed the limit, save the current piece
                    if len(current_piece_tokens) + len(word_tokens) > max_length:
                        if current_piece:
                            piece_text = " ".join(current_piece)
                            # Add special tokens for final count
                            piece_tokens = tokenizer.encode(piece_text, add_special_tokens=True)
                            
                            chunks.append({
                                "text": piece_text,
                                "token_count": len(piece_tokens),
                                "start_idx": 0,
                                "end_idx": 0
                            })
                            
                            # Keep some overlap for context
                            overlap_tokens = min(overlap, len(current_piece_tokens))
                            if overlap_tokens > 0:
                                # Keep last few words for overlap
                                overlap_word_count = max(1, len(current_piece) // 3)
                                current_piece = current_piece[-overlap_word_count:]
                                # Recalculate tokens for the overlap portion
                                overlap_text = " ".join(current_piece)
                                current_piece_tokens = tokenizer.encode(overlap_text, add_special_tokens=False)
                            else:
                                current_piece = []
                                current_piece_tokens = []
                        
                        # If the word itself is too long (rare), we must truncate it
                        if len(word_tokens) > max_length:
                            trunc_word = word[:int(len(word) * max_length / len(word_tokens))]
                            current_piece.append(trunc_word)
                            current_piece_tokens = tokenizer.encode(" ".join(current_piece), add_special_tokens=False)
                        else:
                            current_piece.append(word)
                            current_piece_tokens = tokenizer.encode(" ".join(current_piece), add_special_tokens=False)
                    else:
                        current_piece.append(word)
                        current_piece_tokens = tokenizer.encode(" ".join(current_piece), add_special_tokens=False)
                
                # Add the last piece if not empty
                if current_piece:
                    piece_text = " ".join(current_piece)
                    piece_tokens = tokenizer.encode(piece_text, add_special_tokens=True)
                    
                    chunks.append({
                        "text": piece_text,
                        "token_count": len(piece_tokens),
                        "start_idx": 0,
                        "end_idx": 0
                    })
                
                # Continue with the next sentence
                current_chunk_sentences = []
                current_chunk_tokens = []
            
            # For normal-length sentences, add to current chunk if it fits
            elif len(current_chunk_tokens) + len(sentence_tokens) <= max_length:
                current_chunk_sentences.append(sentence)
                # Recalculate tokens to account for spacing and interaction between tokens
                current_chunk_tokens = tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)
            else:
                # Current chunk is full, save it
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    # Add special tokens for final count
                    chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                    
                    chunks.append({
                        "text": chunk_text,
                        "token_count": len(chunk_tokens),
                        "start_idx": 0,
                        "end_idx": 0
                    })
                
                # Start a new chunk with overlap
                if overlap > 0 and len(current_chunk_sentences) > 0:
                    # Calculate overlap sentences (keep approximately 1/3 of previous sentences)
                    overlap_sentence_count = max(1, len(current_chunk_sentences) // 3)
                    overlap_sentences = current_chunk_sentences[-overlap_sentence_count:]
                    
                    # Start new chunk with overlap sentences
                    current_chunk_sentences = overlap_sentences + [sentence]
                    # Recalculate tokens
                    current_chunk_tokens = tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)
                else:
                    # No overlap, just start with current sentence
                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sentence_tokens
        
        # Add the last chunk if not empty
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            # Add special tokens for final count
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
            
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_idx": 0,
                "end_idx": 0
            })
        
        return chunks
    
    @staticmethod
    def _verify_chunks(chunks: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> List[Dict[str, Any]]:
        """
        Verify all chunks are under the token limit and fix any that aren't.
        """
        verified_chunks = []
        
        for chunk in chunks:
            # Re-tokenize to ensure accurate count
            tokens = tokenizer.encode(chunk["text"], add_special_tokens=True)
            
            # If chunk is within limit, add it as is
            if len(tokens) <= max_length:
                chunk["token_count"] = len(tokens)  # Update the token count
                verified_chunks.append(chunk)
            else:
                # For chunks that are still too big, force truncation
                print(f"Warning: Chunk with {len(tokens)} tokens exceeds limit ({max_length}). Truncating.")
                
                # Truncate tokens and decode back to text
                truncated_tokens = tokens[:max_length]
                if hasattr(tokenizer, "decode"):
                    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                    
                    # Create new chunk with truncated text
                    verified_chunks.append({
                        "text": truncated_text,
                        "token_count": len(truncated_tokens),
                        "start_idx": chunk.get("start_idx", 0),
                        "end_idx": chunk.get("end_idx", 0),
                        "paragraph_idx": chunk.get("paragraph_idx", None),
                        "truncated": True
                    })
                else:
                    # If tokenizer doesn't support decode, use character-based truncation
                    ratio = max_length / len(tokens)
                    char_limit = int(len(chunk["text"]) * ratio * 0.9)  # 10% safety margin
                    truncated_text = chunk["text"][:char_limit]
                    
                    # Verify truncation worked
                    check_tokens = tokenizer.encode(truncated_text, add_special_tokens=True)
                    if len(check_tokens) > max_length:
                        # Further truncate if still too long
                        char_limit = int(char_limit * max_length / len(check_tokens) * 0.9)
                        truncated_text = chunk["text"][:char_limit]
                    
                    verified_chunks.append({
                        "text": truncated_text,
                        "token_count": min(max_length, len(check_tokens)),
                        "start_idx": chunk.get("start_idx", 0),
                        "end_idx": chunk.get("end_idx", 0),
                        "paragraph_idx": chunk.get("paragraph_idx", None),
                        "truncated": True
                    })
        
        return verified_chunks

class DocumentProcessor:
    """Process documents for indexing with support for both traditional models and MLX embedding models"""
    
    def __init__(self, 
                model_name: str, 
                weights_path: str, 
                dense_model: str = DEFAULT_DENSE_MODEL, 
                sparse_model: str = DEFAULT_SPARSE_MODEL,
                top_k: int = 64,
                custom_repo_id: str = None,
                custom_ndim: int = None,
                custom_pooling: str = "mean",
                custom_normalize: bool = True,
                custom_max_length: int = 512,
                use_mlx_embedding: bool = True,
                verbose: bool = False):
        """
        Initialize DocumentProcessor with MLX Embedding Models support.
        
        Args:
            model_name: Original model name (fallback if MLX embedding models not available)
            weights_path: Path to model weights (fallback)
            dense_model: MLX embedding model name for dense vectors
            sparse_model: MLX embedding model name for sparse vectors
            top_k: Top-k tokens to keep in sparse vectors
            custom_repo_id: Custom model repo ID
            custom_ndim: Custom model embedding dimension
            custom_pooling: Custom model pooling strategy
            custom_normalize: Whether to normalize embeddings
            custom_max_length: Maximum sequence length
            use_mlx_embedding: Whether to use MLX embedding models
            verbose: Whether to show verbose output
        """
        self.verbose = verbose
        self.model_name = model_name
        self.weights_path = weights_path
        self.tokenizer = None
        self.pytorch_model = None
        self.mlx_model = None
        self.use_mlx = HAS_MLX
        self.vector_size = VECTOR_SIZE  # Default, will be updated during load_model
        
        # Initialize MLX embedding provider if available
        self.use_mlx_embedding = use_mlx_embedding and HAS_MLX_EMBEDDING_MODELS
        self.mlx_embedding_provider = None
        
        # Store model identifiers for Qdrant vector names
        self.dense_model_id = dense_model
        self.sparse_model_id = sparse_model
        
        if self.use_mlx_embedding:
            self.mlx_embedding_provider = MLXEmbeddingProvider(
                dense_model_name=dense_model,
                sparse_model_name=sparse_model,
                custom_repo_id=custom_repo_id,
                custom_ndim=custom_ndim,
                custom_pooling=custom_pooling,
                custom_normalize=custom_normalize,
                custom_max_length=custom_max_length,
                top_k=top_k,
                verbose=verbose
            )
            self.mlx_embedding_provider.patch_splade_model()
            
            # Set vector size from embedding model
            if self.mlx_embedding_provider.dense_model is not None:
                self.vector_size = self.mlx_embedding_provider.ndim
                if verbose:
                    print(f"Using vector size from MLX embedding model: {self.vector_size}")
        
        # Always load tokenizer and models even if using MLX embedding models as fallback
        self.load_model()
        
        # Update vector size in case MLX embedding model was used
        if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider and hasattr(self.mlx_embedding_provider, 'ndim'):
            self.vector_size = self.mlx_embedding_provider.ndim
        
    def load_model(self) -> None:
        """Load the model (PyTorch or MLX based on availability)"""
        try:
            if self.verbose:
                print(f"Loading model {self.model_name}")
                
            start_time = time.time()
            
            # Load tokenizer from HuggingFace
            resolved_tokenizer = MODEL_TOKENIZER_MAP.get(self.model_name, self.model_name)
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers package is required. Install with: pip install transformers")
            
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer)

            if self.use_mlx:
                # Load MLX model
                if self.verbose:
                    print(f"Using MLX model with weights from {self.weights_path}")
                self.mlx_model = MLXModel(self.weights_path, self.verbose)
                self.mlx_model.load()
                self.vector_size = self.mlx_model.config.hidden_size
            else:
                # Load PyTorch model
                if self.verbose:
                    print(f"Using PyTorch model from {self.model_name}")
                if not HAS_PYTORCH:
                    raise ImportError("PyTorch is required when MLX is not available. Install with: pip install torch")
                self.pytorch_model = AutoModel.from_pretrained(self.model_name)
                config = AutoConfig.from_pretrained(self.model_name)
                self.vector_size = config.hidden_size
            
            if self.verbose:
                print(f"Model loaded in {time.time() - start_time:.2f} seconds")
                print(f"Model embedding size: {self.vector_size}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _generate_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a simple sparse vector using term frequencies (internal fallback method).
        
        This method is a fallback for when the MLX embedding provider is not available
        or encounters errors.
        """
        from collections import Counter
        import re
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very common words (simple stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 'that', 'it', 'with', 'for'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # Count terms
        counter = Counter(tokens)
        
        # Convert to sparse vector format (for simplicity, use term hashes as indices)
        indices = []
        values = []
        
        for term, count in counter.items():
            # Simple hash function for terms - use a better one in production
            # Limit to 100000 dimensions
            term_index = hash(term) % 100000
            term_value = count / max(1, len(tokens))  # Normalize by document length, avoid division by zero
            indices.append(term_index)
            values.append(term_value)
        
        # If empty, return a single default dimension
        if not indices:
            return [0], [0.0]
            
        return indices, values
            
    def get_embedding_pytorch(self, text: str) -> np.ndarray:
        """Get embedding for a text using PyTorch backend"""
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch is not installed. Cannot compute embeddings.")

        if not text.strip():
            if self.verbose:
                print("[PyTorch] Input text is empty. Returning zero vector.")
            return np.zeros(self.vector_size, dtype=np.float32)

        if self.pytorch_model is None:
            raise RuntimeError("PyTorch model is not loaded.")

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CHUNK_SIZE
        )

        try:
            with torch.no_grad():
                outputs = self.pytorch_model(**tokens)
                pooled_output = outputs.pooler_output[0].cpu().numpy()

            return pooled_output.astype(np.float32)

        except Exception as e:
            print(f"[PyTorch] Error during PyTorch embedding: {e}")
            return np.zeros(self.vector_size, dtype=np.float32)
        
    def get_embedding_mlx(self, text: str) -> np.ndarray:
        """Get embedding for a text using the MLX backend"""
        if not text.strip():
            if self.verbose:
                print("[MLX] Input text is empty. Returning zero vector.")
            return np.zeros(self.vector_size, dtype=np.float32)

        if self.mlx_model is None or not self.mlx_model.loaded:
            raise RuntimeError("MLX model is not loaded. Cannot compute embeddings.")

        tokenized = self.tokenizer(text, return_tensors="np", truncation=True, max_length=CHUNK_SIZE)

        try:
            # Convert input to MLX arrays
            input_ids = mx.array(tokenized["input_ids"])
            attention_mask = mx.array(tokenized["attention_mask"])
            
            # Get embeddings using the model
            _, pooled_output = self.mlx_model.model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            # Convert to numpy
            embedding_np = mx.eval(pooled_output)[0]
            
            return embedding_np.astype(np.float32)

        except Exception as e:
            print(f"[MLX] Error during MLX embedding: {e}")
            return np.zeros(self.vector_size, dtype=np.float32)
            
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text input using preferred backend"""
        text = text.strip()
        if not text:
            return np.zeros(self.vector_size, dtype=np.float32)

        # Try to use MLX embedding models first if available
        if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
            try:
                return self.mlx_embedding_provider.get_dense_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"Error using MLX embedding model: {e}")
                    print("Falling back to original embedding method")

        # Use MLX if available
        if self.use_mlx and self.mlx_model is not None:
            return self.get_embedding_mlx(text)

        # Fall back to PyTorch
        if self.pytorch_model is not None:
            return self.get_embedding_pytorch(text)

        raise RuntimeError("No valid model backend available (MLX or PyTorch).")
        
    def process_file(self, file_path: str) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Process a single file into chunks with embeddings (dense only)"""
        if self.verbose:
            print(f"Processing file: {file_path}")
            
        # Extract text from file
        text = TextExtractor.extract_from_file(file_path, self.verbose)
        if not text:
            if self.verbose:
                print(f"No text extracted from {file_path}")
            return []
            
        # Create chunks
        chunks = ChunkPreparation.prepare_chunks(text, self.tokenizer)
        if self.verbose:
            print(f"Created {len(chunks)} chunks from {file_path}")
            
        # Calculate embeddings for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            if self.verbose and (i == 0 or (i+1) % 10 == 0 or i == len(chunks) - 1):
                print(f"Calculating embedding for chunk {i+1}/{len(chunks)}")
                
            embedding = self.get_embedding(chunk["text"])
            
            # Create payload
            payload = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": {
                    "file_ext": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path),
                    "created_at": time.time(),
                    "embedder": self.dense_model_id
                }
            }
            
            results.append((embedding, payload))
            
        return results
    
    def process_file_with_sparse(self, file_path: str) -> List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]:
        """
        Process a file into chunks with both dense and sparse embeddings.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if self.verbose:
            print(f"Processing file: {file_path}")
            
        # Extract text from file
        text = TextExtractor.extract_from_file(file_path, self.verbose)
        if not text:
            if self.verbose:
                print(f"No text extracted from {file_path}")
            return []
            
        # Create chunks
        chunks = ChunkPreparation.prepare_chunks(text, self.tokenizer)
        if self.verbose:
            print(f"Created {len(chunks)} chunks from {file_path}")
        
        # Extract text from chunks for batch processing
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Process chunks in batches
        # Calculate ideal batch size based on number of chunks
        batch_size = min(50, max(1, len(chunks) // 5))  # Max 50, min 1, aim for ~5 batches
        
        # Track results
        results = []
        
        # Track success/failure
        success_count = 0
        error_count = 0
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_indices = list(range(batch_start, batch_end))
            batch_chunks = [chunks[i] for i in batch_indices]
            batch_texts = [chunk_texts[i] for i in batch_indices]
            
            if self.verbose and len(chunks) > batch_size:
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {batch_start+1}-{batch_end})")
            
            # Get dense embeddings for this batch
            dense_batch_embeddings = []
            try:
                if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
                    dense_batch_embeddings = self.mlx_embedding_provider.get_dense_embedding(batch_texts)
                    if self.verbose:
                        print(f"Generated dense embeddings for {len(batch_texts)} chunks")
            except Exception as e:
                if self.verbose:
                    print(f"Batch dense embedding failed: {e}")
            
            # Get sparse embeddings for this batch
            sparse_batch_embeddings = []
            try:
                if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
                    sparse_batch_embeddings = self.mlx_embedding_provider.get_sparse_embeddings_batch(batch_texts)
                    if self.verbose and sparse_batch_embeddings:
                        print(f"Generated sparse embeddings for {len(sparse_batch_embeddings)} chunks")
            except Exception as e:
                if self.verbose:
                    print(f"Batch sparse embedding failed: {e}")
            
            # Process each chunk in the batch
            for i, chunk_idx in enumerate(batch_indices):
                chunk = chunks[chunk_idx]
                
                try:
                    # Get dense embedding (from batch or individual)
                    if isinstance(dense_batch_embeddings, list) and len(dense_batch_embeddings) > i:
                        dense_embedding = dense_batch_embeddings[i]
                    elif isinstance(dense_batch_embeddings, np.ndarray) and dense_batch_embeddings.ndim == 2 and dense_batch_embeddings.shape[0] > i:
                        dense_embedding = dense_batch_embeddings[i]
                    else:
                        # Individual fallback
                        dense_embedding = self.get_embedding(chunk["text"])
                    
                    # Get sparse embedding (from batch or individual)
                    if sparse_batch_embeddings and i < len(sparse_batch_embeddings):
                        sparse_embedding = sparse_batch_embeddings[i]
                    elif hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
                        try:
                            # Try individual embedding through provider
                            sparse_embedding = self.mlx_embedding_provider.get_sparse_embedding(chunk["text"])
                        except Exception as e:
                            # Use document processor's fallback method if provider has issues
                            sparse_embedding = self._generate_sparse_vector(chunk["text"])
                    else:
                        # Use document processor's fallback method
                        sparse_embedding = self._generate_sparse_vector(chunk["text"])
                    
                    # Create payload with embedder information
                    payload = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "metadata": {
                            "file_ext": os.path.splitext(file_path)[1],
                            "file_size": os.path.getsize(file_path),
                            "created_at": time.time(),
                            "dense_embedder": self.dense_model_id,
                            "sparse_embedder": self.sparse_model_id
                        }
                    }
                    
                    results.append((dense_embedding, payload, sparse_embedding))
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if self.verbose:
                        print(f"Error processing chunk {chunk_idx+1}: {e}")
        
        if self.verbose:
            print(f"File processing complete: {success_count} chunks successful, {error_count} chunks failed")
            
        return results


class QdrantManager:
    """Manager for Qdrant vector database operations with multi-model support"""
    
    def __init__(self, 
                host: str = "localhost", 
                port: int = 6333, 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = DEFAULT_DENSE_MODEL,
                sparse_model_id: str = DEFAULT_SPARSE_MODEL):
        """
        Initialize QdrantManager with model-specific vector configuration.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Storage path for local mode
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path
        self.verbose = verbose
        self.client = None
        self.is_remote = False
        
        # Sanitize model IDs for vector names - replace hyphens and slashes with underscores
        self.dense_model_id = dense_model_id.replace("-", "_").replace("/", "_")
        self.sparse_model_id = sparse_model_id.replace("-", "_").replace("/", "_")
        
        self.connect()

    def is_local(self):
        """Check if we're running in local mode"""
        return not self.is_remote and self.storage_path is not None
        
    def connect(self):
        """Connect to Qdrant using either local storage or remote server"""
        try:
            # Determine if we're using a remote server or local storage
            self.is_remote = self.host != "localhost" or self.port != 6333
            
            if self.is_remote:
                # Connect to remote Qdrant server
                if self.verbose:
                    print(f"Connecting to Qdrant server at {self.host}:{self.port}")
                    
                self.client = QdrantClient(host=self.host, port=self.port)
                
                # Test the connection
                try:
                    collections = self.client.get_collections()
                    if self.verbose:
                        print(f"Connected to Qdrant server. Collections: {[c.name for c in collections.collections]}")
                except Exception as e:
                    print(f"Error connecting to Qdrant server: {str(e)}")
                    
                    # Only fall back to local mode if storage_path is provided
                    if self.storage_path:
                        print(f"Falling back to local storage mode")
                        self.is_remote = False
                    else:
                        raise  # Re-raise the exception if we can't fall back
            
            # If not using remote (either by choice or after fallback)
            if not self.is_remote:
                # Check if storage path is provided
                if self.storage_path:
                    # Use provided storage path
                    storage_path = self.storage_path
                else:
                    # Create a persistent directory in the current working directory
                    storage_path = os.path.join(os.getcwd(), "qdrant_storage")
                
                # Ensure directory exists
                os.makedirs(storage_path, exist_ok=True)
                
                if self.verbose:
                    print(f"Using local storage at: {storage_path}")
                    
                # Initialize client with path for local mode
                self.client = QdrantClient(path=storage_path)
                
                # Store the storage path being used
                self.storage_path = storage_path
                
                # Test the client by getting collections
                collections = self.client.get_collections()
                
                if self.verbose:
                    print(f"Connected to local Qdrant storage. Collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            print(f"Error initializing Qdrant: {str(e)}")
            print("You may need to install Qdrant client: pip install qdrant-client")
            raise
        
    def create_collection(self, recreate: bool = False) -> None:
        """Create Qdrant collection with support for both dense and sparse vectors"""
        try:
            # Import Qdrant models
            from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
            
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            # Create vector name based on model ID
            dense_vector_name = f"dense_{self.dense_model_id.replace('-', '_').replace('/', '_')}"
            sparse_vector_name = f"sparse_{self.sparse_model_id.replace('-', '_').replace('/', '_')}"
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")

            if collection_exists:
                if recreate:
                    if self.verbose:
                        print(f"Recreating collection '{self.collection_name}'...")
                    self.client.delete_collection(self.collection_name)
                else:
                    if self.verbose:
                        print(f"Collection '{self.collection_name}' already exists.")
                    return

            if self.verbose:
                print(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")

            # Create collection with dense vectors
            vectors_config = {
                dense_vector_name: VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            }
            
            # Create sparse vectors config
            sparse_vectors_config = {
                sparse_vector_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
            
            # Create collection with both dense and sparse vectors
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                
                if self.verbose:
                    print(f"Successfully created collection with dense and sparse vectors")
            except Exception as e:
                # Fallback for compatibility with older clients
                if self.verbose:
                    print(f"Error with integrated creation: {e}")
                    print("Falling back to two-step creation")
                    
                # Create with just dense vectors first
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                
                # Then add sparse vectors separately
                try:
                    self.client.create_sparse_vector(
                        collection_name=self.collection_name,
                        vector_name=sparse_vector_name,
                        on_disk=False
                    )
                    if self.verbose:
                        print(f"Added sparse vector configuration: {sparse_vector_name}")
                except Exception as e2:
                    # Handle older clients that might not have create_sparse_vector
                    if "AttributeError" in str(e2):
                        try:
                            # Try alternative approach for older clients
                            self.client.update_collection(
                                collection_name=self.collection_name,
                                sparse_vectors_config=sparse_vectors_config
                            )
                            if self.verbose:
                                print(f"Added sparse vector configuration using update_collection")
                        except Exception as e3:
                            print(f"Warning: Could not add sparse vectors: {e3}")
                            print("Sparse search may not work properly.")
                    else:
                        print(f"Warning: Could not add sparse vectors: {e2}")
                        print("Sparse search may not work properly.")

            # Create payload indexes for better filtering performance
            try:
                # Create text index for full-text search
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="text",
                    field_schema="text"  # Simplified schema that works across versions
                )
                
                # Create keyword indexes for exact matching
                for field in ["file_name", "file_path"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema="keyword"
                    )
                    
                if self.verbose:
                    print(f"✅ Collection '{self.collection_name}' created successfully with indexes")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not create payload indexes: {e}")
                    print(f"✅ Collection '{self.collection_name}' created successfully (without indexes)")

        except Exception as e:
            print(f"❌ Error creating collection '{self.collection_name}': {e}")
            raise


    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client:
                return {"error": "Not connected to Qdrant"}
                
            collection_info = self.client.get_collection(self.collection_name)
            points_count = self.client.count(self.collection_name).count
            
            # Get disk usage if possible
            disk_usage = None
            try:
                if self.storage_path:
                    collection_path = Path(f"{self.storage_path}/collections/{self.collection_name}")
                    if collection_path.exists():
                        disk_usage = sum(f.stat().st_size for f in collection_path.glob('**/*') if f.is_file())
            except Exception as e:
                if self.verbose:
                    print(f"Could not calculate disk usage: {str(e)}")
            
            # Safely get vector configurations
            vector_configs = {}
            try:
                if hasattr(collection_info.config.params, 'vectors'):
                    # Safely get vectors dictionary using model_dump() instead of dict()
                    if hasattr(collection_info.config.params.vectors, 'model_dump'):
                        vectors_dict = collection_info.config.params.vectors.model_dump()
                    elif hasattr(collection_info.config.params.vectors, 'dict'):
                        vectors_dict = collection_info.config.params.vectors.dict()
                    else:
                        vectors_dict = {}
                        
                    for name, config in vectors_dict.items():
                        # Safely access configuration
                        if isinstance(config, dict):
                            vector_configs[name] = {
                                "size": config.get("size"),
                                "distance": config.get("distance")
                            }
                        else:
                            vector_configs[name] = {"info": str(config)}
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting vector configs: {e}")
            
            # Safely get sparse vector configurations
            sparse_vector_configs = {}
            try:
                if hasattr(collection_info.config.params, 'sparse_vectors'):
                    if hasattr(collection_info.config.params.sparse_vectors, 'model_dump'):
                        sparse_vectors_dict = collection_info.config.params.sparse_vectors.model_dump()
                    elif hasattr(collection_info.config.params.sparse_vectors, 'dict'):
                        sparse_vectors_dict = collection_info.config.params.sparse_vectors.dict()
                    else:
                        sparse_vectors_dict = {}
                    
                    sparse_vector_configs = {
                        name: {"type": "sparse"}
                        for name in sparse_vectors_dict.keys()
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting sparse vector configs: {e}")
            
            return {
                "name": self.collection_name,
                "points_count": points_count,
                "disk_usage": disk_usage,
                "vector_configs": vector_configs,
                "sparse_vector_configs": sparse_vector_configs
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting collection info: {str(e)}")
                import traceback
                traceback.print_exc()
            return {"error": str(e)}
    
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into Qdrant (dense only, with generated sparse)"""
        if not embeddings_with_payloads:
            return
            
        try:
            # Determine vector name based on model ID (fallback to default if not in payload)
            first_payload = embeddings_with_payloads[0][1]
            dense_model_id = first_payload.get("metadata", {}).get("embedder", self.dense_model_id.replace('-', '_').replace('/', '_'))
            dense_vector_name = f"dense_{dense_model_id}"
            
            # Generate sparse vector name based on a default or fallback
            sparse_vector_name = f"sparse_{self.sparse_model_id}"
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")
            
            # Prepare points for insertion
            points = []
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Generate a UUID for the point
                import uuid
                point_id = str(uuid.uuid4())
                
                # Generate sparse vector for the text
                text = payload.get("text", "")
                sparse_indices, sparse_values = self._generate_sparse_vector(text)
                
                # Create point with vectors in the same dictionary
                # NOTE: This is the corrected format based on Qdrant documentation
                vector_dict = {
                    dense_vector_name: embedding.tolist(),
                    sparse_vector_name: {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=vector_dict,  # Include both dense and sparse in vector dictionary
                    payload=payload
                )
                points.append(point)
            
            if self.verbose:
                print(f"Inserting {len(points)} points into collection '{self.collection_name}'")
                
            # Insert points in batches
            BATCH_SIZE = 100
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i:i+BATCH_SIZE]
                if self.verbose and len(points) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            if self.verbose:
                print(f"Successfully inserted {len(points)} points")
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Qdrant following documentation format.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            # Import UUID for generating point IDs
            import uuid
            import time
            
            # Get model IDs from the first payload, but sanitize them for consistency
            first_payload = embeddings_with_sparse[0][1]
            dense_model_id = first_payload.get("metadata", {}).get("dense_embedder", self.dense_model_id)
            sparse_model_id = first_payload.get("metadata", {}).get("sparse_embedder", self.sparse_model_id)
            
            # Sanitize the model IDs to match the collection creation pattern
            dense_model_id = dense_model_id.replace("-", "_").replace("/", "_")
            sparse_model_id = sparse_model_id.replace("-", "_").replace("/", "_")
            
            # Create vector names based on sanitized model IDs
            dense_vector_name = f"dense_{dense_model_id}"
            sparse_vector_name = f"sparse_{sparse_model_id}"
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")
                
            # Process and insert one by one to avoid batch issues
            total_success = 0
            total_failed = 0
            
            for i, (dense_embedding, payload, sparse_embedding) in enumerate(embeddings_with_sparse):
                # Store file info in metadata
                if "metadata" not in payload:
                    payload["metadata"] = {}
                    
                file_name = payload.get("file_name", "")
                chunk_index = payload.get("chunk_index", i)
                payload["metadata"]["original_file"] = file_name
                payload["metadata"]["chunk_index"] = chunk_index
                
                # Get sparse indices and values
                sparse_indices, sparse_values = sparse_embedding
                
                # Validate sparse vectors - warn if they appear to be just a bag of words
                if len(sparse_indices) < 5 or max(sparse_values) < 0.1:
                    if self.verbose:
                        print(f"⚠️ WARNING: Sparse vector for point {i} appears to be low quality.")
                        print(f"Indices: {sparse_indices[:5]}...")
                        print(f"Values: {sparse_values[:5]}...")
                        print(f"This might be just a bag of words rather than a proper SPLADE vector.")
                
                # Try up to 3 times with different IDs if needed
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        # Generate a new UUID for each attempt
                        point_id = str(uuid.uuid4())
                        
                        # Format vectors according to Qdrant documentation
                        vector_dict = {
                            dense_vector_name: dense_embedding.tolist(),
                            sparse_vector_name: {
                                "indices": sparse_indices,
                                "values": sparse_values
                            }
                        }
                        
                        # Create point with the combined vector dictionary
                        point = PointStruct(
                            id=point_id,
                            vector=vector_dict,
                            payload=payload
                        )
                        
                        # Before insertion - check what we're about to send
                        if self.verbose and i == 0:
                            print(f"\n=== Example point structure for first insertion ===")
                            print(f"Point ID: {point_id}")
                            print(f"Vector dict keys: {vector_dict.keys()}")
                            print(f"Sparse vector has {len(sparse_indices)} indices and {len(sparse_values)} values")
                            if len(sparse_indices) > 0:
                                print(f"Sample sparse indices: {sparse_indices[:5]}")
                                print(f"Sample sparse values: {sparse_values[:5]}")

                        # Insert single point
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=[point]
                        )
                        
                        # Success
                        total_success += 1
                        break
                        
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            # Last attempt failed
                            if self.verbose:
                                print(f"Failed to insert point {i} after {max_attempts} attempts: {e}")
                            total_failed += 1
                        elif "Indices must be unique" in str(e):
                            # Try again with a different ID
                            if self.verbose:
                                print(f"ID collision detected on attempt {attempt+1}, retrying...")
                        else:
                            # Other error
                            if self.verbose:
                                print(f"Error inserting point {i}: {e}")
                            total_failed += 1
                            break
                
                # Print progress periodically
                if self.verbose and (i+1) % 5 == 0:
                    print(f"Progress: {i+1}/{len(embeddings_with_sparse)} points processed")
            
            if self.verbose:
                print(f"Indexing complete: {total_success} points inserted, {total_failed} points failed")
                    
        except Exception as e:
            print(f"Error during embeddings insertion: {str(e)}")
            import traceback
            traceback.print_exc()
    

    def _create_preview(self, text, query, context_size=300):
        """
        Create a preview with context around search terms.
        
        Args:
            text: The full text to extract preview from
            query: Search query to look for
            context_size: Maximum context size in characters
        
        Returns:
            A preview string with context around matched terms
        """
        if not text:
            return ""
        
        # If text is shorter than context_size, return the whole text
        if len(text) <= context_size:
            return text
        
        # Find positions of search terms in text
        text_lower = text.lower()
        query_lower = query.lower()
        query_terms = query_lower.split()
        positions = []
        
        # Try complete query first
        pos = text_lower.find(query_lower)
        if pos != -1:
            positions.append(pos)
        
        # Then try individual terms
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                term_pos = text_lower.find(term)
                while term_pos != -1:
                    positions.append(term_pos)
                    term_pos = text_lower.find(term, term_pos + 1)
        
        # If no matches found, return the beginning of the text
        if not positions:
            return text[:context_size] + "..."
        
        # Find the best window that contains the most matches
        best_start = 0
        best_end = min(context_size, len(text))
        
        if positions:
            # Sort positions
            positions.sort()
            # Choose the middle position to center around
            middle_pos = positions[len(positions) // 2]
            
            # Center context window around the middle position
            best_start = max(0, middle_pos - context_size // 2)
            best_end = min(len(text), best_start + context_size)
        
        # Adjust window to not cut words
        if best_start > 0:
            while best_start > 0 and text[best_start] != ' ':
                best_start -= 1
            best_start += 1  # Move past the space
        
        if best_end < len(text):
            while best_end < len(text) and text[best_end] != ' ':
                best_end += 1
        
        # Create preview with ellipses if needed
        preview = ""
        if best_start > 0:
            preview += "..."
        
        preview += text[best_start:best_end]
        
        if best_end < len(text):
            preview += "..."
        
        # Highlight search terms with **
        final_preview = preview
        for term in sorted(query_terms, key=len, reverse=True):
            if len(term) > 2:  # Skip very short terms
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                final_preview = pattern.sub(f"**{term}**", final_preview)
        
        return final_preview
    
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None):
        """
        Perform a keyword-based search optimized for local Qdrant installations.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
                
            if self.verbose:
                print(f"Performing keyword search for: '{query}'")
            
            # Create a payload filter for text matches
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchText
                
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                )
                
                # Detect if we're running in local mode
                is_local_mode = not self.is_remote and self.storage_path is not None
                
                # Try different search methods in order of preference
                matched_points = []
                
                # For local mode, try the direct scroll method first as it's most reliable
                if is_local_mode:
                    if self.verbose:
                        print("Running in local mode, starting with direct scroll method...")
                    
                    try:
                        # Direct scroll with manual filtering
                        all_results = self.client.scroll(
                            collection_name=self.collection_name,
                            limit=1000,  # Get a larger batch
                            with_payload=True
                        )
                        
                        all_points = all_results[0]
                        
                        if self.verbose:
                            print(f"Retrieved {len(all_points)} points for manual filtering")
                            
                        # Manually filter by keyword
                        query_lower = query.lower()
                        matched_points = []
                        
                        for point in all_points:
                            if hasattr(point, "payload") and point.payload and "text" in point.payload:
                                text = point.payload["text"].lower()
                                if query_lower in text:
                                    matched_points.append(point)
                        
                        if self.verbose:
                            print(f"Manual keyword filtering found {len(matched_points)} matches")
                            
                        # If we found matches, skip other methods
                        if matched_points:
                            if self.verbose:
                                print("Using results from direct scroll method")
                        else:
                            if self.verbose:
                                print("No matches found via direct scroll, trying other methods...")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"Direct scroll method failed: {e}")
                            print("Trying other search methods...")
                
                # If we're not in local mode or direct scroll didn't work, try the other methods
                if not matched_points:
                    # Method 1: Use scroll API with filter
                    if self.verbose:
                        print("Attempting scroll-based keyword search...")
                        
                    try:
                        # First try with filter parameter
                        try:
                            results = self.client.scroll(
                                collection_name=self.collection_name,
                                filter=query_filter,
                                limit=limit * 3,
                                with_payload=True
                            )
                            points = results[0]
                            if self.verbose:
                                print(f"Keyword search using scroll with filter= returned {len(points)} results")
                            matched_points = points
                        except Exception as scroll_error:
                            # Check if it's the specific error about unknown arguments
                            if "Unknown arguments" in str(scroll_error) and "filter" in str(scroll_error):
                                if self.verbose:
                                    print(f"Scroll with filter= parameter failed: {scroll_error}")
                                    print("Trying with scroll_filter= parameter...")
                                
                                # Try with scroll_filter parameter (newer Qdrant versions)
                                results = self.client.scroll(
                                    collection_name=self.collection_name,
                                    scroll_filter=query_filter,
                                    limit=limit * 3,
                                    with_payload=True
                                )
                                points = results[0]
                                if self.verbose:
                                    print(f"Keyword search using scroll with scroll_filter= returned {len(points)} results")
                                matched_points = points
                            else:
                                # Re-raise if it's a different error
                                raise
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Scroll-based keyword search failed: {e}")
                            print("Trying query_points method...")
                        
                        # Method 2: Try query_points
                        try:
                            try:
                                # First try with query_filter parameter
                                results = self.client.query_points(
                                    collection_name=self.collection_name,
                                    query_filter=query_filter,
                                    limit=limit * 3,
                                    with_payload=True
                                )
                                
                                points = results.points
                                
                                if self.verbose:
                                    print(f"Keyword search using query_points with query_filter= returned {len(points)} results")
                                    
                                matched_points = points
                            except Exception as query_error:
                                # Try with filter parameter if query_filter fails
                                if "Unknown arguments" in str(query_error) and "query_filter" in str(query_error):
                                    if self.verbose:
                                        print(f"query_points with query_filter= parameter failed: {query_error}")
                                        print("Trying with filter= parameter...")
                                    
                                    results = self.client.query_points(
                                        collection_name=self.collection_name,
                                        filter=query_filter,
                                        limit=limit * 3,
                                        with_payload=True
                                    )
                                    
                                    points = results.points
                                    
                                    if self.verbose:
                                        print(f"Keyword search using query_points with filter= returned {len(points)} results")
                                        
                                    matched_points = points
                                else:
                                    # Re-raise if it's a different error
                                    raise
                                
                        except Exception as e2:
                            if self.verbose:
                                print(f"Query-points keyword search failed: {e2}")
                                print("Trying search with filter method...")
                            
                            # Method 3: Last resort with search + filter
                            try:
                                # Get vector size for dummy vector
                                vector_size = self.vector_dim
                                
                                # Create a dummy vector (all zeros)
                                dummy_vector = [0.0] * vector_size
                                
                                # Get vector name - use first available
                                collection_info = self.client.get_collection(self.collection_name)
                                vector_names = []
                                
                                if hasattr(collection_info.config.params, 'vectors'):
                                    if hasattr(collection_info.config.params.vectors, 'keys'):
                                        vector_names = list(collection_info.config.params.vectors.keys())
                                    elif hasattr(collection_info.config.params.vectors, '__dict__'):
                                        vector_names = list(collection_info.config.params.vectors.__dict__.keys())
                                
                                if not vector_names:
                                    vector_names = [f"dense_{self.dense_model_id.replace('-','_').replace('/','_')}"]
                                
                                vector_name = vector_names[0]
                                
                                try:
                                    # Try with query_filter parameter
                                    results = self.client.search(
                                        collection_name=self.collection_name,
                                        query_vector=(vector_name, dummy_vector),
                                        query_filter=query_filter,
                                        limit=limit * 3,
                                        with_payload=True
                                    )
                                    
                                    points = results
                                    
                                    if self.verbose:
                                        print(f"Keyword search using search+query_filter returned {len(points)} results")
                                        
                                    matched_points = points
                                except Exception as search_error:
                                    # Try with filter parameter if query_filter fails
                                    if "Unknown arguments" in str(search_error) and "query_filter" in str(search_error):
                                        if self.verbose:
                                            print(f"search with query_filter= parameter failed: {search_error}")
                                            print("Trying with filter= parameter...")
                                        
                                        results = self.client.search(
                                            collection_name=self.collection_name,
                                            query_vector=(vector_name, dummy_vector),
                                            filter=query_filter,
                                            limit=limit * 3,
                                            with_payload=True
                                        )
                                        
                                        points = results
                                        
                                        if self.verbose:
                                            print(f"Keyword search using search+filter returned {len(points)} results")
                                            
                                        matched_points = points
                                    else:
                                        # Re-raise if it's a different error
                                        raise
                            
                            except Exception as e3:
                                if self.verbose:
                                    print(f"All keyword search methods failed: {e3}")
                                return {"error": f"All keyword search methods failed: {e3}"}
                
                # If we have no points to process, return empty results
                if not matched_points:
                    if self.verbose:
                        print("No matches found for keyword search")
                    return []
                    
                # Define a simpler Point class for our processed results
                class ScoredPoint:
                    def __init__(self, id, payload, score):
                        self.id = id
                        self.payload = payload
                        self.score = score
                        self.version = 0  # Default version
                
                # Apply post-processing to improve scores and validate matches
                processed_points = []
                query_lower = query.lower()
                query_terms = query_lower.split()
                
                for point in matched_points:
                    # Extract the necessary attributes from the original point
                    point_id = getattr(point, "id", None)
                    payload = getattr(point, "payload", {})
                    
                    # Skip invalid points
                    if point_id is None or not payload:
                        continue
                    
                    # Get text content and validate the match
                    text = payload.get("text", "").lower()
                    if not text or query_lower not in text:
                        # Skip points that don't actually contain the query
                        if self.verbose:
                            print(f"Skipping point {point_id}: Does not actually contain '{query}'")
                        continue
                    
                    # Calculate a better relevance score
                    score = 0.0
                    
                    # Calculate score based on exact match, frequency, and position
                    if query_lower in text:
                        # Base score for exact match
                        score = 0.7
                        
                        # Boost for match at beginning
                        position = text.find(query_lower) / max(1, len(text))
                        position_boost = max(0, 0.3 - position * 0.5)  # Higher boost for earlier positions
                        score += position_boost
                        
                        # Frequency bonus (multiple occurrences)
                        freq = text.count(query_lower)
                        freq_boost = min(0.2, 0.05 * (freq - 1))  # Up to 0.2 for frequency
                        score += freq_boost
                    else:
                        # This shouldn't happen due to our earlier check, but just in case
                        continue
                    
                    # Create a new ScoredPoint object
                    processed_point = ScoredPoint(
                        id=point_id,
                        payload=payload,
                        score=score
                    )
                    
                    processed_points.append(processed_point)
                
                # Sort by score (descending)
                processed_points.sort(key=lambda p: p.score, reverse=True)
                
                # Apply score threshold if provided
                if score_threshold is not None:
                    processed_points = [p for p in processed_points if p.score >= score_threshold]
                
                # Limit to requested number
                points = processed_points[:limit]
                
                if self.verbose:
                    print(f"Final result: {len(points)} validated keyword matches")
                    
                return points
                
            except Exception as e:
                return {"error": f"Error creating keyword filter: {e}"}
        
        except Exception as e:
            print(f"Error in keyword search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in keyword search: {str(e)}"}
    
    def search_hybrid(self, query: str, processor: Any, limit: int, 
                  prefetch_limit: int = 50, fusion_type: str = "rrf",
                  score_threshold: float = None, rerank: bool = False):
        """
        Perform a hybrid search combining both dense and sparse vectors with optional reranking
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking as a third step
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Get vector names with consistent sanitization
            has_mlx_provider = (
                processor is not None
                and hasattr(processor, 'mlx_embedding_provider')
                and processor.mlx_embedding_provider is not None
            )
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            sparse_model_id = getattr(processor, 'sparse_model_id', self.sparse_model_id)
            
            # Sanitize vector names
            dense_model_id = dense_model_id.replace("-", "_").replace("/", "_")
            sparse_model_id = sparse_model_id.replace("-", "_").replace("/", "_")
            
            dense_vector_name = f"dense_{dense_model_id}"
            sparse_vector_name = f"sparse_{sparse_model_id}"
            
            if self.verbose:
                print(f"Using vector names for hybrid search: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")
            
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Generate sparse embedding
            if has_mlx_provider:
                sparse_indices, sparse_values = processor.mlx_embedding_provider.get_sparse_embedding(query)
            else:
                sparse_indices, sparse_values = processor._generate_sparse_vector(query)
                
            # Warn if sparse vector appears to be just a bag of words
            if len(sparse_indices) < 5 or max(sparse_values) < 0.1:
                print(f"⚠️ WARNING: Sparse vector appears to be low quality (possibly just a bag of words).")
                print(f"This might reduce hybrid search quality.")
            
            # Try using modern fusion API first
            try:
                from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
                
                # Create prefetch list
                prefetch_list = [
                    # Dense vector prefetch
                    Prefetch(
                        query=query_vector.tolist(),
                        using=dense_vector_name,
                        limit=prefetch_limit,
                    ),
                    # Sparse vector prefetch 
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using=sparse_vector_name,
                        limit=prefetch_limit,
                    )
                ]
                
                # Choose fusion method
                fusion_enum = Fusion.DBSF if fusion_type.lower() == "dbsf" else Fusion.RRF
                
                # Hybrid query
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch_list,
                    query=FusionQuery(fusion=fusion_enum),
                    limit=limit * 3 if rerank else limit,  # Get more if we're going to rerank
                    with_payload=True
                )
                
                points = response.points
                
                if self.verbose:
                    print(f"Hybrid search with fusion returned {len(points)} results")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Modern fusion-based hybrid search failed: {e}")
                    print("Using manual hybrid search approach")
                
                # Perform separate searches and combine manually
                dense_results = self.search_dense(query, processor, prefetch_limit, score_threshold)
                if "error" in dense_results:
                    dense_results = []
                
                sparse_results = self.search_sparse(query, processor, prefetch_limit, score_threshold)
                if "error" in sparse_results:
                    sparse_results = []
                    
                # Combine and rerank results using reciprocal rank fusion
                points = self._manual_fusion(dense_results, sparse_results, 
                                            limit * 3 if rerank else limit, 
                                            fusion_type)
            
            # Apply reranking if requested
            if rerank and len(points) > 0:
                if self.verbose:
                    print(f"Applying reranking to {len(points)} results")
                
                points = self._rerank_results(query, points, processor, limit)
                
            # Apply score threshold if not already applied
            if score_threshold is not None:
                points = [p for p in points if self._get_score(p) >= score_threshold]
                
            # Limit to requested number
            points = points[:limit]
            
            return points
        
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in hybrid search: {str(e)}"}


    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """
        Perform a dense vector search with consistent handling
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Get dense vector name with consistent sanitization
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            dense_model_id = dense_model_id.replace("-", "_").replace("/", "_")
            dense_vector_name = f"dense_{dense_model_id}"
            
            if self.verbose:
                print(f"Using vector name for dense search: {dense_vector_name}")
            
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Attempt to use different search methods in order of preference
            try:
                # Modern approach with specific vector name
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(dense_vector_name, query_vector.tolist()),
                    limit=limit,
                    with_payload=True,
                    score_threshold=score_threshold
                )
                
                if self.verbose:
                    print(f"Dense search returned {len(search_result)} results")
                
                return search_result
                
            except Exception as e:
                if self.verbose:
                    print(f"Standard dense search failed: {e}")
                    print("Trying alternative dense search methods...")
                
                # Try with query_points API
                try:
                    result = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector.tolist(),
                        using=dense_vector_name,
                        limit=limit,
                        with_payload=True,
                        score_threshold=score_threshold
                    )
                    
                    if self.verbose:
                        print(f"Dense search with query_points returned {len(result.points)} results")
                    
                    return result.points
                    
                except Exception as e2:
                    # Last resort - try without named vectors
                    try:
                        search_result = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector.tolist(),
                            limit=limit,
                            with_payload=True,
                            score_threshold=score_threshold
                        )
                        
                        if self.verbose:
                            print(f"Dense search with legacy method returned {len(search_result)} results")
                        
                        return search_result
                        
                    except Exception as e3:
                        return {"error": f"All dense search methods failed: {e3}"}
        
        except Exception as e:
            print(f"Error in dense search: {str(e)}")
            return {"error": f"Error in dense search: {str(e)}"}


    def _manual_fusion(self, dense_results, sparse_results, limit: int, fusion_type: str = "rrf"):
        """
        Perform manual fusion of dense and sparse search results
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            limit: Maximum number of results to return
            fusion_type: Type of fusion to use (rrf or dbsf)
            
        Returns:
            Combined and reranked list of results
        """
        # Safety check for inputs
        if not dense_results and not sparse_results:
            return []
        
        # Helper function to get score safely
        def get_score(item):
            return getattr(item, "score", 0.0)
        
        # Dictionary to store combined results with ID as key
        combined_dict = {}
        
        # Process dense results
        for rank, item in enumerate(dense_results):
            item_id = getattr(item, "id", str(rank))
            combined_dict[item_id] = {
                "item": item,
                "dense_rank": rank + 1,
                "dense_score": get_score(item),
                "sparse_rank": float('inf'),
                "sparse_score": 0.0
            }
        
        # Process sparse results
        for rank, item in enumerate(sparse_results):
            item_id = getattr(item, "id", str(rank))
            if item_id in combined_dict:
                # Update existing entry
                combined_dict[item_id]["sparse_rank"] = rank + 1
                combined_dict[item_id]["sparse_score"] = get_score(item)
            else:
                # Add new entry
                combined_dict[item_id] = {
                    "item": item,
                    "dense_rank": float('inf'),
                    "dense_score": 0.0,
                    "sparse_rank": rank + 1,
                    "sparse_score": get_score(item)
                }
        
        # Apply fusion based on chosen method
        if fusion_type.lower() == "dbsf":
            # Distribution-based Score Fusion
            # Normalize scores within each result set
            dense_max = max([d["dense_score"] for d in combined_dict.values()]) if dense_results else 1.0
            sparse_max = max([d["sparse_score"] for d in combined_dict.values()]) if sparse_results else 1.0
            
            # Calculate combined scores
            for item_id, data in combined_dict.items():
                norm_dense = data["dense_score"] / dense_max if dense_max > 0 else 0
                norm_sparse = data["sparse_score"] / sparse_max if sparse_max > 0 else 0
                
                # DBSF: Weighted sum of normalized scores
                data["combined_score"] = 0.5 * norm_dense + 0.5 * norm_sparse
        else:
            # Reciprocal Rank Fusion (default)
            k = 60  # Constant for RRF
            
            # Calculate combined scores using RRF formula
            for item_id, data in combined_dict.items():
                rrf_dense = 1.0 / (k + data["dense_rank"]) if data["dense_rank"] != float('inf') else 0
                rrf_sparse = 1.0 / (k + data["sparse_rank"]) if data["sparse_rank"] != float('inf') else 0
                
                # RRF: Sum of reciprocal ranks
                data["combined_score"] = rrf_dense + rrf_sparse
        
        # Sort by combined score and convert back to a list
        sorted_results = sorted(
            combined_dict.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        # Return only the original items, limited to requested count
        return [data["item"] for data in sorted_results[:limit]]


    def _rerank_results(self, query: str, results, processor: Any, limit: int):
        """
        Rerank results using a cross-encoder style approach with MLX
        
        Args:
            query: Original search query
            results: List of search results to rerank
            processor: Document processor with MLX capabilities
            limit: Maximum number of results to return
            
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        if self.verbose:
            print(f"Reranking {len(results)} results with MLX")
        
        # Extract text from results
        passages = []
        for result in results:
            if hasattr(result, "payload") and result.payload and "text" in result.payload:
                passages.append(result.payload["text"])
            else:
                # Use empty string if no text available
                passages.append("")
        
        # Reranking depends on which features are available
        has_mlx_provider = (
            processor is not None
            and hasattr(processor, 'mlx_embedding_provider')
            and processor.mlx_embedding_provider is not None
        )
        
        reranked_scores = []
        
        if has_mlx_provider and hasattr(processor.mlx_embedding_provider, 'rerank_with_mlx'):
            # Use dedicated reranking method if available
            reranked_scores = processor.mlx_embedding_provider.rerank_with_mlx(query, passages)
        else:
            # Fallback: Use ColBERT-style late interaction scoring
            try:
                # Get query embedding
                query_embedding = processor.get_embedding(query)
                
                # Get passage embeddings
                passage_embeddings = []
                for passage in passages:
                    embedding = processor.get_embedding(passage)
                    passage_embeddings.append(embedding)
                
                # Compute similarity scores
                import numpy as np
                reranked_scores = []
                
                for passage_emb in passage_embeddings:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, passage_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(passage_emb)
                    )
                    reranked_scores.append(float(similarity))
            except Exception as e:
                if self.verbose:
                    print(f"Error in fallback reranking: {e}")
                # If reranking fails, keep original order
                return results[:limit]
        
        # Create tuples of (result, score) for sorting
        scored_results = list(zip(results, reranked_scores))
        
        # Sort by reranked score
        reranked_results = [result for result, _ in sorted(
            scored_results, 
            key=lambda x: x[1], 
            reverse=True
        )]
        
        return reranked_results[:limit]


    def _get_score(self, result):
        """Safely get score from a result"""
        if hasattr(result, "score"):
            return getattr(result, "score", 0)
        return 0


    def search(self, query: str, search_type: str = "hybrid", limit: int = 10,
            processor: Any = None, prefetch_limit: int = 50, fusion_type: str = "rrf",
            relevance_tuning: bool = True, context_size: int = 300, 
            score_threshold: float = None, rerank: bool = False):
        """
        Enhanced search using dense, sparse, or hybrid search with improved relevance
        and optional reranking stage.
        
        Args:
            query: Search query string
            search_type: Type of search to perform (hybrid, vector, sparse, keyword)
            limit: Maximum number of results to return
            processor: Document processor with embedding capabilities
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            relevance_tuning: Whether to apply relevance tuning
            context_size: Size of context window for preview
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking as a third step
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            if self.verbose:
                print(f"Searching for '{query}' using {search_type} search" + 
                    (" with reranking" if rerank else ""))
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                return {"error": f"Collection {self.collection_name} does not exist"}
            
            # Determine correct search method based on type
            if search_type == "vector" or search_type == "dense":
                points = self.search_dense(query, processor, limit * 3 if rerank else limit, score_threshold)
            elif search_type == "sparse":
                points = self.search_sparse(query, processor, limit * 3 if rerank else limit, score_threshold)
            elif search_type == "keyword":
                points = self.search_keyword(query, limit * 3 if rerank else limit)
            else:  # Default to hybrid
                points = self.search_hybrid(query, processor, limit, prefetch_limit, 
                                            fusion_type, score_threshold, rerank=False)
            
            # Apply reranking as a third step if requested and not already applied
            if rerank and search_type != "hybrid" and not isinstance(points, dict):
                if self.verbose:
                    print(f"Applying reranking to {len(points)} results")
                
                points = self._rerank_results(query, points, processor, limit)
            
            # Check for errors
            if isinstance(points, dict) and "error" in points:
                return points
                
            # Format results with improved preview
            return self._format_search_results(points, query, search_type, processor, context_size)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


    def _format_search_results(self, points, query, search_type, processor, context_size=300):
        """Format search results with improved preview"""
        # Extract model IDs for result metadata
        dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id) if processor else self.dense_model_id
        sparse_model_id = getattr(processor, 'sparse_model_id', self.sparse_model_id) if processor else self.sparse_model_id
        
        if not points or len(points) == 0:
            return {
                "query": query,
                "search_type": search_type,
                "count": 0,
                "embedder_info": {
                    "dense": dense_model_id,
                    "sparse": sparse_model_id
                },
                "results": []
            }
            
        # Filter out tiny documents (< 20 chars) as they won't have meaningful content
        filtered_points = []
        for point in points:
            if hasattr(point, "payload") and point.payload and "text" in point.payload:
                if len(point.payload["text"]) >= 20:  # Only include meaningful chunks
                    filtered_points.append(point)
                elif self.verbose:
                    print(f"Skipping tiny chunk (id: {point.id}): '{point.payload['text']}'")
        
        # Use original points if all were filtered out
        if not filtered_points and points:
            if self.verbose:
                print("Warning: All results were filtered out due to small size. Using original results.")
            filtered_points = points
            
        # Update points to filtered list
        points = filtered_points
        
        # Format results
        formatted_results = []
        for i, result in enumerate(points):
            payload = result.payload if hasattr(result, "payload") else {}
            score = self._get_score(result)
            text = payload.get("text", "")
            
            # Create preview with context handling
            preview = self._create_smart_preview(text, query, context_size)
            
            # Get chunk size information
            chunk_size = {
                "characters": len(text),
                "words": len(text.split()),
                "lines": len(text.splitlines())
            }
            
            # Get embedding information
            embedder_meta = payload.get("metadata", {})
            embedder_info = {
                "dense_embedder": embedder_meta.get("dense_embedder", embedder_meta.get("embedder", "unknown")),
                "sparse_embedder": embedder_meta.get("sparse_embedder", "unknown"),
            }
            
            formatted_results.append({
                "rank": i + 1,
                "score": score,
                "id": result.id,
                "file_path": payload.get("file_path", ""),
                "file_name": payload.get("file_name", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "chunk_size": chunk_size,
                "preview": preview,
                "text": text,
                "embedder_info": embedder_info
            })
        
        return {
            "query": query,
            "search_type": search_type,
            "count": len(formatted_results),
            "embedder_info": {
                "dense": dense_model_id,
                "sparse": sparse_model_id
            },
            "results": formatted_results
        }

    def _create_smart_preview(self, text, query, context_size=300):
        """
        Create a better preview with clear highlighting of searched terms.
        
        Args:
            text: The full text to extract preview from
            query: Search query to look for
            context_size: Maximum context size in characters
        
        Returns:
            A preview string with context around matched terms and highlighting
        """
        if not text:
            return ""
        
        # If text is shorter than context_size, return the whole text
        if len(text) <= context_size:
            return self._highlight_query_terms(text, query)
        
        # Import re if not already imported
        import re
        
        # Normalize query and text for searching
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find the exact query in the text
        match_pos = text_lower.find(query_lower)
        
        # If exact query not found, try individual words
        if match_pos == -1:
            # Extract query terms, filtering out short words
            query_terms = [term for term in query_lower.split() if len(term) > 2]
            
            # Find positions of all query terms
            term_positions = []
            for term in query_terms:
                pos = text_lower.find(term)
                if pos != -1:
                    term_positions.append(pos)
            
            # If any terms were found, use the earliest one
            if term_positions:
                match_pos = min(term_positions)
            else:
                # No terms found, return beginning of text
                return text[:context_size] + "..."
        
        # Center the preview around the match position
        half_context = context_size // 2
        start = max(0, match_pos - half_context)
        end = min(len(text), match_pos + half_context)
        
        # Adjust to try to keep whole sentences
        # Look for sentence boundaries before start
        sentence_start = max(0, start - 100)  # Look back up to 100 chars
        potential_start = text.rfind('. ', sentence_start, start)
        if potential_start != -1:
            start = potential_start + 2  # Move past the period and space
        
        # Look for sentence boundaries after end
        potential_end = text.find('. ', end, min(len(text), end + 100))
        if potential_end != -1:
            end = potential_end + 1  # Include the period
        
        # Extract preview
        preview = text[start:end]
        
        # Add ellipses to indicate truncation
        if start > 0:
            preview = "..." + preview
        if end < len(text):
            preview += "..."
        
        # Highlight the query in the preview
        highlighted_preview = self._highlight_query_terms(preview, query)
        
        return highlighted_preview

    def _highlight_query_terms(self, text, query):
        """
        Highlight query terms in the text using bold markdown syntax.
        
        Args:
            text: Text to highlight terms in
            query: Query string containing terms to highlight
        
        Returns:
            Text with highlighted terms
        """
        if not text or not query:
            return text
        
        import re
        
        # Get both the full query and individual terms
        query_lower = query.lower()
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        
        # Add the full query to the list of terms to highlight
        if len(query) > 2 and query not in query_terms:
            query_terms.append(query_lower)
        
        # Sort terms by length (descending) to avoid partial matches
        query_terms = sorted(set(query_terms), key=len, reverse=True)
        
        # Initialize result text
        result = text
        
        # Loop through each term and highlight it
        for term in query_terms:
            # Skip highlighting for very short terms
            if len(term) <= 2:
                continue
                
            # Create a case-insensitive pattern to find the term
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            # Find all occurrences of the term
            matches = list(pattern.finditer(result))
            
            # Process matches from end to start to avoid position issues
            offset = 0
            for match in matches:
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                
                # Extract the original term with original case
                original_term = result[start_pos:end_pos]
                
                # Replace with highlighted version
                result = result[:start_pos] + f"**{original_term}**" + result[end_pos:]
                
                # Update offset for next replacements
                offset += 4  # Length of added characters "**" + "**"
        
        return result
    
    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """
        Perform a sparse vector search according to Qdrant documentation format
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        # Get vector names
        has_mlx_provider = (
            processor is not None
            and hasattr(processor, 'mlx_embedding_provider')
            and processor.mlx_embedding_provider is not None
        )
        dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
        sparse_model_id = getattr(processor, 'sparse_model_id', self.sparse_model_id)
        
        # Sanitize vector names
        dense_model_id = dense_model_id.replace("-", "_").replace("/", "_")
        sparse_model_id = sparse_model_id.replace("-", "_").replace("/", "_")
        
        dense_vector_name = f"dense_{dense_model_id}"
        sparse_vector_name = f"sparse_{sparse_model_id}"
        
        if self.verbose:
            print(f"[DEBUG] Sparse search using vector name '{sparse_vector_name}'")

        try:
            # Generate sparse vector
            if has_mlx_provider:
                sparse_indices, sparse_values = processor.mlx_embedding_provider.get_sparse_embedding(query)
            else:
                sparse_indices, sparse_values = processor._generate_sparse_vector(query)
            
            # Warn if this appears to be just a bag of words
            if len(sparse_indices) < 5 or max(sparse_values) < 0.1:
                print(f"⚠️ WARNING: Sparse vector appears to be low quality (possibly just a bag of words).")
                print(f"This might significantly reduce search quality.")
                
            if self.verbose:
                print(f"[DEBUG] Created sparse vector with {len(sparse_indices)} non-zero terms")
                if len(sparse_indices) > 0:
                    print(f"Sample indices: {sparse_indices[:5]}")
                    print(f"Sample values: {sparse_values[:5]}")
            
            # Perform search according to Qdrant documentation format
            try:
                # Format the query according to Qdrant documentation
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(sparse_vector_name, {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }),
                    limit=limit * 3,  # Get more results to filter later
                    with_payload=True,
                    score_threshold=score_threshold
                )
                
                if self.verbose:
                    print(f"[DEBUG] Sparse search returned {len(search_result)} results")
                
                return search_result
                
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Standard sparse search failed: {e}")
                    print("Trying alternative sparse search methods...")
                
                # Try with NamedSparseVector format (newer Qdrant versions)
                try:
                    from qdrant_client.models import NamedSparseVector, SparseVector
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=NamedSparseVector(
                            name=sparse_vector_name,
                            vector=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        ),
                        limit=limit * 3,
                        with_payload=True,
                        score_threshold=score_threshold
                    )
                    
                    if self.verbose:
                        print(f"[DEBUG] Alternative sparse search returned {len(search_result)} results")
                    
                    return search_result
                    
                except Exception as e2:
                    # Last resort - try with query_points
                    try:
                        from qdrant_client.models import SparseVector
                        result = self.client.query_points(
                            collection_name=self.collection_name,
                            query=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using=sparse_vector_name,
                            limit=limit * 3,
                            with_payload=True,
                            score_threshold=score_threshold
                        )
                        
                        if self.verbose:
                            print(f"[DEBUG] query_points sparse search returned {len(result.points)} results")
                        
                        return result.points
                    except Exception as e3:
                        return {"error": f"All sparse search methods failed: {e3}"}
        
        except Exception as e:
            print(f"Error generating sparse vectors: {e}")
            return {"error": f"Error in sparse search: {str(e)}"}
    
        
    def cleanup(self, remove_storage=False):
        """Clean up resources"""
        # Close the client if it exists
        if hasattr(self, 'client') and self.client:
            try:
                if hasattr(self.client, 'close'):
                    self.client.close()
            except:
                pass
                
        # Remove the storage directory only if requested
        if remove_storage and hasattr(self, 'storage_path') and self.storage_path and os.path.exists(self.storage_path):
            try:
                if self.verbose:
                    print(f"Removing storage directory: {self.storage_path}")
                shutil.rmtree(self.storage_path, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up storage directory: {e}")


def get_files_to_process(
    search_dir: str,
    include_patterns: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    limit: Optional[int] = None,
    verbose: bool = False
) -> List[str]:
    """Get list of files to process based on patterns and limits"""
    if verbose:
        print(f"Scanning directory: {search_dir}")
        if include_patterns:
            print(f"Include patterns: {include_patterns}")
        if exclude_dirs:
            print(f"Exclude directories: {exclude_dirs}")
        if limit:
            print(f"Limit: {limit} files")
    
    files = []
    exclude_dirs = exclude_dirs or []
    
    # Normalize exclude dirs
    exclude_dirs = [os.path.normpath(d) for d in exclude_dirs]
    
    # Default to all supported extensions if no patterns provided
    if not include_patterns:
        include_patterns = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]
    
    for root, dirs, filenames in os.walk(search_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d)) not in exclude_dirs]
        
        for filename in filenames:
            file_path = os.path.join(root, filename)
            
            # Check if file matches any pattern
            if any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
                files.append(file_path)
                
                if verbose and len(files) % 100 == 0:
                    print(f"Found {len(files)} matching files so far...")
                
                # Check limit
                if limit is not None and len(files) >= limit:
                    if verbose:
                        print(f"Reached limit of {limit} files")
                    return files[:limit]
    
    if verbose:
        print(f"Found {len(files)} files to process")
    
    return files


def check_qdrant_available(host="localhost", port=6333, verbose=False):
    """Check if Qdrant is available (either server or local)"""
    try:
        if not qdrant_available:
            print("Qdrant client not available. Install with: pip install qdrant-client")
            return False
            
        # First try remote connection if specified
        if host != "localhost" or port != 6333:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.settimeout(2)
                s.connect((host, port))
                s.close()
                if verbose:
                    print(f"✓ Qdrant server available at {host}:{port}")
                return True
            except:
                if verbose:
                    print(f"✗ Remote Qdrant server not available at {host}:{port}")
                    print("Falling back to local mode")
        
        # Try local mode
        try:
            # Create a test client with local storage
            temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
            
            # Initialize client
            client = QdrantClient(path=temp_dir)
            
            # Create a small test collection to verify functionality
            client.create_collection(
                collection_name="test_collection",
                vectors_config=VectorParams(size=4, distance=Distance.COSINE)
            )
            
            # Get collections to verify it worked
            collections = client.get_collections()
            
            # Clean up
            client.delete_collection("test_collection")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if verbose:
                print("✓ Local Qdrant storage mode is available")
            return True
        except Exception as e:
            print(f"Error testing Qdrant in local mode: {str(e)}")
            print("Please install Qdrant client: pip install qdrant-client")
            return False
    except:
        return False


def run_indexing(args):
    """Main function to run the indexing process with SPLADE support"""

    # First check if Qdrant is available
    if not check_qdrant_available(args.host, args.port, args.verbose):
        print("ERROR: Qdrant not available in either server or local mode")
        return
        
    start_time = time.time()
    
    # 1. Download model if needed (only needed for fallback model)
    if args.verbose:
        print("\n====== STEP 1: Downloading Model (if needed) ======")
    download_model_files(args.model, args.weights, args.verbose)
    
    # 2. Initialize document processor
    if args.verbose:
        print("\n====== STEP 2: Initializing Document Processor ======")
    
    # Custom model params
    custom_repo_id = getattr(args, 'custom_repo_id', None)
    custom_ndim = getattr(args, 'custom_ndim', None)
    custom_pooling = getattr(args, 'custom_pooling', 'mean')
    custom_normalize = getattr(args, 'custom_normalize', True)
    custom_max_length = getattr(args, 'custom_max_length', 512)
    
    # Use MLX embedding models if available
    processor_args = {
        "model_name": args.model,
        "weights_path": args.weights,
        "verbose": args.verbose
    }
    
    if hasattr(args, 'use_mlx_models') and HAS_MLX_EMBEDDING_MODELS:
        processor_args.update({
            "dense_model": args.dense_model,
            "sparse_model": args.sparse_model,
            "top_k": getattr(args, 'top_k', 64),
            "custom_repo_id": custom_repo_id,
            "custom_ndim": custom_ndim,
            "custom_pooling": custom_pooling,
            "custom_normalize": custom_normalize,
            "custom_max_length": custom_max_length,
            "use_mlx_embedding": args.use_mlx_models
        })
    
    processor = DocumentProcessor(**processor_args)
    
    # 3. Initialize Qdrant manager and create collection
    if args.verbose:
        print("\n====== STEP 3: Setting up Qdrant ======")

    try:
        qdrant = QdrantManager(
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            vector_size=processor.vector_size,
            storage_path=args.storage_path,
            verbose=args.verbose,
            dense_model_id=args.dense_model,
            sparse_model_id=args.sparse_model
        )
        qdrant.create_collection(recreate=args.recreate)
    except Exception as e:
        print(f"Error setting up Qdrant: {e}")
        return
    
    # 4. Get files to process
    if args.verbose:
        print("\n====== STEP 4: Finding Files to Process ======")
    include_patterns = args.include.split() if args.include else None
    files = get_files_to_process(
        args.directory,
        include_patterns=include_patterns,
        limit=args.limit,
        verbose=args.verbose
    )
    
    if not files:
        print("No files found to process")
        return
    
    # 5. Process files and index them
    if args.verbose:
        print(f"\n====== STEP 5: Processing and Indexing {len(files)} Files ======")
    
    # Safely get collection info, handling errors gracefully
    try:
        collection_info_before = qdrant.get_collection_info()
        points_before = collection_info_before.get("points_count", 0)
        if isinstance(points_before, dict) and "error" in points_before:
            points_before = 0
    except Exception as e:
        print(f"Warning: Unable to get initial collection info: {e}")
        points_before = 0
    
    total_chunks = 0
    total_files_processed = 0
    successful_files = 0
    
    # Use tqdm for progress bar 
    if HAS_TQDM: # and args.verbose:
        file_iterator = tqdm(files, desc="Processing files")
    else:
        file_iterator = files
    
    # Determine whether to use regular or sparse processing
    use_sparse = hasattr(args, 'use_mlx_models') and args.use_mlx_models and hasattr(processor, 'mlx_embedding_provider')
    
    for file_path in file_iterator:
        try:
            # Show progress details when verbose
            if args.verbose and not HAS_TQDM:
                print(f"\nProcessing file {total_files_processed + 1}/{len(files)}: {file_path}")
                
                # Show system stats periodically
                if total_files_processed % 10 == 0 and HAS_PSUTIL:
                    mem = psutil.virtual_memory()
                    cpu = psutil.cpu_percent()
                    print(f"System stats - CPU: {cpu}%, RAM: {mem.percent}% ({get_size_str(mem.used)}/{get_size_str(mem.total)})")
            
            # Process file with appropriate method (with or without sparse)
            if use_sparse:
                # Process with sparse embeddings
                try:
                    results = processor.process_file_with_sparse(file_path)
                    
                    if results:
                        # Insert embeddings with sparse vectors
                        qdrant.insert_embeddings_with_sparse(results)
                        total_chunks += len(results)
                        successful_files += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Process with dense embeddings only
                try:
                    results = processor.process_file(file_path)
                    
                    if results:
                        # Insert embeddings
                        qdrant.insert_embeddings(results)
                        total_chunks += len(results)
                        successful_files += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            total_files_processed += 1
            
            # Show progress update
            if args.verbose and not HAS_TQDM:
                print(f"Completed {total_files_processed}/{len(files)} files, {total_chunks} chunks indexed")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. Show final statistics
    try:
        collection_info_after = qdrant.get_collection_info()
        points_after = collection_info_after.get("points_count", 0)
        if isinstance(points_after, dict) and "error" in points_after:
            points_after = total_chunks
        points_added = points_after - points_before
    except Exception as e:
        print(f"Warning: Unable to get final collection info: {e}")
        points_after = total_chunks
        points_added = total_chunks
    
    print("\n====== Indexing Complete ======")
    print(f"Processed {total_files_processed} files")
    print(f"Successfully indexed {successful_files} files")
    print(f"Added {points_added} chunks to Qdrant")
    print(f"Total chunks in collection: {points_after}")
    
    # Safely display collection info
    try:
        if "disk_usage" in collection_info_after and collection_info_after["disk_usage"]:
            print(f"Collection size on disk: {get_size_str(collection_info_after['disk_usage'])}")
        
        # Show vector configurations
        if "vector_configs" in collection_info_after and collection_info_after["vector_configs"]:
            print("\nVector configurations:")
            for name, config in collection_info_after["vector_configs"].items():
                print(f"  - {name}: {config}")
                
        if "sparse_vector_configs" in collection_info_after and collection_info_after["sparse_vector_configs"]:
            print("\nSparse vector configurations:")
            for name, config in collection_info_after["sparse_vector_configs"].items():
                print(f"  - {name}: {config}")
    except Exception as e:
        print(f"Warning: Error displaying collection info details: {e}")
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    


def run_search(args):
    """Run search on the existing collection with enhanced display and relevance"""
    # 1. Initialize Qdrant manager
    if args.verbose:
        print("\n====== Initializing Qdrant Connection ======")
    qdrant = QdrantManager(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        storage_path=args.storage_path,  # Add storage path for better local support
        verbose=args.verbose,
        dense_model_id=args.dense_model,
        sparse_model_id=args.sparse_model
    )
    
    # 2. Load model for vector search if needed - but NOT for keyword search
    processor = None
    if args.search_type in ["vector", "sparse", "hybrid"]:
        if args.verbose:
            print("\n====== Loading Models for Search ======")
            
        # Custom model params
        custom_repo_id = getattr(args, 'custom_repo_id', None)
        custom_ndim = getattr(args, 'custom_ndim', None)
        custom_pooling = getattr(args, 'custom_pooling', 'mean')
        custom_normalize = getattr(args, 'custom_normalize', True)
        custom_max_length = getattr(args, 'custom_max_length', 512)
            
        # Initialize document processor with MLX embedding models if available
        processor_args = {
            "model_name": args.model,
            "weights_path": args.weights,
            "verbose": args.verbose
        }
        
        if hasattr(args, 'use_mlx_models') and HAS_MLX_EMBEDDING_MODELS and args.use_mlx_models:
            processor_args.update({
                "dense_model": args.dense_model,
                "sparse_model": args.sparse_model,
                "top_k": getattr(args, 'top_k', 64),
                "custom_repo_id": custom_repo_id,
                "custom_ndim": custom_ndim,
                "custom_pooling": custom_pooling,
                "custom_normalize": custom_normalize,
                "custom_max_length": custom_max_length,
                "use_mlx_embedding": True
            })
            
            if args.verbose:
                if custom_repo_id:
                    print(f"Using custom model: {custom_repo_id}")
                else:
                    print(f"Using MLX embedding models: {args.dense_model} (dense), {args.sparse_model} (sparse)")
        
        processor = DocumentProcessor(**processor_args)
    
    # 3. Run search with improved relevance 
    if args.verbose:
        print(f"\n====== Running {args.search_type.capitalize()} Search ======")
    
    # Read new parameters from args (provide defaults for backward compatibility)
    relevance_tuning = getattr(args, 'relevance_tuning', True)
    context_size = getattr(args, 'context_size', 300)
    score_threshold = getattr(args, 'score_threshold', None)
    rerank = getattr(args, 'rerank', False)  # Add support for reranking parameter
    
    # Use the search method from QdrantManager
    results = qdrant.search(
        query=args.query,
        search_type=args.search_type,
        limit=args.limit,
        processor=processor,
        prefetch_limit=args.prefetch_limit,
        fusion_type=args.fusion,
        relevance_tuning=relevance_tuning,
        context_size=context_size,
        score_threshold=score_threshold,
        rerank=rerank
    )
    
    # 4. Display results 
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Use terminal colors if enabled
    show_debug = getattr(args, 'debug', False)
    color_output = not getattr(args, 'no_color', False)
    
    if color_output:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        BLUE = "\033[34m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        HIGHLIGHT = "\033[43m"  # Yellow background
    else:
        RESET = BOLD = BLUE = YELLOW = CYAN = HIGHLIGHT = ""
    
    # Display header
    print(f"\n{BOLD}{BLUE}====== Search Results ======{RESET}\n")
    print(f"Query: {BOLD}'{results['query']}'{RESET}")
    print(f"Search type: {results['search_type']}")
    print(f"Using embedders: {results['embedder_info']['dense']} (dense), "
          f"{results['embedder_info']['sparse']} (sparse)")
    print(f"Found {BOLD}{results['count']}{RESET} results\n")
    
    if results['count'] == 0:
        print(f"{YELLOW}No results found for your query.{RESET}")
        return
    
    # Display results
    for result in results['results']:
        print(f"{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}Rank: {result['rank']}, Score: {YELLOW}{result['score']:.4f}{RESET}")
        print(f"File: {result['file_name']}")
        print(f"Path: {result['file_path']}")
        print(f"Chunk: {result['chunk_index']}")
        
        # Chunk size information
        print(f"Chunk size: {result['chunk_size']['characters']} chars, "
              f"{result['chunk_size']['words']} words, "
              f"{result['chunk_size']['lines']} lines")
        
        print(f"Embedders: {result['embedder_info']['dense_embedder']} (dense), "
              f"{result['embedder_info']['sparse_embedder']} (sparse)")
        
        # Preview with highlighted terms
        print(f"\n{BOLD}Preview:{RESET}")
        
        # Convert markdown ** to terminal formatting
        preview = result['preview']
        if color_output:
            preview = preview.replace('**', HIGHLIGHT)
            # Fix uneven highlights by adding RESET
            if preview.count(HIGHLIGHT) % 2 != 0:
                preview += RESET
            else:
                preview = preview.replace(HIGHLIGHT + HIGHLIGHT, HIGHLIGHT)
                
            # Make sure highlighting is reset at the end
            preview += RESET
        
        print(preview)
        print()


def main():
    """Main function with additional SPLADE support and flexible model selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Qdrant Indexer with MLX and SPLADE support")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--storage-path", help="Path to store Qdrant data")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("directory", help="Directory containing documents to index")
    index_parser.add_argument("--include", type=str, help="File patterns to include (space separated, e.g. '*.txt *.pdf')")
    index_parser.add_argument("--limit", type=int, help="Maximum number of files to index")
    index_parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    index_parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    index_parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name")
    index_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model name (fallback)")
    index_parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to store/load MLX weights (fallback)")
    index_parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    
    # Add MLX Embedding Models arguments
    index_parser.add_argument("--use-mlx-models", action="store_true", help="Use mlx_embedding_models instead of traditional models")
    index_parser.add_argument("--dense-model", type=str, default=DEFAULT_DENSE_MODEL, help="MLX dense embedding model name")
    index_parser.add_argument("--sparse-model", type=str, default=DEFAULT_SPARSE_MODEL, help="MLX sparse embedding model name")
    index_parser.add_argument("--top-k", type=int, default=64, help="Top-k tokens to keep in sparse vectors")
    
    # Add custom model parameters
    index_parser.add_argument("--custom-repo-id", type=str, help="Custom model HuggingFace repo ID")
    index_parser.add_argument("--custom-ndim", type=int, help="Custom model embedding dimension")
    index_parser.add_argument("--custom-pooling", type=str, choices=["mean", "first", "max"], default="mean", 
                             help="Custom model pooling strategy")
    index_parser.add_argument("--custom-normalize", action="store_true", default=True, help="Normalize embeddings")
    index_parser.add_argument("--custom-max-length", type=int, default=512, help="Custom model max sequence length")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--search-type", choices=["keyword", "vector", "sparse", "hybrid"], default="hybrid", 
                             help="Type of search to perform")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    search_parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    search_parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    search_parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name")
    search_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model name (fallback)")
    search_parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to MLX weights (fallback)")
    search_parser.add_argument("--prefetch-limit", type=int, default=50, help="Prefetch limit for hybrid search")
    search_parser.add_argument("--fusion", choices=["rrf", "dbsf"], default="rrf", help="Fusion strategy for hybrid search")
    search_parser.add_argument("--relevance-tuning", action="store_true", default=True, 
                            help="Apply relevance tuning to hybrid search")
    search_parser.add_argument("--context-size", type=int, default=300, 
                            help="Size of context window for preview text")
    search_parser.add_argument("--score-threshold", type=float,
                            help="Minimum score threshold for results (0.0-1.0)")
    search_parser.add_argument("--debug", action="store_true", 
                            help="Show detailed debug information")
    search_parser.add_argument("--no-color", action="store_true", 
                            help="Disable colored output")

    # Add MLX Embedding Models arguments for search
    search_parser.add_argument("--use-mlx-models", action="store_true", help="Use mlx_embedding_models instead of traditional models")
    search_parser.add_argument("--dense-model", type=str, default=DEFAULT_DENSE_MODEL, help="MLX dense embedding model name")
    search_parser.add_argument("--sparse-model", type=str, default=DEFAULT_SPARSE_MODEL, help="MLX sparse embedding model name")
    search_parser.add_argument("--top-k", type=int, default=64, help="Top-k tokens for sparse vectors")
    
    # Add custom model parameters for search
    search_parser.add_argument("--custom-repo-id", type=str, help="Custom model HuggingFace repo ID")
    search_parser.add_argument("--custom-ndim", type=int, help="Custom model embedding dimension")
    search_parser.add_argument("--custom-pooling", type=str, choices=["mean", "first", "max"], default="mean", 
                             help="Custom model pooling strategy")
    search_parser.add_argument("--custom-normalize", action="store_true", default=True, help="Normalize embeddings")
    search_parser.add_argument("--custom-max-length", type=int, default=512, help="Custom model max sequence length")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available MLX embedding models")

    args = parser.parse_args()
    
    if args.command == "index":
        # Check if required dependencies are installed
        missing_deps = []
        try:
            import qdrant_client
        except ImportError:
            missing_deps.append("qdrant-client")
            
        if args.use_mlx_models and not HAS_MLX_EMBEDDING_MODELS:
            print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")
            print("Falling back to traditional model")
        
        run_indexing(args)
    elif args.command == "search":
        if not args.query:
            print("Error: Search query cannot be empty")
            return
            
        if args.use_mlx_models and not HAS_MLX_EMBEDDING_MODELS:
            print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")
            print("Falling back to traditional model")
            
        run_search(args)
    elif args.command == "list-models":
        # List available models in the registry
        if not HAS_MLX_EMBEDDING_MODELS:
            print("Error: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")
            return
            
        print("\n====== Available MLX Embedding Models ======")
        print("\nDense Models:")
        dense_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if not v.get("lm_head")]
        for i, model in enumerate(dense_models):
            dim = MLX_EMBEDDING_REGISTRY[model].get("ndim", "unknown")
            repo = MLX_EMBEDDING_REGISTRY[model].get("repo", "unknown")
            print(f"{i+1:2d}. {model:30s} - Dim: {dim:4d}, Repo: {repo}")
            
        print("\nSparse Models (SPLADE):")
        sparse_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if v.get("lm_head")]
        for i, model in enumerate(sparse_models):
            dim = MLX_EMBEDDING_REGISTRY[model].get("ndim", "unknown")
            repo = MLX_EMBEDDING_REGISTRY[model].get("repo", "unknown")
            print(f"{i+1:2d}. {model:30s} - Dim: {dim:4d}, Repo: {repo}")
            
        print("\nNote: You can also use custom models with --custom-repo-id parameter")
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
