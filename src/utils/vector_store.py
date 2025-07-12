"""
Vector store utilities for the RAG chatbot

This module provides comprehensive vector store management for the pediatric RAG chatbot.
It handles document embedding, storage, and retrieval using FAISS (Facebook AI Similarity Search)
as the backend vector database.

Key Features:
- Efficient document embedding using HuggingFace models
- FAISS-based vector storage for fast similarity search
- Support for incremental document addition
- Persistence (save/load) for vector stores
- Similarity search with relevance scoring
- Memory-efficient operations for large document collections

The VectorStore class abstracts the complexity of working with embeddings and provides
a clean interface for the RAG pipeline to store and retrieve relevant documents.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

# LangChain imports for document handling and vector operations
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # For creating embeddings
from langchain_community.vectorstores import FAISS               # Facebook AI Similarity Search
import numpy as np                                                # Numerical operations

# Set up logging for this module
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Enhanced vector store manager with FAISS backend
    
    This class provides a comprehensive interface for managing document embeddings
    in a FAISS vector database. It's designed specifically for RAG applications
    where quick similarity search over large document collections is essential.
    
    Key capabilities:
    1. Document Embedding: Converts text documents into high-dimensional vectors
    2. Efficient Storage: Uses FAISS for optimized vector storage and retrieval
    3. Similarity Search: Finds documents semantically similar to queries
    4. Persistence: Save and load vector stores to/from disk
    5. Incremental Updates: Add new documents to existing stores
    
    The embedding model converts text into numerical vectors that capture semantic meaning.
    Documents with similar meanings will have similar vector representations, enabling
    effective semantic search for the RAG system.
    """
    
    def __init__(self, embedding_model_name: str = "Qwen/Qwen3-Embedding-8B"):
        """
        Initialize the VectorStore with specified embedding model
        
        Args:
            embedding_model_name (str): Name of the HuggingFace embedding model
                                      Default: "Qwen/Qwen3-Embedding-8B"
                                      - High-quality embeddings optimized for Qwen models
                                      - Better semantic understanding than general models
                                      - Optimal compatibility with Qwen3-8B language model
        
        The embedding model is crucial for RAG performance:
        - Qwen embeddings: Best compatibility with Qwen language models
        - Higher quality: Better semantic understanding for domain-specific content
        - Model alignment: Same architecture family as the generation model
        """
        self.embedding_model_name = embedding_model_name
        
        # Initialize HuggingFace embeddings with specified configuration
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},              # Use CPU (change to 'cuda' for GPU)
            encode_kwargs={'normalize_embeddings': True}  # L2 normalization for better similarity
        )
        
        # Vector store instance (initialized when documents are added)
        self.vector_store = None
        
        logger.info(f"Initialized VectorStore with embedding model: {embedding_model_name}")
    
    def create_from_documents(self, documents: List[Document]) -> bool:
        """
        Create vector store from a list of Document objects
        
        This method creates a new FAISS vector store from a collection of LangChain Document objects.
        It embeds all documents using the configured embedding model and builds an index for
        fast similarity search.
        
        Args:
            documents (List[Document]): List of LangChain Document objects to embed
            
        Returns:
            bool: True if successful, False if error occurred
            
        Process:
        1. Convert documents to embeddings using the HuggingFace model
        2. Create FAISS index for efficient similarity search
        3. Store both embeddings and original document content
        """
        try:
            if not documents:
                logger.warning("No documents provided to create vector store")
                return False
            
            logger.info(f"Creating vector store from {len(documents)} documents...")
            
            # FAISS.from_documents handles the embedding process internally:
            # 1. Extracts text content from Document objects
            # 2. Converts text to embeddings using the embedding model
            # 3. Builds FAISS index for fast similarity search
            # 4. Associates embeddings with original document metadata
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            logger.info("✅ Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            return False
    
    def create_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool:
        """Create vector store from a list of texts"""
        try:
            if not texts:
                logger.warning("No texts provided to create vector store")
                return False
            
            logger.info(f"Creating vector store from {len(texts)} texts...")
            self.vector_store = FAISS.from_texts(
                texts, 
                self.embeddings,
                metadatas=metadatas
            )
            logger.info("✅ Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to existing vector store"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized. Creating new one...")
                return self.create_from_documents(documents)
            
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            logger.info(f"Adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents(documents)
            logger.info("✅ Documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool:
        """Add texts to existing vector store"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized. Creating new one...")
                return self.create_from_texts(texts, metadatas)
            
            if not texts:
                logger.warning("No texts provided to add")
                return False
            
            logger.info(f"Adding {len(texts)} texts to vector store...")
            self.vector_store.add_texts(texts, metadatas=metadatas)
            logger.info("✅ Texts added successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding texts: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            logger.debug(f"Searching for similar documents: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with relevance scores"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            logger.debug(f"Searching with scores: {query}")
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching documents with scores: {str(e)}")
            return []
    
    def save(self, path: str) -> bool:
        """Save vector store to disk"""
        try:
            if not self.vector_store:
                logger.warning("No vector store to save")
                return False
            
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving vector store to: {path}")
            self.vector_store.save_local(path)
            logger.info("✅ Vector store saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving vector store: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """Load vector store from disk"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Vector store path does not exist: {path}")
                return False
            
            logger.info(f"Loading vector store from: {path}")
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {str(e)}")
            return False
    
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store"""
        try:
            if not self.vector_store:
                return 0
            return self.vector_store.index.ntotal
        except Exception as e:
            logger.error(f"❌ Error getting vector count: {str(e)}")
            return 0
    
    def delete_by_ids(self, ids: List[str]) -> bool:
        """Delete documents by their IDs"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return False
            
            logger.info(f"Deleting {len(ids)} documents")
            self.vector_store.delete(ids)
            logger.info("✅ Documents deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {str(e)}")
            return False
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None):
        """Get the vector store as a retriever"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return None
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )