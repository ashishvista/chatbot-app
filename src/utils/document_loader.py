"""
Document loading utilities for the RAG chatbot

This module provides comprehensive document loading capabilities for the pediatric RAG chatbot.
It handles multiple file formats (TXT, MD, PDF) and includes text preprocessing and chunking
functionality to prepare documents for vector embedding and retrieval.

Key Features:
- Multi-format document loading (TXT, Markdown, PDF)
- Intelligent text chunking with overlap for context preservation
- Document preprocessing and cleaning
- Error handling and logging for robust operation
- Legacy compatibility functions for backward compatibility

The DocumentLoader class is the main interface for loading and processing documents
that will be used to build the knowledge base for the RAG system.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports for document handling and text processing
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,           # For loading plain text files
    DirectoryLoader,      # For batch loading from directories
    PyPDFLoader,         # For loading PDF documents
    UnstructuredMarkdownLoader  # For loading Markdown files with structure preservation
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For intelligent text chunking

# Set up logging for this module
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Enhanced document loader with support for multiple file types
    
    This class provides comprehensive document loading capabilities for the pediatric RAG chatbot.
    It handles the complete pipeline from raw documents to processed chunks ready for embedding:
    
    1. Multi-format Loading: Supports TXT, Markdown, and PDF files
    2. Text Processing: Cleans and preprocesses document content
    3. Intelligent Chunking: Splits documents into overlapping chunks for better retrieval
    4. Batch Processing: Efficiently handles entire directories of documents
    
    The chunking strategy is crucial for RAG systems:
    - chunk_size: Maximum size of each text chunk (affects retrieval granularity)
    - chunk_overlap: Overlap between chunks (preserves context across boundaries)
    - Recursive splitting: Uses multiple separators (paragraphs, sentences, words)
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentLoader with chunking parameters
        
        Args:
            chunk_size (int): Maximum number of characters per chunk
                             Smaller chunks = more precise retrieval but less context
                             Larger chunks = more context but less precise matching
            chunk_overlap (int): Number of characters to overlap between chunks
                                This helps preserve context across chunk boundaries
        
        The text splitter uses a hierarchical approach:
        1. Try to split on paragraph breaks (\n\n)
        2. Fall back to line breaks (\n)
        3. Fall back to spaces
        4. Finally split by characters if needed
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter is the recommended splitter for most use cases
        # It tries to keep related content together by using semantic separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # Maximum chunk size in characters
            chunk_overlap=chunk_overlap,    # Overlap to preserve context
            length_function=len,            # Function to measure text length
            separators=["\n\n", "\n", " ", ""]  # Hierarchy of split points
        )
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        This method performs batch loading of documents from a specified directory.
        It supports multiple file formats and handles errors gracefully.
        
        Args:
            directory (str): Path to directory containing documents
            
        Returns:
            List[Document]: List of loaded LangChain Document objects
            
        Supported formats:
        - .txt: Plain text files (patient histories, guidelines)
        - .md: Markdown files (structured medical documents)
        - .pdf: PDF files (research papers, clinical guidelines)
        
        The method uses DirectoryLoader for efficient batch processing
        and logs progress for monitoring large document collections.
        """
        documents = []
        
        # Check if directory exists before attempting to load
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return documents
        
        # Define loaders for different file types
        # Each file type requires a specialized loader for proper content extraction
        loaders = {
            "**/*.txt": TextLoader,                    # Plain text files
            "**/*.md": UnstructuredMarkdownLoader,     # Markdown with structure preservation
            "**/*.pdf": PyPDFLoader                    # PDF documents with text extraction
        }
        
        # Process each file type with its appropriate loader
        for pattern, loader_class in loaders.items():
            try:
                # DirectoryLoader handles batch loading with glob patterns
                # show_progress=True provides feedback during large document loading
                loader = DirectoryLoader(
                    directory,
                    glob=pattern,                  # File pattern to match
                    loader_cls=loader_class,       # Loader class for this file type
                    show_progress=True             # Show progress bar for user feedback
                )
                
                # Load all documents matching the current pattern
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} files matching {pattern}")
                
            except Exception as e:
                # Log errors but continue processing other file types
                logger.error(f"Error loading files with pattern {pattern}: {str(e)}")
        
        return documents
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document file
        
        This method loads an individual document file and returns it as a list of Document objects.
        It automatically detects the file type based on extension and uses the appropriate loader.
        
        Args:
            file_path (str): Path to the document file to load
            
        Returns:
            List[Document]: List containing the loaded document (usually just one)
            
        File type detection:
        - .txt files use TextLoader for plain text content
        - .md files use UnstructuredMarkdownLoader to preserve structure
        - .pdf files use PyPDFLoader for text extraction from PDF content
        """
        try:
            # Extract file extension to determine appropriate loader
            file_extension = Path(file_path).suffix.lower()
            
            # Select loader based on file extension
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            # Load and return the document
            return loader.load()
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval
        
        This method takes a list of documents and splits them into smaller chunks
        using the configured RecursiveCharacterTextSplitter. Chunking is essential
        for RAG systems because it:
        
        1. Improves retrieval precision by creating smaller, focused segments
        2. Reduces context window limitations for language models
        3. Enables better semantic matching between queries and content
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of document chunks with preserved metadata
        """
        if not documents:
            return []
        
        try:
            # Split documents using the configured text splitter
            # This preserves document metadata while creating smaller chunks
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return documents
    
    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a single document to clean and standardize content
        
        This method performs basic text cleaning operations to improve
        the quality of document content for embedding and retrieval:
        
        1. Removes leading/trailing whitespace
        2. Normalizes excessive whitespace to single spaces
        3. Preserves document metadata for tracking and filtering
        
        Args:
            document (Document): LangChain Document object to preprocess
            
        Returns:
            Document: Cleaned Document object with same metadata
        """
        # Clean the content by removing leading/trailing whitespace
        content = document.page_content.strip()
        
        # Normalize excessive whitespace to single spaces
        # This helps with consistent text processing and reduces noise
        content = ' '.join(content.split())
        
        # Update the document with cleaned content while preserving metadata
        document.page_content = content
        return document
    
    def load_and_process_documents(self, directory: str) -> List[Document]:
        """
        Complete pipeline: load, preprocess, and split documents
        
        This is the main method that orchestrates the entire document processing pipeline:
        1. Load documents from directory (multiple file formats)
        2. Preprocess each document (clean text)
        3. Split documents into chunks (for better retrieval)
        
        Args:
            directory (str): Path to directory containing documents
            
        Returns:
            List[Document]: List of processed and chunked documents ready for embedding
        """
        logger.info(f"Loading and processing documents from: {directory}")
        
        # Step 1: Load documents from the specified directory
        documents = self.load_documents_from_directory(directory)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # Step 2: Preprocess documents to clean and standardize content
        documents = [self.preprocess_document(doc) for doc in documents]
        
        # Step 3: Split documents into smaller chunks for better retrieval
        splits = self.split_documents(documents)
        
        logger.info(f"Successfully processed {len(documents)} documents into {len(splits)} chunks")
        return splits

# Legacy compatibility functions for backward compatibility with existing code
# These functions maintain the same interface as the original implementation
# while leveraging the enhanced DocumentLoader class internally

def load_documents_from_directory(directory: str) -> List[str]:
    """
    Legacy function - returns document contents as strings
    
    This function maintains backward compatibility with existing code
    that expects a list of string contents rather than Document objects.
    """
    loader = DocumentLoader()
    documents = loader.load_documents_from_directory(directory)
    return [doc.page_content for doc in documents]

def load_document(file_path: str) -> str:
    """
    Legacy function - returns single document content as string
    
    Loads a single document and returns just the text content,
    maintaining compatibility with older code.
    """
    loader = DocumentLoader()
    documents = loader.load_single_document(file_path)
    return documents[0].page_content if documents else ""

def preprocess_document(document: str) -> str:
    """
    Legacy function - preprocesses document content
    
    Simple text preprocessing that maintains backward compatibility.
    """
    return document.strip()

def load_and_preprocess_documents(directory: str) -> List[str]:
    """
    Legacy function - returns preprocessed document contents as strings
    
    Complete pipeline that returns list of processed document strings
    rather than Document objects, for backward compatibility.
    """
    loader = DocumentLoader()
    documents = loader.load_and_process_documents(directory)
    return [doc.page_content for doc in documents]