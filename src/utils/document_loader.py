"""
Document loading utilities for the RAG chatbot
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Enhanced document loader with support for multiple file types"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return documents
        
        # Load different file types
        loaders = {
            "**/*.txt": TextLoader,
            "**/*.md": UnstructuredMarkdownLoader,
            "**/*.pdf": PyPDFLoader
        }
        
        for pattern, loader_class in loaders.items():
            try:
                loader = DirectoryLoader(
                    directory,
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=True
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} files matching {pattern}")
                
            except Exception as e:
                logger.error(f"Error loading files with pattern {pattern}: {str(e)}")
        
        return documents
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document file"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        try:
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return documents
    
    def preprocess_document(self, document: Document) -> Document:
        """Preprocess a single document"""
        # Clean the content
        content = document.page_content.strip()
        
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Update the document
        document.page_content = content
        return document
    
    def load_and_process_documents(self, directory: str) -> List[Document]:
        """Complete pipeline: load, preprocess, and split documents"""
        logger.info(f"Loading and processing documents from: {directory}")
        
        # Load documents
        documents = self.load_documents_from_directory(directory)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # Preprocess documents
        documents = [self.preprocess_document(doc) for doc in documents]
        
        # Split documents
        splits = self.split_documents(documents)
        
        logger.info(f"Successfully processed {len(documents)} documents into {len(splits)} chunks")
        return splits

# Convenience functions for backward compatibility
def load_documents_from_directory(directory: str) -> List[str]:
    """Legacy function - returns document contents as strings"""
    loader = DocumentLoader()
    documents = loader.load_documents_from_directory(directory)
    return [doc.page_content for doc in documents]

def load_document(file_path: str) -> str:
    """Legacy function - returns single document content as string"""
    loader = DocumentLoader()
    documents = loader.load_single_document(file_path)
    return documents[0].page_content if documents else ""

def preprocess_document(document: str) -> str:
    """Legacy function - preprocesses document content"""
    return document.strip()

def load_and_preprocess_documents(directory: str) -> List[str]:
    """Legacy function - returns preprocessed document contents as strings"""
    loader = DocumentLoader()
    documents = loader.load_and_process_documents(directory)
    return [doc.page_content for doc in documents]