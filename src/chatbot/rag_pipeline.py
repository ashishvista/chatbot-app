"""
RAG (Retrieval-Augmented Generation) Pipeline Implementation

This module implements the core RAG functionality that powers the chatbot.
RAG combines document retrieval with text generation to provide contextually accurate responses.

How RAG Works:
1. INDEXING PHASE:
   - Documents are loaded from the specified directory
   - Text is split into manageable chunks (1000 chars with 200 overlap)
   - Each chunk is converted to embeddings using sentence-transformers
   - Embeddings are stored in a FAISS vector database for fast similarity search

2. QUERY PHASE:
   - User query is converted to embeddings
   - Similar document chunks are retrieved from the vector store
   - Retrieved context + user query are fed to the language model
   - Model generates a response based on the provided context

Key Components:
- HuggingFace Transformers: For embeddings and text generation
- FAISS: Vector database for efficient similarity search
- LangChain: Framework for chaining retrieval and generation
- Gradio: Web interface for user interaction

The pipeline is designed to be:
- Scalable: Can handle large document collections
- Efficient: Fast retrieval using vector similarity
- Accurate: Context-aware responses based on your documents
- Flexible: Supports multiple document formats and models
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document

# Import using absolute imports from the src directory
from config.settings import settings
from chatbot.qwen_llm import QwenLLM

# Setup logging to track pipeline operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Advanced RAG Pipeline with Hugging Face models and LangChain
    
    This class orchestrates the entire RAG process:
    1. Document Loading & Processing
    2. Vector Store Creation & Management
    3. Language Model Setup
    4. Query Processing & Response Generation
    
    Architecture:
    - Embeddings: sentence-transformers for semantic understanding
    - Vector Store: FAISS for fast similarity search
    - LLM: HuggingFace transformers for text generation
    - Chain: LangChain RetrievalQA for coordinating retrieval + generation
    """
    
    def __init__(self, model_name: Optional[str] = None, documents_path: Optional[str] = None):
        """
        Initialize the RAG pipeline with configurable models and paths
        
        Args:
            model_name: HuggingFace model name for text generation
            documents_path: Path to directory containing documents to index
        """
        # Configuration - use provided values or defaults from settings
        self.model_name = model_name or settings.MODEL_NAME
        self.documents_path = documents_path or settings.DOCUMENTS_PATH
        self.vector_store_path = settings.VECTOR_STORE_PATH
        
        # Initialize components (will be populated during setup)
        self.embeddings = None          # For converting text to vectors
        self.vectorstore = None         # FAISS database for document chunks
        self.llm = None                # Language model for generation
        self.qa_chain = None           # LangChain QA chain
        self.device = self._get_device()  # CPU or CUDA
        
        # Setup the complete pipeline
        self._initialize_pipeline()
    
    def _get_device(self) -> str:
        """
        Determine the best device to use for model inference
        
        Returns:
            "cpu" for stable operation
            
        This affects performance significantly:
        - CPU: Slower but works on any machine and is more stable
        """
        logger.info("💻 Using CPU for stable operation")
        return "cpu"
    
    def _initialize_pipeline(self):
        """
        Initialize the complete RAG pipeline in the correct order
        
        Pipeline Initialization Steps:
        1. Setup embeddings model (for vectorizing text)
        2. Load and process documents (create vector database)
        3. Setup language model (for generating responses)
        4. Create QA chain (connects retrieval + generation)
        
        Error handling ensures graceful degradation if any step fails
        """
        logger.info("🚀 Initializing RAG Pipeline...")
        
        try:
            # Step 1: Setup embeddings for document vectorization
            self._setup_embeddings()
            
            # Step 2: Load documents and create searchable vector database
            self._load_and_process_documents()
            
            # Step 3: Setup language model for response generation
            self._setup_llm()
            
            # Step 4: Create the QA chain that ties everything together
            self._setup_qa_chain()
            
            logger.info("✅ RAG Pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Pipeline: {str(e)}")
            raise
    
    def _setup_embeddings(self):
        """
        Setup Hugging Face embeddings model for text vectorization
        
        How embeddings work:
        1. Convert text into high-dimensional vectors (embeddings)
        2. Similar texts have similar vectors (semantic similarity)
        3. Enable fast similarity search using vector operations
        
        Model: sentence-transformers/all-MiniLM-L6-v2
        - Optimized for semantic similarity tasks
        - Good balance of speed vs quality
        - 384-dimensional embeddings
        """
        logger.info("📊 Loading embeddings model...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': self.device},  # Use GPU if available
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
    
    def _load_and_process_documents(self):
        """
        Load documents and create vector store for semantic search
        
        Document Processing Pipeline:
        1. Check if vector store already exists (skip if found)
        2. Load documents from the specified directory (.txt, .md files)
        3. Split documents into chunks (1000 chars with 200 overlap)
        4. Convert chunks to embeddings using the embedding model
        5. Store embeddings in FAISS vector database
        6. Save vector store to disk for future use
        
        Why chunking is important:
        - Large documents exceed model context limits
        - Smaller chunks provide more focused context
        - Overlap ensures important information isn't lost at boundaries
        """
        logger.info("📚 Loading documents...")
        
        # Check if vector store already exists to avoid reprocessing
        if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            logger.info("📂 Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return
        
        # Load documents from directory
        if not os.path.exists(self.documents_path):
            logger.warning(f"⚠️ Documents directory not found: {self.documents_path}")
            self._create_empty_vectorstore()
            return
        
        try:
            # Load documents with multiple file types
            documents = []
            
            # Load text files (.txt)
            txt_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(txt_loader.load())
            
            # Load markdown files (.md)
            md_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(md_loader.load())
            
            if not documents:
                logger.warning("⚠️ No documents found!")
                self._create_empty_vectorstore()
                return
            
            # Split documents into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,      # 1000 characters per chunk
                chunk_overlap=settings.CHUNK_OVERLAP, # 200 character overlap
                length_function=len,
                separators=["\n\n", "\n", " ", ""]   # Split on paragraphs, then lines, then words
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store from document chunks
            logger.info(f"🔍 Creating vector store with {len(splits)} document chunks...")
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            # Save vector store to disk for future use
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vectorstore.save_local(self.vector_store_path)
            
            logger.info(f"✅ Loaded {len(documents)} documents, {len(splits)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error loading documents: {str(e)}")
            self._create_empty_vectorstore()
    
    def _create_empty_vectorstore(self):
        """
        Create an empty vector store with placeholder content
        
        This fallback ensures the system works even without documents.
        Users will see a helpful message explaining how to add documents.
        """
        placeholder_doc = Document(
            page_content="No documents have been loaded yet. Please add documents to the data/documents folder and restart the application.",
            metadata={"source": "system"}
        )
        self.vectorstore = FAISS.from_documents([placeholder_doc], self.embeddings)
    
    def _setup_llm(self):
        """
        Setup language model for text generation using Qwen models
        
        Model Configuration:
        - Uses Qwen3-8B from Hugging Face transformers
        - Supports advanced text generation capabilities
        - Configurable generation parameters
        
        Generation Parameters:
        - max_new_tokens: Limit response length
        - temperature: Control randomness (0.3 = focused for Q&A)
        """
        logger.info(f"🤖 Setting up language model: {self.model_name}")
        logger.info(f"🔧 Model provider: {settings.MODEL_PROVIDER}")
        
        try:
            # Use Qwen models via QwenLLM wrapper
            logger.info("🚀 Using Qwen models via QwenLLM")
            self.llm = QwenLLM.from_settings(
                settings,
                system_prompt="You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely. If the context doesn't contain relevant information, say so clearly."
            )
            
            # Verify model initialization
            model_info = self.llm.get_model_info()
            if 'error' in model_info:
                logger.error(f"❌ Qwen model error: {model_info['error']}")
                raise Exception(f"Qwen model setup failed: {model_info['error']}")
            else:
                logger.info(f"✅ Qwen model '{self.model_name}' loaded successfully")
                logger.info(f"📊 Model info: {model_info}")
            
            logger.info(f"✅ Language model setup completed")
            
        except Exception as e:
            logger.error(f"❌ Error setting up language model: {str(e)}")
            logger.info("💡 Tip: Make sure the model is compatible and accessible")
            raise
    
    def _setup_qa_chain(self):
        """
        Setup the QA retrieval chain that connects retrieval and generation
        
        How the QA Chain Works:
        1. User asks a question
        2. Question is converted to embeddings
        3. Similar document chunks are retrieved from vector store
        4. Retrieved context + question are sent to language model
        5. Model generates response based on the context
        
        Chain Type: "stuff"
        - Concatenates all retrieved documents as context
        - Simple and effective for most use cases
        - Alternative: "map_reduce" for very large contexts
        """
        logger.info("🔗 Setting up QA chain...")
        
        # Create retriever from vector store
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",                    # Use cosine similarity
            search_kwargs={"k": settings.RETRIEVAL_K}   # Retrieve top K documents
        )
        
        # Custom prompt template for better DialoGPT compatibility
        from langchain.prompts import PromptTemplate
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        
        qa_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain that ties everything together
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,                               # Language model for generation
            chain_type="stuff",                         # How to combine retrieved docs
            retriever=retriever,                        # Document retriever
            return_source_documents=True,               # Include sources in response
            verbose=settings.DEBUG_MODE,                # Enable detailed logging
            chain_type_kwargs={"prompt": qa_prompt}     # Use custom prompt
        )
    
    def generate_response(self, query: str, conversation_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Generate response for user query using the complete RAG pipeline
        
        RAG Response Generation Process:
        1. Validate pipeline is initialized
        2. Convert user query to embeddings
        3. Search vector store for similar document chunks
        4. Retrieve top K most relevant chunks
        5. Combine retrieved context with user query (and optionally conversation history)
        6. Send to language model for response generation
        7. Format response with source citations
        8. Return complete response to user
        
        Args:
            query: User's question or message
            conversation_history: Optional list of (question, answer) tuples for context
            
        Returns:
            Generated response with source citations
        """
        if not self.qa_chain:
            logger.error("❌ QA chain not initialized")
            return "Sorry, the system is not ready yet. Please try again in a moment."
        
        try:
            logger.info(f"🔍 Processing query: {query[:100]}...")
            
            # Format query with conversation history if enabled and provided
            formatted_query = query
            if (settings.USE_CONVERSATION_HISTORY and 
                conversation_history and 
                len(conversation_history) > 0):
                # Include limited history to avoid token limit issues
                recent_history = conversation_history[-settings.MAX_HISTORY_TURNS:]
                history_context = "\n".join([
                    f"Previous Q: {q}\nPrevious A: {a}" 
                    for q, a in recent_history
                ])
                formatted_query = f"Conversation History:\n{history_context}\n\nCurrent Question: {query}"
                logger.info(f"📝 Including {len(recent_history)} previous conversation turns")
            
            # Use the QA chain to get response with sources
            result = self.qa_chain.invoke({"query": formatted_query})
            
            # Extract response and sources
            response = result.get("result", "").strip()
            sources = result.get("source_documents", [])
            
            # Format response with source citations
            formatted_response = self._format_response(response, sources)
            
            logger.info(f"✅ Generated response ({len(response)} chars)")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _format_response(self, response: str, sources: List[Document]) -> str:
        """
        Format the response with source information for transparency
        
        This adds source citations to responses so users know:
        - Which documents were used to generate the answer
        - How to verify or find more information
        - Whether the response is based on their documents or general knowledge
        
        Args:
            response: Generated text from the language model
            sources: List of source documents used for context
            
        Returns:
            Formatted response with source citations
        """
        # Clean up the response text
        response = response.strip()
        
        # Add source information if documents were retrieved
        if sources and len(sources) > 0:
            response += "\n\n📚 **Sources:**\n"
            
            seen_sources = set()
            for i, doc in enumerate(sources[:3], 1):  # Limit to top 3 sources
                source = doc.metadata.get('source', 'Unknown')
                if source not in seen_sources:
                    seen_sources.add(source)
                    filename = os.path.basename(source) if source != 'Unknown' else 'System'
                    response += f"{i}. {filename}\n"
        
        return response
    
    def add_documents(self, documents_path: str):
        """Add new documents to the vectorstore"""
        try:
            loader = DirectoryLoader(
                documents_path, 
                glob="**/*.{txt,md}", 
                loader_cls=TextLoader
            )
            new_docs = loader.load()
            
            if new_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP
                )
                splits = text_splitter.split_documents(new_docs)
                
                # Add to existing vectorstore
                self.vectorstore.add_documents(splits)
                
                # Save updated vectorstore
                self.vectorstore.save_local(self.vector_store_path)
                
                logger.info(f"✅ Added {len(new_docs)} new documents")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return False
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve similar documents for a query"""
        if not self.vectorstore:
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {str(e)}")
            return []