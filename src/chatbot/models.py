"""
Chatbot Model Module for Pediatric RAG System

This module contains the core chatbot model that integrates:
1. Large Language Models (LLMs) from Hugging Face Transformers
2. Vector Store retrieval for Retrieval-Augmented Generation (RAG)
3. Pediatric domain-specific knowledge base

The ChatbotModel class handles:
- Loading and managing pre-trained language models
- Interfacing with vector stores for document retrieval
- Combining retrieved context with user queries
- Generating contextually relevant responses for pediatric healthcare questions

Key Components:
- Transformers library for LLM inference
- LangChain for RAG pipeline orchestration
- FAISS vector store for efficient similarity search
- HuggingFace embeddings for semantic understanding
"""

# Core ML libraries for language model handling
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face transformers for LLM loading
from langchain.chains import RetrievalQA                      # LangChain for retrieval-augmented generation
from langchain.embeddings import HuggingFaceEmbeddings        # Embeddings for semantic similarity
from langchain.vectorstores import FAISS                      # Vector database for document retrieval
import os                                                      # Operating system interface

class ChatbotModel:
    """
    Enhanced Chatbot Model for Pediatric Healthcare RAG System
    
    This class provides a complete RAG (Retrieval-Augmented Generation) implementation
    specifically designed for pediatric healthcare applications. It combines:
    
    1. Document Retrieval: Uses FAISS vector store to find relevant pediatric documents
    2. Context Integration: Combines retrieved documents with user queries
    3. Response Generation: Uses transformer models to generate accurate responses
    
    The model is designed to handle pediatric-specific queries like:
    - Baby milestone questions
    - Common illness inquiries
    - Feeding and nutrition guidance
    - Emergency situation recognition
    - Development concerns
    """
    
    def __init__(self, model_name: str, vector_store_path: str):
        """
        Initialize the ChatbotModel with specified language model and vector store
        
        Args:
            model_name (str): Name of the Hugging Face model to use for generation
                             Examples: "microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"
            vector_store_path (str): Path to the pre-built FAISS vector store containing
                                   pediatric documents and knowledge base
        
        The initialization process:
        1. Loads the tokenizer for text preprocessing
        2. Loads the language model for response generation
        3. Loads the vector store for document retrieval
        """
        # Load tokenizer for converting text to tokens and vice versa
        # Tokenizers handle the conversion between human-readable text and model-readable tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the pre-trained language model for text generation
        # This model will generate responses based on the provided context and query
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Load the vector store containing embedded pediatric documents
        # This enables semantic search over the pediatric knowledge base
        # The embeddings model should match the one used to create the vector store
        self.vector_store = FAISS.load_local(
            vector_store_path, 
            HuggingFaceEmbeddings(model_name)
        )

    def generate_response(self, query: str) -> str:
        """
        Generate a response using the language model
        
        This method performs direct text generation without retrieval.
        It's used internally by retrieve_and_generate() but can also be used
        for general-purpose text generation.
        
        Args:
            query (str): Input text/question to generate a response for
            
        Returns:
            str: Generated response text
            
        Process:
        1. Tokenize the input query into model-readable format
        2. Generate response tokens using the language model
        3. Decode tokens back to human-readable text
        4. Return the cleaned response
        """
        # Convert input text to tokens (numbers) that the model can process
        # return_tensors='pt' returns PyTorch tensors
        inputs = self.tokenizer.encode(query, return_tensors='pt')
        
        # Generate response tokens using the language model
        # max_length: Maximum number of tokens in the response
        # num_return_sequences: Number of different responses to generate
        outputs = self.model.generate(
            inputs, 
            max_length=150, 
            num_return_sequences=1
        )
        
        # Convert generated tokens back to human-readable text
        # skip_special_tokens=True removes special model tokens like [PAD], [SEP], etc.
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

    def retrieve_and_generate(self, query: str) -> str:
        """
        Perform Retrieval-Augmented Generation (RAG) for pediatric queries
        
        This is the main method for answering pediatric healthcare questions.
        It implements the complete RAG pipeline:
        
        1. RETRIEVAL: Search vector store for relevant pediatric documents
        2. AUGMENTATION: Combine retrieved context with user query  
        3. GENERATION: Generate response using enhanced context
        
        Args:
            query (str): User's pediatric healthcare question
            
        Returns:
            str: Contextually informed response based on pediatric knowledge base
            
        Example queries this method handles well:
        - "When should my 6-month-old start solid foods?"
        - "What are signs of fever in newborns?"
        - "Is my toddler's speech development normal?"
        """
        # Step 1: RETRIEVAL - Find relevant documents from pediatric knowledge base
        # Create a retriever interface to the vector store
        retriever = self.vector_store.as_retriever()
        
        # Search for documents semantically similar to the user's query
        # This uses embeddings to find the most relevant pediatric information
        docs = retriever.get_relevant_documents(query)
        
        # Step 2: AUGMENTATION - Combine retrieved context with user query
        # Extract text content from retrieved documents
        # This provides the model with relevant pediatric knowledge to inform its response
        context = " ".join([doc.page_content for doc in docs])
        
        # Create an enhanced query that includes both context and original question
        # Format: [CONTEXT FROM DOCUMENTS]\n\n[USER QUESTION]
        # This gives the model access to specific pediatric information
        full_query = f"{context}\n\n{query}"
        
        # Step 3: GENERATION - Generate response using the enhanced query
        # The model now has access to relevant pediatric documents to inform its answer
        return self.generate_response(full_query)

# Example usage and integration:
# 
# # Initialize the chatbot model with a specific language model and vector store
# model = ChatbotModel(
#     model_name="microsoft/DialoGPT-medium",  # Choose appropriate conversational model
#     vector_store_path="data/vector_store"    # Path to pre-built pediatric knowledge base
# )
# 
# # Ask pediatric-specific questions
# response = model.retrieve_and_generate("What should I do if my baby has a fever?")
# print(response)
# 
# # The system will:
# 1. Search pediatric documents for fever-related information
# 2. Combine relevant medical guidance with the question
# 3. Generate a response based on both the query and retrieved knowledge