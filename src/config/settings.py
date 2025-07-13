"""
Configuration settings for the chatbot application

This module defines all configuration settings used throughout the RAG chatbot application.
It uses Pydantic for settings validation and environment variable management.

Key Features:
- Environment variable support via .env files
- Type validation using Pydantic
- Automatic directory creation for data storage
- Centralized configuration management

Settings Categories:
- Model Configuration: LLM and embedding model settings
- API Keys: Integration with external services
- Gradio UI: Web interface configuration
- Document Processing: RAG pipeline parameters
- System: Device and debug settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables from .env file if it exists
# This allows for easy configuration without hardcoding values
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings with validation
    
    This class defines all configuration parameters for the RAG chatbot.
    Settings can be overridden via environment variables or .env file.
    
    How it works:
    1. Default values are provided for all settings
    2. Environment variables override defaults if present
    3. Pydantic validates types and constraints
    4. Settings are accessible globally via the settings instance
    """
    
    # === Model Configuration ===
    # These settings control which AI models are used
    
    # Model provider: "huggingface" (now using Qwen models exclusively)
    MODEL_PROVIDER: str = "huggingface"
    
    # Primary language model for text generation
    # Using Qwen3-8B for better performance and capabilities
    MODEL_NAME: str = "Qwen/Qwen3-8B"
    
    # Embedding model for document vectorization
    # Using Qwen3-Embedding-8B for optimal compatibility with Qwen3-8B model
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-4B"
    
    # Directory to cache downloaded models
    # Use relative path from project root, not from src directory
    MODEL_PATH: str = "models"
    
    # === API Keys ===
    # API keys for external services (optional)
    HUGGINGFACE_API_KEY: Optional[str] = None  # For Hugging Face Hub access (optional for public models)
    OPENAI_API_KEY: Optional[str] = None  # For OpenAI compatibility (optional)
    
    # === Gradio UI Configuration ===
    # These settings control the web interface appearance and behavior
    GRADIO_TITLE: str = "🤖 Domain-Specific RAG Chatbot"
    GRADIO_DESCRIPTION: str = "Ask questions about your documents using Retrieval-Augmented Generation"
    GRADIO_PORT: int = 7860  # Port for the web interface
    GRADIO_HOST: str = "0.0.0.0"  # Host address (0.0.0.0 allows external access)
    
    # === Document Processing Configuration ===
    # These settings control how documents are processed and stored
    
    # Directory containing source documents to be indexed
    # Use relative path from project root, not from src directory
    DOCUMENTS_PATH: str = str(Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent)).joinpath("data/documents").resolve())
    
    # Directory to store the vector database
    VECTOR_STORE_PATH: str = str(Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent)).joinpath("data/vector_store").resolve())
    
    # Size of text chunks when splitting documents
    # Increased for better context with larger models
    CHUNK_SIZE: int = 800
    
    # Overlap between consecutive chunks to maintain continuity
    CHUNK_OVERLAP: int = 100
    
    # === RAG Pipeline Configuration ===
    # These settings control the retrieval and generation behavior
    
    # Number of similar documents to retrieve for each query
    # Increased for better context with larger models
    RETRIEVAL_K: int = 4
    
    # Maximum number of new tokens to generate in responses
    MAX_NEW_TOKENS: int = 512  # Increased for more detailed responses
    
    # Temperature for text generation (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE: float = 0.3   # Lower temperature for more focused medical answers
    
    # Top-p sampling parameter for nucleus sampling
    TOP_P: float = 0.9
    
    # === Conversation History Configuration ===
    # Control whether conversation history is used in RAG context
    USE_CONVERSATION_HISTORY: bool = True  # Enable conversation context
    MAX_HISTORY_TURNS: int = 3  # Keep more history with larger context window
    
    # === System Configuration ===
    DEBUG_MODE: bool = False  # Disable detailed logging for cleaner output
    DEVICE: str = "cpu"  # Force CPU usage for stability
    
    class Config:
        """Pydantic configuration class"""
        env_file = ".env"  # Load environment variables from .env file
        case_sensitive = True  # Environment variable names are case-sensitive

# Global settings instance - this is imported throughout the application
# It automatically loads configuration from environment variables and .env file
settings = Settings()

# Create necessary directories if they don't exist
# This ensures the application can store documents and vector data
Path(settings.DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.MODEL_PATH).mkdir(parents=True, exist_ok=True)