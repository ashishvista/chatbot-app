#!/usr/bin/env python3
"""
Main application entry point for the Domain-Specific RAG Chatbot

This module serves as the main entry point for the chatbot application.
It initializes the system and launches the web interface.

Application Flow:
1. Setup logging and configuration
2. Initialize the RAG pipeline (models, vector store, etc.)
3. Launch the Gradio web interface
4. Handle graceful shutdown

The application supports both simple and advanced interfaces:
- Simple: ChatGPT-like interface for basic usage
- Advanced: Multi-tab interface with configuration options
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path for relative imports
sys.path.append(str(Path(__file__).parent))

from ui.gradio_interface import launch_simple_interface, launch_advanced_interface
from config.settings import settings

# Setup comprehensive logging for the entire application
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main application function
    
    This function:
    1. Displays startup information and configuration
    2. Validates system requirements
    3. Launches the web interface
    4. Handles errors and graceful shutdown
    """
    logger.info("🚀 Starting Domain-Specific RAG Chatbot...")
    
    # Display current configuration for debugging
    logger.info(f"📊 Model: {settings.MODEL_NAME}")
    logger.info(f"📚 Documents path: {settings.DOCUMENTS_PATH}")
    logger.info(f"🌐 Starting server on {settings.GRADIO_HOST}:{settings.GRADIO_PORT}")
    
    try:
        # Launch the advanced interface by default (more features)
        # Users can modify this to launch_simple_interface() if preferred
        launch_advanced_interface()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main()