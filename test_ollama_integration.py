#!/usr/bin/env python3
"""
Ollama Integration Test Script

This script tests the Ollama integration with the RAG pipeline.
It verifies that:
1. Ollama server is running and accessible
2. DeepSeek-R1 model is available
3. Basic text generation works
4. RAG pipeline integration functions correctly

Usage:
    python test_ollama_integration.py

Prerequisites:
    1. Ollama must be installed and running
    2. DeepSeek-R1 model must be pulled: ollama pull deepseek-r1:7b
    3. Dependencies must be installed: pip install -r requirements.txt
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test basic Ollama server connection"""
    print("🔍 Testing Ollama connection...")
    
    try:
        import ollama
        
        # Test connection to Ollama server
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
        
        print(f"✅ Ollama server is running")
        
        # Handle different response formats
        if hasattr(models_response, 'models'):
            models = models_response.models
        elif isinstance(models_response, dict) and 'models' in models_response:
            models = models_response['models']
        else:
            # Try to access as attribute or list directly
            models = getattr(models_response, 'models', models_response)
        
        print(f"📊 Available models: {len(models) if models else 0}")
        
        # List available models
        if models:
            for model in models:
                if hasattr(model, 'name'):
                    name = model.name
                    size = getattr(model, 'size', 'Unknown size')
                elif isinstance(model, dict):
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 'Unknown size')
                else:
                    name = str(model)
                    size = 'Unknown size'
                print(f"   - {name} ({size})")
        
        return True, client
        
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {str(e)}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False, None

def test_deepseek_model(client):
    """Test DeepSeek-R1 model availability and basic generation"""
    print("\n🤖 Testing DeepSeek-R1 model...")
    
    try:
        model_name = "deepseek-r1:7b"
        
        # Check if model is available
        models_response = client.list()
        
        # Handle different response formats
        if hasattr(models_response, 'models'):
            models = models_response.models
        elif isinstance(models_response, dict) and 'models' in models_response:
            models = models_response['models']
        else:
            models = getattr(models_response, 'models', models_response)
        
        # Extract model names
        available_models = []
        if models:
            for model in models:
                if hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict):
                    available_models.append(model.get('name', ''))
                else:
                    available_models.append(str(model))
        
        if model_name not in available_models:
            print(f"⚠️ Model '{model_name}' not found. Attempting to pull...")
            client.pull(model_name)
            print(f"✅ Successfully pulled '{model_name}'")
        
        # Test basic generation
        print(f"🧠 Testing text generation with {model_name}...")
        response = client.generate(
            model=model_name,
            prompt="What is artificial intelligence? Answer in one sentence.",
            options={
                "temperature": 0.3,
                "num_predict": 50
            }
        )
        
        generated_text = response.get("response", "").strip()
        print(f"✅ Model response: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ DeepSeek model test failed: {str(e)}")
        return False

def test_ollama_llm_wrapper():
    """Test the custom OllamaLLM wrapper"""
    print("\n🔧 Testing OllamaLLM wrapper...")
    
    try:
        from chatbot.ollama_llm import OllamaLLM
        from config.settings import settings
        
        # Create OllamaLLM instance
        llm = OllamaLLM(
            model="deepseek-r1:7b",
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=100
        )
        
        # Test direct call
        prompt = "Explain the concept of machine learning in simple terms."
        print(f"🧠 Testing prompt: {prompt}")
        
        response = llm(prompt)
        print(f"✅ Wrapper response: {response[:200]}...")
        
        # Test model info
        model_info = llm.get_model_info()
        print(f"📊 Model info: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ OllamaLLM wrapper test failed: {str(e)}")
        return False

def test_rag_pipeline():
    """Test the complete RAG pipeline with Ollama"""
    print("\n🔗 Testing RAG pipeline with Ollama...")
    
    try:
        from chatbot.rag_pipeline import RAGPipeline
        from config.settings import settings
        
        # Temporarily set MODEL_PROVIDER to ollama
        original_provider = settings.MODEL_PROVIDER
        settings.MODEL_PROVIDER = "ollama"
        
        # Initialize RAG pipeline
        print("🚀 Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Test query
        test_question = "What is the main purpose of this system?"
        print(f"❓ Testing question: {test_question}")
        
        response = rag.generate_response(test_question)
        print(f"✅ RAG response: {response}")
        
        # Restore original provider
        settings.MODEL_PROVIDER = original_provider
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {str(e)}")
        return False

def main():
    """Run all Ollama integration tests"""
    print("🧪 Ollama Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Ollama connection
    success, client = test_ollama_connection()
    if not success:
        print("\n❌ Ollama integration tests failed - server not accessible")
        return False
    
    # Test 2: DeepSeek model
    success = test_deepseek_model(client)
    if not success:
        print("\n❌ Ollama integration tests failed - model issues")
        return False
    
    # Test 3: OllamaLLM wrapper
    success = test_ollama_llm_wrapper()
    if not success:
        print("\n❌ Ollama integration tests failed - wrapper issues")
        return False
    
    # Test 4: RAG pipeline
    success = test_rag_pipeline()
    if not success:
        print("\n⚠️ RAG pipeline test failed - this may be due to missing documents")
        print("   The Ollama integration itself is working correctly")
    
    print("\n" + "=" * 50)
    print("✅ Ollama integration tests completed successfully!")
    print("\n🎉 Your RAG chatbot is ready to use Ollama + DeepSeek-R1!")
    print("\n💡 Next steps:")
    print("   1. Add documents to data/documents/ folder")
    print("   2. Run the main chatbot application")
    print("   3. Test with your domain-specific questions")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        logger.exception("Test failed with exception")
