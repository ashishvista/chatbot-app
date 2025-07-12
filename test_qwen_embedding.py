#!/usr/bin/env python3
"""
Test Qwen3-Embedding-8B Model Loading

This script tests if the Qwen3-Embedding-8B model can be loaded successfully
and provides basic functionality for embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_embedding():
    """Test Qwen3-Embedding-8B model loading and basic functionality"""
    print("🧪 Testing Qwen3-Embedding-8B Model Loading")
    print("=" * 50)
    
    try:
        # Test model loading
        print(f"📊 Loading embedding model: {settings.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Model loaded successfully!")
        
        # Test basic embedding
        test_text = "What are the symptoms of common childhood illnesses?"
        print(f"\n🔍 Testing embedding generation for: '{test_text}'")
        
        embedding_vector = embeddings.embed_query(test_text)
        print(f"✅ Generated embedding vector with {len(embedding_vector)} dimensions")
        print(f"📊 First 5 dimensions: {embedding_vector[:5]}")
        
        # Test batch embeddings
        test_documents = [
            "Fever in children can be concerning for parents.",
            "Regular checkups are important for child development.",
            "Vaccination schedules help prevent serious diseases."
        ]
        
        print(f"\n📚 Testing batch embeddings for {len(test_documents)} documents")
        document_embeddings = embeddings.embed_documents(test_documents)
        print(f"✅ Generated {len(document_embeddings)} document embeddings")
        
        # Test similarity (basic check)
        import numpy as np
        sim_score = np.dot(embedding_vector, document_embeddings[0])
        print(f"🔗 Similarity score between query and first document: {sim_score:.4f}")
        
        print("\n🎉 All tests passed! Qwen3-Embedding-8B is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_qwen_embedding()
