#!/usr/bin/env python3
"""
RAG Pipeline Debugging Script

This script provides specific debugging tools for your RAG chatbot.
Use this to test and debug individual components of your pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot.rag_pipeline import RAGPipeline
from utils.debug_logging import setup_advanced_logging, debug_decorator, log_performance
import logging

# Setup advanced logging
setup_advanced_logging(log_level="DEBUG")
logger = logging.getLogger(__name__)

def debug_document_loading():
    """Debug document loading process"""
    print("\n=== DEBUGGING DOCUMENT LOADING ===")
    
    documents_path = "data/documents"
    
    # Check if documents directory exists
    if not os.path.exists(documents_path):
        print(f"❌ Documents directory not found: {documents_path}")
        return False
    
    # List all files in documents directory
    print(f"📁 Documents directory: {documents_path}")
    for root, dirs, files in os.walk(documents_path):
        level = root.replace(documents_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")
    
    return True

def debug_embeddings():
    """Debug embeddings model loading"""
    print("\n=== DEBUGGING EMBEDDINGS ===")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("🔄 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Test embeddings with sample text
        test_text = "What are baby milestones?"
        print(f"📝 Testing with: '{test_text}'")
        
        embedding_vector = embeddings.embed_query(test_text)
        print(f"✅ Embedding generated - Dimension: {len(embedding_vector)}")
        print(f"📊 First 5 values: {embedding_vector[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False

def debug_vector_store():
    """Debug vector store operations"""
    print("\n=== DEBUGGING VECTOR STORE ===")
    
    vector_store_path = "data/vector_store"
    
    # Check if vector store exists
    faiss_index = os.path.join(vector_store_path, "index.faiss")
    faiss_pkl = os.path.join(vector_store_path, "index.pkl")
    
    print(f"📁 Vector store path: {vector_store_path}")
    print(f"📄 FAISS index exists: {os.path.exists(faiss_index)}")
    print(f"📄 FAISS pkl exists: {os.path.exists(faiss_pkl)}")
    
    if os.path.exists(faiss_index):
        file_size = os.path.getsize(faiss_index)
        print(f"📊 FAISS index size: {file_size} bytes")

def debug_full_pipeline():
    """Debug the complete RAG pipeline"""
    print("\n=== DEBUGGING FULL RAG PIPELINE ===")
    
    try:
        # Initialize pipeline
        print("🚀 Initializing RAG Pipeline...")
        pipeline = RAGPipeline()
        
        # Test query
        test_queries = [
            "What are normal milestones for a 6-month-old baby?",
            "How should I handle a baby's fever?",
            "What are signs of developmental delays?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Query: {query}")
            
            try:
                response = pipeline.generate_response(query)
                print(f"Response length: {len(response)} characters")
                print(f"Response preview: {response[:200]}...")
                print("✅ Query processed successfully")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_debug_session():
    """Start an interactive debugging session"""
    print("\n=== INTERACTIVE DEBUG SESSION ===")
    
    try:
        pipeline = RAGPipeline()
        
        print("🤖 RAG Pipeline loaded. Type 'quit' to exit.")
        print("Commands:")
        print("  - Ask any question to test the pipeline")
        print("  - 'docs' to see similar documents for a query")
        print("  - 'debug' to enter Python debugger")
        print("  - 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n💬 Enter query: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'debug':
                    import pdb; pdb.set_trace()
                elif user_input.lower() == 'docs':
                    query = input("Enter query for document search: ")
                    docs = pipeline.get_similar_documents(query, k=3)
                    print(f"Found {len(docs)} similar documents:")
                    for i, doc in enumerate(docs, 1):
                        print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
                        print(f"   Preview: {doc.page_content[:100]}...")
                elif user_input:
                    response = pipeline.generate_response(user_input)
                    print(f"🤖 Response: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to start interactive session: {e}")

def main():
    """Main debugging function"""
    print("🔧 RAG Pipeline Debugging Tool")
    print("=" * 50)
    
    # Run debugging steps
    steps = [
        ("Document Loading", debug_document_loading),
        ("Embeddings", debug_embeddings),
        ("Vector Store", debug_vector_store),
        ("Full Pipeline", debug_full_pipeline)
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 DEBUGGING SUMMARY")
    print("=" * 50)
    
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{step_name}: {status}")
    
    # Start interactive session if all tests pass
    if all(results.values()):
        print("\n🎉 All tests passed! Starting interactive session...")
        interactive_debug_session()
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before using the pipeline.")

if __name__ == "__main__":
    main()
