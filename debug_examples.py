#!/usr/bin/env python3
"""
Python Debugging Examples and Techniques

This file demonstrates various debugging approaches you can use
in your RAG chatbot project or any Python application.
"""

import pdb
import logging
import traceback
from typing import List, Dict, Any

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DebugExample:
    """Example class to demonstrate debugging techniques"""
    
    def __init__(self):
        self.data = []
        logger.debug("DebugExample initialized")
    
    def process_data(self, items: List[str]) -> Dict[str, Any]:
        """
        Example method with various debugging techniques
        """
        logger.info(f"Processing {len(items)} items")
        
        # 1. PRINT DEBUGGING - Simple but effective
        print(f"DEBUG: Input items: {items}")
        
        # 2. ASSERTION DEBUGGING - Validate assumptions
        assert isinstance(items, list), f"Expected list, got {type(items)}"
        assert len(items) > 0, "Items list cannot be empty"
        
        result = {"processed": [], "errors": []}
        
        for i, item in enumerate(items):
            try:
                # 3. PDB BREAKPOINT - Interactive debugging
                # Uncomment the next line to start interactive debugging
                # pdb.set_trace()  # Execution will pause here
                
                # 4. CONDITIONAL BREAKPOINT
                if item.startswith("error"):
                    pdb.set_trace()  # Only break on error items
                
                processed_item = self._transform_item(item, i)
                result["processed"].append(processed_item)
                
                # 5. DETAILED LOGGING
                logger.debug(f"Processed item {i}: {item} -> {processed_item}")
                
            except Exception as e:
                # 6. EXCEPTION DEBUGGING
                logger.error(f"Error processing item {i}: {item}")
                logger.error(f"Exception: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append({"item": item, "error": str(e)})
        
        # 7. FINAL STATE INSPECTION
        logger.info(f"Processing complete. Processed: {len(result['processed'])}, Errors: {len(result['errors'])}")
        print(f"DEBUG: Final result keys: {result.keys()}")
        
        return result
    
    def _transform_item(self, item: str, index: int) -> str:
        """Transform an item with debugging"""
        
        # Simulate some processing logic
        if item == "error_item":
            raise ValueError(f"Cannot process error_item at index {index}")
        
        # 8. VARIABLE INSPECTION
        transformed = f"processed_{item}_{index}"
        logger.debug(f"Transform: {item} -> {transformed}")
        
        return transformed

def demonstrate_debugging_techniques():
    """
    Demonstrate various debugging approaches
    """
    print("=== Python Debugging Techniques Demo ===\n")
    
    # 1. Basic debugging with prints
    print("1. PRINT DEBUGGING:")
    test_data = ["item1", "item2", "error_item", "item3"]
    print(f"Input data: {test_data}")
    
    # 2. Using logging for structured debugging
    print("\n2. LOGGING DEBUGGING:")
    logger.info("Starting debugging demonstration")
    
    # 3. Try-except with detailed error info
    print("\n3. EXCEPTION HANDLING:")
    try:
        debug_example = DebugExample()
        result = debug_example.process_data(test_data)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback:\n{traceback.format_exc()}")

def debug_rag_pipeline_example():
    """
    Example of debugging RAG pipeline components
    """
    print("\n=== RAG Pipeline Debugging Example ===\n")
    
    # Simulate RAG pipeline debugging
    query = "What are baby milestones?"
    
    # 1. Input validation debugging
    print(f"DEBUG: Query received: '{query}'")
    print(f"DEBUG: Query type: {type(query)}")
    print(f"DEBUG: Query length: {len(query)}")
    print(f"DEBUG: Is query empty: {not query.strip()}")
    
    # 2. Simulate document retrieval debugging
    retrieved_docs = ["doc1", "doc2", "doc3"]
    print(f"DEBUG: Retrieved {len(retrieved_docs)} documents")
    print(f"DEBUG: Document IDs: {retrieved_docs}")
    
    # 3. Simulate model response debugging
    model_response = "Babies typically reach milestones..."
    print(f"DEBUG: Model response length: {len(model_response)}")
    print(f"DEBUG: Response preview: {model_response[:50]}...")
    
    # 4. Debugging with breakpoints (uncomment to use)
    # pdb.set_trace()  # Interactive debugging session

if __name__ == "__main__":
    # Run debugging demonstrations
    demonstrate_debugging_techniques()
    debug_rag_pipeline_example()
    
    print("\n=== Debugging Commands Reference ===")
    print("""
    PDB Commands (when in interactive mode):
    - n (next): Execute next line
    - s (step): Step into function calls
    - c (continue): Continue execution
    - l (list): Show current code
    - p <variable>: Print variable value
    - pp <variable>: Pretty-print variable
    - h (help): Show all commands
    - q (quit): Quit debugger
    
    Advanced Commands:
    - w (where): Show stack trace
    - u (up): Move up stack frame
    - d (down): Move down stack frame
    - b <line>: Set breakpoint at line
    - cl: Clear all breakpoints
    """)
