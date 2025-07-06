# Python Debugging Comprehensive Guide

This guide covers all essential debugging techniques for Python development, specifically tailored for your RAG chatbot project.

## 🎯 **Quick Debug Your RAG Pipeline**

### 1. **Basic Print Debugging**
```python
# Add these lines to your code where issues occur
print(f"DEBUG: Variable value = {variable}")
print(f"DEBUG: Type = {type(variable)}")
print(f"DEBUG: Length = {len(variable)}")
```

### 2. **Advanced Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### 3. **Interactive Debugging with PDB**
```python
import pdb

# Add this line where you want to break
pdb.set_trace()

# Or for conditional breakpoints
if some_condition:
    pdb.set_trace()
```

## 🔧 **Debugging Commands**

### **PDB Commands (Interactive Debugger)**
- `n` (next) - Execute next line
- `s` (step) - Step into function calls  
- `c` (continue) - Continue execution
- `l` (list) - Show current code around breakpoint
- `p variable_name` - Print variable value
- `pp variable_name` - Pretty print variable
- `w` (where) - Show stack trace
- `h` (help) - Show all commands
- `q` (quit) - Quit debugger

### **Inspect Variables**
```python
# In PDB session:
(Pdb) p query                    # Print query value
(Pdb) pp result                  # Pretty print result dict
(Pdb) type(response)             # Check variable type
(Pdb) len(documents)             # Get length
(Pdb) dir(pipeline)              # See available methods
```

## 🚀 **VS Code Debugging**

### **Setup Debug Configuration**
1. Open VS Code in your project
2. Go to Run & Debug (Ctrl+Shift+D)
3. Click "create a launch.json file"
4. Use this configuration:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug RAG Chatbot",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/app.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/src",
            "justMyCode": false
        }
    ]
}
```

### **Set Breakpoints**
- Click left margin in VS Code to set breakpoints
- Right-click for conditional breakpoints
- Use "Debug Console" to inspect variables

## 🔍 **Debug Your Specific RAG Issues**

### **1. Document Loading Issues**
```python
# Add to your document loading function
import os
print(f"Looking for documents in: {documents_path}")
print(f"Path exists: {os.path.exists(documents_path)}")
print(f"Files found: {os.listdir(documents_path) if os.path.exists(documents_path) else 'None'}")
```

### **2. Vector Store Issues**
```python
# Debug vector store creation
print(f"Creating embeddings for {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"Document {i}: {len(doc.page_content)} characters")
    if i >= 3:  # Just show first few
        break
```

### **3. Model Response Issues**
```python
# Debug model responses
print(f"Query: {query}")
print(f"Retrieved docs: {len(source_documents)}")
for i, doc in enumerate(source_documents):
    print(f"Source {i}: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:100]}...")
```

## 📊 **Performance Debugging**

### **Time Function Execution**
```python
import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@time_function
def your_function():
    # Your code here
    pass
```

### **Memory Usage**
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 🧪 **Testing & Debugging**

### **Unit Test with Debugging**
```python
import unittest

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()
    
    def test_document_loading(self):
        # Add debugging prints
        print(f"Testing with {len(self.pipeline.documents)} documents")
        self.assertGreater(len(self.pipeline.documents), 0)
    
    def test_query_response(self):
        query = "What are baby milestones?"
        response = self.pipeline.generate_response(query)
        print(f"Query: {query}")
        print(f"Response: {response[:100]}...")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
```

## 🔥 **Common Debugging Scenarios**

### **1. Import Errors**
```python
# Debug import issues
try:
    from chatbot.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"Import error: {e}")
    import sys
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
```

### **2. Path Issues**
```python
# Debug path problems
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Documents path exists: {os.path.exists('data/documents')}")
```

### **3. Model Loading Issues**
```python
# Debug model loading
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✅ Model loaded successfully: {model_name}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print(f"Trying smaller model...")
```

## 🛠️ **Debugging Tools & Libraries**

### **Install Debugging Tools**
```bash
pip install memory-profiler
pip install line-profiler
pip install py-spy
```

### **Memory Profiling**
```python
@profile
def memory_intensive_function():
    # Your code here
    pass

# Run with: python -m memory_profiler your_script.py
```

### **Line Profiling**
```python
@profile
def cpu_intensive_function():
    # Your code here
    pass

# Run with: kernprof -l -v your_script.py
```

## 🎯 **Quick Debugging Checklist**

When your RAG pipeline has issues:

1. **✅ Check paths**: Are documents/models in the right location?
2. **✅ Check imports**: Are all modules importing correctly?
3. **✅ Check data**: Are documents loading and processing?
4. **✅ Check models**: Are embeddings and LLM loading?
5. **✅ Check queries**: Are test queries working?
6. **✅ Check responses**: Are responses being generated?

## 🚨 **Emergency Debugging Commands**

When things go wrong, run these:

```bash
# Quick test of your environment
cd /Users/ashish_kumar/chat-bot/chatbot-app
python debug_paths.py

# Test RAG pipeline components
python debug_rag.py

# Run with debugging enabled
cd src && python -u app.py 2>&1 | tee debug_output.log

# Check what's running
ps aux | grep python

# Check memory usage
top -pid $(pgrep -f "python.*app.py")
```

## 🎉 **Debug Successfully!**

Remember:
- **Start simple**: Use print statements first
- **Be systematic**: Test one component at a time  
- **Use the right tool**: PDB for logic, logging for flow, profiling for performance
- **Save your debugging**: Keep debug scripts for future use

Good luck debugging your RAG chatbot! 🤖🔧
