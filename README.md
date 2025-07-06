# 🤖 Domain-Specific RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Hugging Face models, LangChain, and Gradio. This chatbot specializes in answering questions about your domain-specific documents using state-of-the-art AI technologies.

## 🎯 Features

- **🧠 Smart Document Understanding**: Uses embeddings to understand document context
- **💬 Natural Conversations**: Powered by advanced language models
- **📚 Source Attribution**: Shows which documents were used for each answer
- **🌐 Modern Web Interface**: Beautiful, responsive Gradio UI
- **⚡ Fast Retrieval**: FAISS vector database for lightning-fast search
- **🔧 Configurable**: Easy to customize models and parameters
- **📱 Mobile Friendly**: Works on desktop, tablet, and mobile devices

## 🏗️ Architecture

### How RAG Works in This System

```
User Query → Embedding Model → Vector Search → Document Retrieval → Language Model → Response
```

1. **Document Processing**: Your documents are split into chunks and converted to embeddings
2. **Query Processing**: User questions are converted to embeddings using the same model
3. **Similarity Search**: System finds the most relevant document chunks
4. **Context Augmentation**: Retrieved chunks provide context to the language model
5. **Response Generation**: Model generates accurate, context-aware responses

### Technical Stack

- **Language Model**: Microsoft DialoGPT (conversational AI)
- **Embeddings**: Sentence Transformers (semantic understanding)
- **Vector Database**: FAISS (fast similarity search)
- **Framework**: LangChain (RAG orchestration)
- **UI**: Gradio (modern web interface)
- **Backend**: Python with PyTorch

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9-3.11 recommended)
- **4GB+ RAM** (8GB+ recommended for better performance)
- **2GB+ disk space** (for models and data)
- **Optional**: NVIDIA GPU with CUDA for faster inference

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd chatbot-app
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Documents

Place your documents in the `data/documents/` folder:

```bash
# Example documents are already provided
ls data/documents/
# baby_milestones_0_24_months.txt
# pediatrician_recommendations.md
# patient_history_emma_johnson.txt
# patient_history_oliver_martinez.txt
# patient_history_sophia_chen.txt
# patient_history_aiden_thompson.txt
# patient_history_lucas_anderson.txt
# patient_history_zoe_williams.txt
```

**Sample Patient Cases Included:**
- **Emma Johnson (18 months)**: Typical development, healthy growth
- **Oliver Martinez (6 months)**: Formula-fed baby, beginning solid foods
- **Sophia Chen (12 months)**: Allergies and eczema management
- **Aiden Thompson (24 months)**: Advanced 2-year-old development
- **Lucas Anderson (9 months)**: Premature baby with special considerations
- **Zoe Williams (15 months)**: Developmental delays requiring intervention

**Supported formats:**
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (experimental)

### 5. Run the Application

```bash
cd src
python app.py
```

The application will:
1. Initialize the RAG pipeline
2. Process your documents
3. Create vector embeddings
4. Launch the web interface at `http://localhost:7860`

## 📖 Usage Guide

### Web Interface

Once running, open your browser to `http://localhost:7860` and you'll see:

#### Simple Interface
- Clean ChatGPT-like chat interface
- Ask questions about your documents
- Get responses with source citations
- Copy responses to clipboard

#### Advanced Interface (Multi-tab)
- **💬 Chat Tab**: Full conversation with parameter controls
- **🔍 Document Search Tab**: Search for specific document content
- **ℹ️ System Info Tab**: View configuration and model details

### Example Queries

Try these sample questions with the provided pediatric documents:

**General Milestone Questions:**
```
"What are the key milestones for a 12-month-old baby?"
"When should I introduce solid foods to my baby?"
"What vaccinations are needed in the first year?"
"What are red flags I should watch for in infant development?"
"When do babies typically start walking?"
```

**Patient-Specific Questions:**
```
"Tell me about Emma's developmental progress at 18 months"
"How is Oliver doing with his feeding at 6 months?"
"What concerns does Sophia have with her allergies?"
"Describe Aiden's language development at 2 years"
"What special considerations does Lucas need as a premature baby?"
"What developmental concerns are noted for Zoe?"
```

**Clinical Questions:**
```
"What is the vaccination schedule for the first 2 years?"
"How should I manage a baby with eczema?"
"What are signs of developmental delays to watch for?"
"When should I be concerned about speech development?"
"What are the recommendations for premature baby care?"
```

### Configuration Options

The advanced interface allows you to adjust:
- **Temperature** (0.1-1.0): Controls response creativity
- **Max Tokens** (50-1000): Limits response length
- **Retrieval Count**: Number of documents to use for context

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_NAME=microsoft/DialoGPT-medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Keys (Optional)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# Interface Settings
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# Generation Settings
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9

# System
DEBUG_MODE=false
DEVICE=auto
```

### Model Options

You can use different models by changing the configuration:

**Language Models:**
- `microsoft/DialoGPT-small` - Faster, less memory
- `microsoft/DialoGPT-medium` - Balanced (default)
- `microsoft/DialoGPT-large` - Better quality, more memory
- `facebook/blenderbot-400M-distill` - Alternative option

**Embedding Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, good quality (default)
- `sentence-transformers/all-mpnet-base-v2` - Better quality, slower
- `sentence-transformers/distilbert-base-nli-stsb-mean-tokens` - Lightweight

## 🗂️ Project Structure

```
chatbot-app/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── data/
│   ├── documents/           # Your source documents
│   └── vector_store/        # Generated embeddings (auto-created)
└── src/
    ├── app.py              # Main application entry point
    ├── config/
    │   ├── __init__.py
    │   └── settings.py     # Configuration management
    ├── chatbot/
    │   ├── __init__.py
    │   ├── models.py       # Model definitions
    │   └── rag_pipeline.py # Core RAG implementation
    ├── ui/
    │   ├── __init__.py
    │   └── gradio_interface.py # Web interface
    └── utils/
        ├── __init__.py
        ├── document_loader.py  # Document processing
        └── vector_store.py     # Vector database utilities
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the src directory
cd src
python app.py
```

**2. Memory Issues**
```bash
# Use smaller model
export MODEL_NAME=microsoft/DialoGPT-small
```

**3. Slow Performance**
- Enable GPU if available
- Reduce chunk size in settings
- Use smaller embedding model

**4. No Documents Found**
```bash
# Check documents directory
ls -la data/documents/
# Make sure files are .txt or .md format
```

**5. Port Already in Use**
```bash
# Change port in .env file
echo "GRADIO_PORT=7861" >> .env
```

### Performance Optimization

**For CPU-only systems:**
```env
DEVICE=cpu
MODEL_NAME=microsoft/DialoGPT-small
CHUNK_SIZE=500
```

**For GPU systems:**
```env
DEVICE=cuda
MODEL_NAME=microsoft/DialoGPT-medium
CHUNK_SIZE=1000
```

## 🚀 Advanced Usage

### Adding New Documents

1. Place new documents in `data/documents/`
2. Restart the application to reprocess
3. Or use the document management API (if implemented)

### Custom Models

To use custom Hugging Face models:

```python
# In settings.py or .env
MODEL_NAME=your-username/your-model-name
EMBEDDING_MODEL=your-embedding-model
```

### API Integration

The system can be extended to provide REST API endpoints:

```python
# Example API endpoint (not implemented by default)
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    response = rag_pipeline.generate_response(data['message'])
    return {'response': response}
```

## 📊 Monitoring and Logs

### View Logs
```bash
# Application logs
tail -f logs/chatbot.log

# Real-time debugging
DEBUG_MODE=true python app.py
```

### Performance Metrics
- Response time tracking
- Document retrieval accuracy
- Model inference speed
- Memory usage monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing excellent pre-trained models
- **LangChain**: For the RAG framework
- **Gradio**: For the beautiful web interface
- **FAISS**: For efficient vector search
- **Sentence Transformers**: For semantic embeddings

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all dependencies are installed correctly
4. Verify your documents are in supported formats

## 🔄 Updates and Maintenance

To update the system:

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear vector store to rebuild with new models
rm -rf data/vector_store/
python src/app.py
```

---

**Happy Chatting! 🎉**

This RAG chatbot will help you unlock the knowledge in your documents through natural conversation. Whether you're building a medical assistant, legal advisor, or any domain-specific AI helper, this system provides a solid foundation for intelligent document interaction.