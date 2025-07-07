"""
Gradio Web Interface for the RAG Chatbot

This module creates a modern web-based chat interface using Gradio.
It provides two interface options:
1. Simple ChatInterface - Clean, ChatGPT-like interface
2. Advanced Interface - Multi-tab interface with additional features

How the Chat Interface Works:
1. User types a message in the input box
2. Message is sent to the RAG pipeline for processing
3. Pipeline retrieves relevant documents and generates response
4. Response is displayed in the chat history
5. Sources are shown for transparency

Key Features:
- Real-time chat with streaming responses
- Source document citations
- Configurable model parameters
- Document search functionality
- System information display
- Modern, responsive UI design

The interface uses Gradio's ChatInterface component which provides:
- Automatic message history management
- Retry/undo/clear functionality
- Mobile-responsive design
- Markdown support for rich text
- Copy-to-clipboard functionality
"""

import gradio as gr
import logging
from typing import List, Tuple

# Import using absolute imports from the src directory
from chatbot.rag_pipeline import RAGPipeline
from config.settings import settings

# Setup logging for interface operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotInterface:
    """
    Modern Gradio interface for the RAG chatbot
    
    This class creates and manages the web-based chat interface.
    It handles user interactions and connects them to the RAG pipeline.
    
    Interface Components:
    - Chat input/output area
    - Message history management
    - Error handling and user feedback
    - Optional advanced controls (temperature, max tokens)
    - Document search functionality
    - System information display
    """
    
    def __init__(self):
        """
        Initialize the chatbot interface
        
        Creates a RAG pipeline instance and sets up error handling.
        The pipeline initialization happens in the background.
        """
        self.rag_pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """
        Initialize the RAG pipeline with error handling
        
        This method safely initializes the RAG pipeline and provides
        user feedback about the initialization status.
        """
        try:
            logger.info("🔄 Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline()
            logger.info("✅ RAG pipeline ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG pipeline: {str(e)}")
            self.rag_pipeline = None
    
    def chat_response(self, message: str, history: List[Tuple[str, str]]) -> str:
        """
        Generate response using RAG pipeline with conversation history
        
        This is the main function that handles user messages:
        1. Validates the message and pipeline state
        2. Sends the message to the RAG pipeline with conversation history
        3. Returns the generated response
        4. Handles any errors gracefully
        
        Args:
            message: User's input message
            history: Chat history as list of (user_message, bot_response) tuples
            
        Returns:
            Generated response or error message
        """
        if not self.rag_pipeline:
            return "❌ Sorry, the chatbot is not properly initialized. Please check the logs."
        
        if not message.strip():
            return "Please enter a question or message."
        
        try:
            # Generate response using the RAG pipeline with conversation history
            response = self.rag_pipeline.generate_response(message, conversation_history=history)
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def create_interface(self) -> gr.ChatInterface:
        """
        Create modern Gradio ChatInterface
        
        This creates a ChatGPT-like interface with:
        - Clean, modern design
        - Example prompts to get users started
        - Copy-to-clipboard functionality
        - Mobile-responsive layout
        - Custom CSS styling
        """
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            font-size: 14px !important;
        }
        """
        
        interface = gr.ChatInterface(
            fn=self.chat_response,
            title=settings.GRADIO_TITLE,
            description=settings.GRADIO_DESCRIPTION,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
                neutral_hue="gray"
            ),
            css=custom_css,
            examples=[
                "What is the main topic discussed in the documents?",
                "Can you summarize the key points from the documents?",
                "What are the technical specifications mentioned?",
                "Tell me about the methodology described in the documents.",
                "What conclusions can be drawn from the data?"
            ],
            cache_examples=False,
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo", 
            clear_btn="🗑️ Clear Chat",
            submit_btn="Send 📤",
            stop_btn="⏹️ Stop",
            textbox=gr.Textbox(
                placeholder="Ask me anything about your documents...",
                container=False,
                scale=7
            ),
            chatbot=gr.Chatbot(
                height=500,
                bubble_full_width=False,
                show_label=False,
                show_copy_button=True
            )
        )
        
        return interface
    
    def create_advanced_interface(self) -> gr.Interface:
        """
        Create an advanced interface with additional features
        
        This advanced interface includes:
        - Multi-tab layout for different functions
        - Configurable model parameters (temperature, max tokens)
        - Document search functionality
        - System information display
        - Real-time parameter adjustment
        """
        
        def advanced_chat(message, history, temperature, max_tokens, use_history):
            """
            Advanced chat with configurable parameters
            
            Allows users to adjust model behavior in real-time:
            - Temperature: Controls randomness (0.1 = focused, 1.0 = creative)
            - Max tokens: Limits response length
            - Use history: Whether to include conversation context
            """
            if not self.rag_pipeline:
                return history + [("System", "❌ Chatbot not initialized")]
            
            try:
                # Update settings temporarily
                original_temp = settings.TEMPERATURE
                original_tokens = settings.MAX_NEW_TOKENS
                original_history = settings.USE_CONVERSATION_HISTORY
                
                settings.TEMPERATURE = temperature
                settings.MAX_NEW_TOKENS = max_tokens
                settings.USE_CONVERSATION_HISTORY = use_history
                
                # Pass conversation history if enabled
                conversation_history = history if use_history else None
                response = self.rag_pipeline.generate_response(message, conversation_history)
                
                # Restore original settings
                settings.TEMPERATURE = original_temp
                settings.MAX_NEW_TOKENS = original_tokens
                settings.USE_CONVERSATION_HISTORY = original_history
                
                history.append((message, response))
                return history, ""
                
            except Exception as e:
                history.append((message, f"Error: {str(e)}"))
                return history, ""
        
        def get_similar_docs(query, k):
            """Get similar documents for a query"""
            if not self.rag_pipeline:
                return "❌ Chatbot not initialized"
            
            docs = self.rag_pipeline.get_similar_documents(query, k)
            if not docs:
                return "No similar documents found."
            
            result = f"Found {len(docs)} similar documents:\n\n"
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                result += f"**{i}. {source}**\n{content}\n\n"
            
            return result
        
        with gr.Blocks(theme=gr.themes.Soft(), title=settings.GRADIO_TITLE) as demo:
            gr.Markdown(f"# {settings.GRADIO_TITLE}")
            gr.Markdown(settings.GRADIO_DESCRIPTION)
            
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=400, show_copy_button=True)
                msg = gr.Textbox(
                    placeholder="Ask me anything about your documents...",
                    label="Your Message"
                )
                
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                    max_tokens = gr.Slider(50, 1000, value=128, label="Max Tokens")
                    use_history = gr.Checkbox(value=False, label="Use Conversation History")
                
                with gr.Row():
                    submit_btn = gr.Button("Send 📤", variant="primary")
                    clear_btn = gr.Button("Clear 🗑️")
                
                submit_btn.click(
                    advanced_chat,
                    inputs=[msg, chatbot, temperature, max_tokens, use_history],
                    outputs=[chatbot, msg]
                )
                msg.submit(
                    advanced_chat,
                    inputs=[msg, chatbot, temperature, max_tokens, use_history],
                    outputs=[chatbot, msg]
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
            
            with gr.Tab("🔍 Document Search"):
                with gr.Row():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter search terms...")
                    num_docs = gr.Slider(1, 10, value=5, label="Number of Documents")
                
                search_btn = gr.Button("Search 🔍", variant="primary")
                search_results = gr.Markdown()
                
                search_btn.click(
                    get_similar_docs,
                    inputs=[search_query, num_docs],
                    outputs=search_results
                )
            
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown(f"""
                ### System Information
                - **Model**: {settings.MODEL_NAME}
                - **Embedding Model**: {settings.EMBEDDING_MODEL}
                - **Documents Path**: {settings.DOCUMENTS_PATH}
                - **Vector Store Path**: {settings.VECTOR_STORE_PATH}
                - **Device**: {settings.DEVICE}
                - **Conversation History**: {"Enabled" if settings.USE_CONVERSATION_HISTORY else "Disabled"}
                - **Max History Turns**: {settings.MAX_HISTORY_TURNS}
                """)
        
        return demo

def launch_simple_interface():
    """Launch the simple chat interface"""
    chatbot = ChatbotInterface()
    interface = chatbot.create_interface()
    
    interface.launch(
        server_name=settings.GRADIO_HOST,
        server_port=settings.GRADIO_PORT,
        share=False,
        show_api=True,
        inbrowser=True
    )

def launch_advanced_interface():
    """Launch the advanced interface with more features"""
    chatbot = ChatbotInterface()
    interface = chatbot.create_advanced_interface()
    
    interface.launch(
        server_name=settings.GRADIO_HOST,
        server_port=settings.GRADIO_PORT,
        share=False,
        show_api=True,
        inbrowser=True
    )

if __name__ == "__main__":
    # Launch simple interface by default
    launch_simple_interface()