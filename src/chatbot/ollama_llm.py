"""
Ollama LLM Wrapper for LangChain Integration

This module provides a LangChain-compatible wrapper for Ollama models,
enabling seamless integration with the existing RAG pipeline.

Key Features:
- Compatible with LangChain LLM interface
- Supports streaming and non-streaming responses
- Configurable model parameters (temperature, top_p, etc.)
- Error handling and timeout management
- Conversation history support

Usage:
    from chatbot.ollama_llm import OllamaLLM
    
    llm = OllamaLLM(
        model="deepseek-r1:7b",
        base_url="http://localhost:11434"
    )
    
    response = llm("What is the capital of France?")
"""

import logging
from typing import Optional, List, Dict, Any, Iterator
import ollama
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """
    LangChain-compatible wrapper for Ollama models
    
    This class implements the LangChain LLM interface to work with Ollama,
    allowing seamless integration with existing RAG chains and agents.
    """
    
    model: str = Field(description="The Ollama model name to use")
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    temperature: float = Field(default=0.3, description="Temperature for generation")
    top_p: float = Field(default=0.9, description="Top-p for nucleus sampling")
    top_k: int = Field(default=40, description="Top-k for sampling")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    keep_alive: str = Field(default="5m", description="How long to keep model loaded")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the model")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the Ollama LLM wrapper"""
        super().__init__(**kwargs)
        
        # Configure Ollama client - don't use Field for this
        self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        
        # Verify model availability
        self._verify_model()
    
    @property
    def client(self):
        """Access the Ollama client"""
        return self._client
    
    def _verify_model(self):
        """Verify that the specified model is available"""
        try:
            # Try to list models to check if Ollama is running
            models_response = self.client.list()
            
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
                    if hasattr(model, 'model'):
                        # New Ollama client format
                        available_models.append(model.model)
                    elif hasattr(model, 'name'):
                        # Old format
                        available_models.append(model.name)
                    elif isinstance(model, dict):
                        available_models.append(model.get('name', model.get('model', '')))
                    else:
                        available_models.append(str(model))
            
            if self.model not in available_models:
                logger.warning(f"Model '{self.model}' not found in available models: {available_models}")
                logger.info(f"Attempting to pull model '{self.model}'...")
                
                # Try to pull the model
                self.client.pull(self.model)
                logger.info(f"Successfully pulled model '{self.model}'")
            else:
                logger.info(f"Model '{self.model}' is available")
                
        except Exception as e:
            logger.error(f"Failed to verify/pull model '{self.model}': {str(e)}")
            logger.info("Please ensure Ollama is running and the model is available")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier"""
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the Ollama model with the given prompt
        
        Args:
            prompt: The input prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LangChain integration
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text response from the model
        """
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Add system prompt if specified
            if self.system_prompt:
                request_params["system"] = self.system_prompt
            
            # Add stop sequences if provided
            if stop:
                request_params["options"]["stop"] = stop
            
            # Add keep_alive parameter
            request_params["keep_alive"] = self.keep_alive
            
            # Make the request to Ollama
            logger.debug(f"Sending request to Ollama: {self.model}")
            response = self.client.generate(**request_params)
            
            # Extract the generated text
            generated_text = response.get("response", "")
            
            # Log token usage if available
            if "eval_count" in response:
                logger.debug(f"Generated {response['eval_count']} tokens")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error calling Ollama model: {str(e)}")
            return f"Error: Failed to generate response. Please check if Ollama is running and the model '{self.model}' is available."
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream the response from Ollama model
        
        Args:
            prompt: The input prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LangChain integration
            **kwargs: Additional keyword arguments
            
        Yields:
            Chunks of generated text as they become available
        """
        try:
            # Prepare the request parameters for streaming
            request_params = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Add system prompt if specified
            if self.system_prompt:
                request_params["system"] = self.system_prompt
            
            # Add stop sequences if provided
            if stop:
                request_params["options"]["stop"] = stop
            
            # Add keep_alive parameter
            request_params["keep_alive"] = self.keep_alive
            
            # Stream the response
            logger.debug(f"Streaming request to Ollama: {self.model}")
            
            for chunk in self.client.generate(**request_params):
                if "response" in chunk:
                    token = chunk["response"]
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                    yield token
                    
        except Exception as e:
            logger.error(f"Error streaming from Ollama model: {str(e)}")
            yield f"Error: Failed to stream response. Please check if Ollama is running and the model '{self.model}' is available."
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        try:
            models_response = self.client.list()
            
            # Handle different response formats
            if hasattr(models_response, 'models'):
                models = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models = models_response['models']
            else:
                models = getattr(models_response, 'models', models_response)
            
            if models:
                for model in models:
                    model_name = None
                    if hasattr(model, 'model'):
                        model_name = model.model
                    elif hasattr(model, 'name'):
                        model_name = model.name
                    elif isinstance(model, dict):
                        model_name = model.get('name', model.get('model', ''))
                    
                    if model_name == self.model:
                        if hasattr(model, 'size'):
                            size = model.size
                        elif isinstance(model, dict):
                            size = model.get('size', 'Unknown')
                        else:
                            size = 'Unknown'
                        
                        return {
                            'name': model_name,
                            'size': size,
                            'digest': getattr(model, 'digest', model.get('digest', 'Unknown') if isinstance(model, dict) else 'Unknown'),
                            'modified_at': getattr(model, 'modified_at', model.get('modified_at', 'Unknown') if isinstance(model, dict) else 'Unknown')
                        }
            return {'error': f'Model {self.model} not found'}
        except Exception as e:
            return {'error': f'Failed to get model info: {str(e)}'}
    
    @classmethod
    def from_settings(cls, settings, **kwargs):
        """
        Create an OllamaLLM instance from application settings
        
        Args:
            settings: Application settings object
            **kwargs: Additional keyword arguments to override settings
            
        Returns:
            Configured OllamaLLM instance
        """
        return cls(
            model=kwargs.get('model', settings.MODEL_NAME),
            base_url=kwargs.get('base_url', settings.OLLAMA_BASE_URL),
            temperature=kwargs.get('temperature', settings.TEMPERATURE),
            top_p=kwargs.get('top_p', settings.TOP_P),
            max_tokens=kwargs.get('max_tokens', settings.MAX_NEW_TOKENS),
            timeout=kwargs.get('timeout', settings.OLLAMA_TIMEOUT),
            keep_alive=kwargs.get('keep_alive', settings.OLLAMA_KEEP_ALIVE),
            **kwargs
        )
