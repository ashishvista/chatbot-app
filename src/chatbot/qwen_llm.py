"""
Qwen LLM Wrapper for LangChain Integration

This module provides a LangChain-compatible wrapper for Qwen models using Hugging Face transformers,
enabling seamless integration with the existing RAG pipeline without requiring API keys.

Key Features:
- Compatible with LangChain LLM interface
- Uses Hugging Face transformers for local inference
- Supports conversation history and context management
- Configurable model parameters (temperature, max_tokens, etc.)
- No API keys required - runs locally

Usage:
    from chatbot.qwen_llm import QwenLLM
    
    llm = QwenLLM(
        model="Qwen/Qwen3-8B",  # Using Qwen3-8B for better performance
        temperature=0.3
    )
    
    response = llm("What is the capital of France?")
"""

import logging
from typing import Optional, List, Dict, Any, Iterator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.llms import HuggingFacePipeline
from pydantic import Field

logger = logging.getLogger(__name__)

class QwenLLM(LLM):
    """
    LangChain-compatible wrapper for Qwen models using Hugging Face transformers
    
    This class implements the LangChain LLM interface to work with Qwen models locally,
    allowing seamless integration with existing RAG chains without API keys.
    """
    
    model: str = Field(description="The Qwen model name to use")
    temperature: float = Field(default=0.3, description="Temperature for generation")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the model")
    device: str = Field(default="cpu", description="Device to run the model on")
    
    # Declare model components as optional fields to avoid Pydantic validation issues
    tokenizer: Optional[Any] = Field(default=None, exclude=True, description="Tokenizer instance")
    model_instance: Optional[Any] = Field(default=None, exclude=True, description="Model instance")
    pipeline: Optional[Any] = Field(default=None, exclude=True, description="HF Pipeline instance")
    hf_pipeline: Optional[Any] = Field(default=None, exclude=True, description="LangChain HF Pipeline wrapper")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra attributes
    
    def __init__(self, **kwargs):
        """Initialize the Qwen LLM wrapper"""
        super().__init__(**kwargs)
        
        # Initialize the model and tokenizer
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen model and tokenizer"""
        try:
            logger.info(f"🔄 Loading Qwen model: {self.model}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_instance = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                trust_remote_code=True,
                device_map="auto" if self.device != "cpu" else None
            )
            
            # Move to specified device
            if self.device == "cpu":
                model_instance = model_instance.to("cpu")
            
            # Create pipeline for easier inference
            pipeline_instance = pipeline(
                "text-generation",
                model=model_instance,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=tokenizer.eos_token_id,
                device=-1 if self.device == "cpu" else 0,
                return_full_text=False
            )
            
            # Create LangChain pipeline wrapper
            hf_pipeline = HuggingFacePipeline(pipeline=pipeline_instance)
            
            # Use object.__setattr__ to bypass Pydantic validation
            object.__setattr__(self, 'tokenizer', tokenizer)
            object.__setattr__(self, 'model_instance', model_instance)
            object.__setattr__(self, 'pipeline', pipeline_instance)
            object.__setattr__(self, 'hf_pipeline', hf_pipeline)
            
            logger.info(f"✅ Qwen model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qwen model: {str(e)}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier"""
        return "qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the Qwen model with the given prompt
        
        Args:
            prompt: The input prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LangChain integration
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text response from the model
        """
        try:
            # Format prompt with system message if provided
            formatted_prompt = self._format_prompt(prompt)
            
            logger.debug(f"Sending request to Qwen model: {self.model}")
            
            # Generate response using the pipeline
            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            if response and len(response) > 0:
                generated_text = response[0]['generated_text'].strip()
            else:
                generated_text = "No response generated"
            
            logger.debug(f"Generated response length: {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error calling Qwen model: {str(e)}")
            return f"Error: Failed to generate response. {str(e)}"
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt with system message if provided"""
        if self.system_prompt:
            return f"System: {self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            return f"User: {prompt}\n\nAssistant:"
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream the response from Qwen model
        
        Args:
            prompt: The input prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LangChain integration
            **kwargs: Additional keyword arguments
            
        Yields:
            Chunks of generated text as they become available
        """
        try:
            # For now, we'll use the non-streaming version and yield the full response
            # This can be enhanced later if streaming support is added to qwen_agent
            response = self._call(prompt, stop, run_manager, **kwargs)
            
            # Simulate streaming by yielding chunks
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                if run_manager:
                    run_manager.on_llm_new_token(chunk)
                yield chunk
                    
        except Exception as e:
            logger.error(f"Error streaming from Qwen model: {str(e)}")
            yield f"Error: Failed to stream response. {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        try:
            return {
                'name': self.model,
                'type': 'Qwen',
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'system_prompt': self.system_prompt is not None
            }
        except Exception as e:
            return {'error': f'Failed to get model info: {str(e)}'}
    
    @classmethod
    def from_settings(cls, settings, **kwargs):
        """
        Create a QwenLLM instance from application settings
        
        Args:
            settings: Application settings object
            **kwargs: Additional keyword arguments to override settings
            
        Returns:
            Configured QwenLLM instance
        """
        # Extract kwargs to avoid conflicts
        system_prompt = kwargs.pop('system_prompt', None)
        device = kwargs.pop('device', settings.DEVICE)
        
        return cls(
            model=kwargs.get('model', settings.MODEL_NAME),
            temperature=kwargs.get('temperature', settings.TEMPERATURE),
            max_tokens=kwargs.get('max_tokens', settings.MAX_NEW_TOKENS),
            system_prompt=system_prompt,
            device=device,
            **kwargs
        )
