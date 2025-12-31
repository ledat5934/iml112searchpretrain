# src/iML/llm/huggingface_chat.py
import logging
import os
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)

# Module-level cache for model instances (shared across all AssistantChatHuggingFace instances)
# Cache key: (model_id, quantization, device_map, trust_remote_code)
# Cache value: {"tokenizer": tokenizer, "model": model, "pipeline": pipeline}
_model_cache: Dict[tuple, Dict[str, Any]] = {}


class AssistantChatHuggingFace(BaseAssistantChat):
    """HuggingFace local model with LangGraph support and 4-bit quantization."""
    
    # Declare fields for Pydantic
    model_id: str = Field(default="")
    device_map: str = Field(default="auto")
    quantization: str = Field(default="4bit")
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.0)
    trust_remote_code: bool = Field(default=False)
    tokenizer: Optional[Any] = Field(default=None, exclude=True)
    model: Optional[Any] = Field(default=None, exclude=True)
    pipeline: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        quantization: str = "4bit",  # "4bit" or "8bit" or None
        max_tokens: int = 8192,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        session_name: str = "default_session",
        **kwargs
    ):
        # Initialize base class first
        super().__init__(session_name=session_name, **kwargs)
        
        # Set fields after super().__init__()
        self.model_id = model
        self.device_map = device_map
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        
        # Load model và tokenizer
        self._load_model()
        
        # Initialize conversation với LangGraph
        self.initialize_conversation(self)
    
    def _get_cache_key(self) -> tuple:
        """Generate cache key based on model configuration."""
        return (self.model_id, self.quantization, self.device_map, self.trust_remote_code)
    
    def _load_model(self):
        """Load model với 4-bit quantization. Uses cache if available."""
        cache_key = self._get_cache_key()
        
        # Check cache first
        if cache_key in _model_cache:
            logger.info(f"Reusing cached model: {self.model_id} with {self.quantization} quantization")
            cached = _model_cache[cache_key]
            self.tokenizer = cached["tokenizer"]
            self.model = cached["model"]
            # Create new pipeline for this instance (pipeline may have internal state)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device_map,
            )
            logger.info(f"Model reused from cache successfully (pipeline created for this instance)")
            return
        
        # Cache miss - load model
        logger.info(f"Loading HuggingFace model: {self.model_id} with {self.quantization} quantization (cache miss)")
        
        # Get HuggingFace token from environment (optional, for gated models or rate limiting)
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info("Using HuggingFace token from environment (for rate limiting/gated models)")
        else:
            logger.info("No HuggingFace token found - using anonymous access (works for public models)")
        
        try:
            # Load tokenizer
            # Note: Some models may trigger 404 for additional_chat_templates, but this is harmless
            # We catch and suppress this specific error, then retry with use_fast=False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    token=hf_token  # Pass token if available
                )
            except Exception as e:
                # Check if error is about additional_chat_templates (harmless 404)
                error_str = str(e)
                error_type = type(e).__name__
                
                # Check for 404 or RemoteEntryNotFoundError or additional_chat_templates
                is_additional_templates_error = (
                    "additional_chat_templates" in error_str or 
                    ("404" in error_str and "additional_chat_templates" in error_str.lower()) or
                    ("Entry Not Found" in error_str and "additional_chat_templates" in error_str.lower())
                )
                
                if is_additional_templates_error:
                    logger.warning("additional_chat_templates not found (harmless 404). Retrying with use_fast=False...")
                    # Retry with use_fast=False - wrap in try-except to handle if retry also fails
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_id,
                            trust_remote_code=self.trust_remote_code,
                            use_fast=False,
                            token=hf_token
                        )
                        logger.info("Tokenizer loaded successfully (slow tokenizer mode)")
                    except Exception as e2:
                        # If retry also fails, check if it's the same harmless error
                        error_str2 = str(e2)
                        is_same_error = (
                            "additional_chat_templates" in error_str2 or
                            ("404" in error_str2 and "additional_chat_templates" in error_str2.lower())
                        )
                        
                        if is_same_error:
                            # This is still the harmless additional_chat_templates error
                            # Transformers library should handle this, but if it doesn't,
                            # we can try to work around it by loading from cache or using a different method
                            logger.warning("Retry also encountered additional_chat_templates 404. This is harmless.")
                            logger.info("Attempting final load - transformers should handle this gracefully...")
                            
                            try:
                                # Final attempt: explicitly set parameters to avoid the check
                                self.tokenizer = AutoTokenizer.from_pretrained(
                                    self.model_id,
                                    trust_remote_code=self.trust_remote_code,
                                    use_fast=False,
                                    token=hf_token,
                                    local_files_only=False
                                )
                                logger.info("Tokenizer loaded successfully after handling additional_chat_templates 404")
                            except Exception as e3:
                                # If still fails, log detailed error but this should be very rare
                                logger.error(f"Unexpected error during tokenizer load after handling additional_chat_templates: {e3}")
                                logger.error("This might indicate a different issue. Please check model availability and network connection.")
                                raise
                        else:
                            # Different error in retry - re-raise
                            logger.error(f"Different error during retry: {e2}")
                            raise
                else:
                    # Re-raise if it's a different error (not about additional_chat_templates)
                    raise
            
            # Check if model is pre-quantized (e.g., GPT-OSS-20B uses Mxfp4Config)
            # Load config first to check quantization status
            from transformers import AutoConfig
            try:
                model_config = AutoConfig.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    token=hf_token
                )
                is_pre_quantized = hasattr(model_config, 'quantization_config') and model_config.quantization_config is not None
                if is_pre_quantized:
                    quant_type = type(model_config.quantization_config).__name__ if model_config.quantization_config else None
                    logger.info(f"Model is pre-quantized with {quant_type}. Skipping additional quantization.")
            except Exception as e:
                logger.warning(f"Could not check model config: {e}. Proceeding with load...")
                is_pre_quantized = False
            
            # Configure quantization (only if model is not pre-quantized)
            quantization_config = None
            
            if not is_pre_quantized:
                # Check if bitsandbytes is available
                try:
                    import bitsandbytes as bnb
                    bitsandbytes_available = True
                    logger.info("bitsandbytes is available for quantization")
                except ImportError:
                    bitsandbytes_available = False
                    logger.warning("bitsandbytes is not available. Quantization will be disabled.")
                    logger.warning("To enable quantization, install bitsandbytes: pip install bitsandbytes>=0.41.0")
                    logger.warning("Note: bitsandbytes requires CUDA and may not work on all environments (e.g., Kaggle CPU-only)")
                
                if bitsandbytes_available:
                    if self.quantization == "4bit":
                        try:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            logger.info("Using 4-bit quantization with BitsAndBytesConfig")
                        except Exception as e:
                            logger.error(f"Failed to create BitsAndBytesConfig for 4-bit quantization: {e}")
                            logger.warning("Falling back to no quantization (full precision)")
                            quantization_config = None
                    elif self.quantization == "8bit":
                        try:
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            logger.info("Using 8-bit quantization with BitsAndBytesConfig")
                        except Exception as e:
                            logger.error(f"Failed to create BitsAndBytesConfig for 8-bit quantization: {e}")
                            logger.warning("Falling back to no quantization (full precision)")
                            quantization_config = None
                    else:
                        logger.info(f"No quantization specified, using default dtype")
                else:
                    # bitsandbytes not available - disable quantization
                    if self.quantization in ["4bit", "8bit"]:
                        logger.warning(f"Requested {self.quantization} quantization but bitsandbytes is not available.")
                        logger.warning("Continuing without quantization (full precision). Model will use more memory.")
                    else:
                        logger.info(f"No quantization specified, using default dtype")
            else:
                logger.info("Model is pre-quantized. Using model's built-in quantization (no additional quantization needed).")
            
            # Load model
            model_kwargs = {
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
            }
            
            # Only add quantization_config if model is not pre-quantized
            if quantization_config and not is_pre_quantized:
                model_kwargs["quantization_config"] = quantization_config
            elif not is_pre_quantized:
                # Only set torch_dtype if model is not pre-quantized
                model_kwargs["torch_dtype"] = torch.float16
            
            # Add token to model_kwargs if available
            if hf_token:
                model_kwargs["token"] = hf_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Set pad_token if it's the same as eos_token to avoid attention mask warnings
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device_map,
            )
            
            # Cache model and tokenizer for reuse (pipeline is created per instance)
            _model_cache[cache_key] = {
                "tokenizer": self.tokenizer,
                "model": self.model,
            }
            logger.info(f"Model loaded successfully with {self.quantization} quantization and cached for reuse")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Invoke model with messages (for LangGraph compatibility)."""
        try:
            # Convert LangChain messages to format expected by model
            formatted_input = self._format_messages(messages)
            
            # Prepare generation kwargs
            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "return_full_text": False,
                "do_sample": self.temperature > 0.0,
            }
            
            # Only add temperature if do_sample is True
            if self.temperature > 0.0:
                generation_kwargs["temperature"] = self.temperature
            
            # Merge with any additional kwargs
            generation_kwargs.update(kwargs)
            
            # Generate response
            outputs = self.pipeline(
                formatted_input,
                **generation_kwargs
            )
            
            # Extract generated text with better error handling
            generated_text = ""
            if outputs is None:
                logger.error("Pipeline returned None")
                generated_text = ""
            elif isinstance(outputs, list):
                if len(outputs) > 0:
                    first_output = outputs[0]
                    if isinstance(first_output, dict):
                        generated_text = first_output.get("generated_text", "")
                    else:
                        generated_text = str(first_output)
                else:
                    logger.warning("Pipeline returned empty list")
                    generated_text = ""
            elif isinstance(outputs, dict):
                generated_text = outputs.get("generated_text", str(outputs))
            else:
                generated_text = str(outputs)
            
            if not generated_text or generated_text.strip() == "":
                logger.warning("Generated text is empty, returning empty response")
                generated_text = ""
            
            # Return as AIMessage
            return AIMessage(content=generated_text)
            
        except Exception as e:
            logger.error(f"Error during model invocation: {e}")
            raise
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to model format."""
        try:
            # Use chat template if available (automatically handles harmony format)
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                # Extract text from messages
                formatted = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        formatted.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        formatted.append({"role": "assistant", "content": msg.content})
                
                # Only apply chat template if we have at least one message
                if len(formatted) > 0:
                    try:
                        formatted_text = self.tokenizer.apply_chat_template(
                            formatted,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        if formatted_text:
                            return formatted_text
                    except (IndexError, KeyError, AttributeError) as e:
                        logger.warning(f"Error applying chat template: {e}, using fallback")
                        # Fallback to simple concatenation
                        pass
                else:
                    logger.warning("No messages to format, using fallback")
            
            # Fallback: concatenate messages
            text_parts = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    text_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    text_parts.append(f"Assistant: {msg.content}")
                elif hasattr(msg, 'content'):
                    text_parts.append(str(msg.content))
            
            if text_parts:
                return "\n".join(text_parts)
            else:
                # Ultimate fallback
                return "\n".join([str(msg) for msg in messages])
                
        except Exception as e:
            logger.warning(f"Error formatting messages with chat template: {e}, using fallback")
            # Fallback: simple concatenation
            return "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])
    
    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "model": self.model_id,
            "provider": "huggingface",
            "quantization": self.quantization,
            "device_map": self.device_map
        }


def get_huggingface_models() -> List[str]:
    """Get available HuggingFace models (hardcoded for now)."""
    return ["openai/gpt-oss-20b", "Qwen/Qwen2.5-Coder-7B-Instruct"]


def create_huggingface_chat(config, session_name: str) -> AssistantChatHuggingFace:
    """Create a HuggingFace chat model instance."""
    model = config.model
    quantization = getattr(config, "quantization", "4bit")
    device_map = getattr(config, "device_map", "auto")
    trust_remote_code = getattr(config, "trust_remote_code", False)
    
    logger.info(f"Creating HuggingFace chat model: {model} for session: {session_name}")
    
    kwargs = {
        "model": model,
        "quantization": quantization,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "session_name": session_name,
    }
    
    if hasattr(config, "max_tokens"):
        kwargs["max_tokens"] = config.max_tokens
    
    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature
    
    return AssistantChatHuggingFace(**kwargs)

