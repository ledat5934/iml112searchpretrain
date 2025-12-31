# src/iML/llm/ollama_chat.py
import logging
import os
import subprocess
import time
import requests
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatOllama(ChatOpenAI, BaseAssistantChat):
    """Ollama server with OpenAI-compatible API and LangGraph support."""
    
    def __init__(
        self,
        model: str,
        ollama_base_url: str = "http://localhost:11434",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        session_name: str = "default_session",
        **kwargs
    ):
        self.model_id = model
        self.ollama_base_url = ollama_base_url
        self.base_url = f"{ollama_base_url}/v1"  # Ollama OpenAI-compatible endpoint
        
        # Check if Ollama is running, if not try to start it
        self._ensure_ollama_running()
        
        # Pull model if not already available
        self._ensure_model_available()
        
        # Initialize OpenAI client pointing to Ollama server
        kwargs.update({
            "model_name": model,
            "openai_api_key": "dummy-key",  # Ollama doesn't require real key
            "openai_api_base": self.base_url,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "session_name": session_name,
        })
        
        super().__init__(**kwargs)
        self.initialize_conversation(self)
    
    def _ensure_ollama_running(self):
        """Check if Ollama is running, start in background if needed."""
        logger.info(f"Checking Ollama server at {self.ollama_base_url}...")
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is already running")
                return
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama server is not running. Attempting to start in background...")
        except Exception as e:
            logger.warning(f"Error checking Ollama server: {e}. Assuming it's running.")
            return
        
        # Try to start Ollama in background (non-blocking)
        try:
            logger.info("Starting Ollama server in background...")
            # Start in background with subprocess.Popen
            # Use preexec_fn to create new process group (Unix) or DETACHED_PROCESS (Windows)
            import sys
            if sys.platform == "win32":
                # Windows: use CREATE_NEW_PROCESS_GROUP
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=creation_flags
                )
            else:
                # Unix/Linux: use setsid to detach from parent
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
            
            logger.info(f"Ollama server process started with PID: {self.ollama_process.pid}")
            
            # Wait a bit for server to start
            logger.info("Waiting for Ollama server to initialize...")
            max_wait = 30  # Wait up to 30 seconds
            for i in range(max_wait):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        logger.info("Ollama server started successfully in background")
                        return
                except:
                    if i % 5 == 0:  # Log every 5 seconds
                        logger.info(f"Still waiting for Ollama server... ({i}s)")
                    pass
            
            # Check if process is still running
            if self.ollama_process.poll() is not None:
                stdout, stderr = self.ollama_process.communicate()
                logger.error(f"Ollama server process exited. stdout: {stdout[-500:]}, stderr: {stderr[-500:]}")
                raise RuntimeError("Ollama server failed to start")
            else:
                logger.warning("Ollama server may still be starting. Continuing...")
                
        except FileNotFoundError:
            logger.error("Ollama command not found. Please install Ollama:")
            logger.error("  Visit: https://ollama.ai/")
            logger.error("  Or run: curl -fsSL https://ollama.ai/install.sh | sh")
            raise RuntimeError("Ollama is not installed or not in PATH")
        except Exception as e:
            logger.error(f"Could not start Ollama automatically: {e}")
            logger.error("Please start Ollama manually in a separate terminal/process:")
            logger.error("  ollama serve")
            raise RuntimeError(f"Ollama server is not running. Please start it manually: `ollama serve`")
    
    def _ensure_model_available(self):
        """Check if model is available, pull if needed."""
        logger.info(f"Checking if model '{self.model_id}' is available in Ollama...")
        
        try:
            # Check available models
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if model exists (Ollama model names might have format like "gpt-oss:20b")
                model_found = False
                for name in model_names:
                    if self.model_id in name or name in self.model_id:
                        model_found = True
                        logger.info(f"Model found in Ollama: {name}")
                        break
                
                if not model_found:
                    logger.info(f"Model '{self.model_id}' not found. Pulling from Ollama...")
                    self._pull_model()
                else:
                    logger.info(f"Model '{self.model_id}' is available")
            else:
                logger.warning("Could not check Ollama models. Assuming model is available.")
        except Exception as e:
            logger.warning(f"Error checking Ollama models: {e}. Attempting to pull model...")
            self._pull_model()
    
    def _pull_model(self):
        """Pull model from Ollama."""
        # Ollama model name format: "gpt-oss:20b" for openai/gpt-oss-20b
        # Map HuggingFace model ID to Ollama model name
        ollama_model_name = self._map_to_ollama_name(self.model_id)
        
        logger.info(f"Pulling model '{ollama_model_name}' from Ollama (this may take a while)...")
        
        try:
            # Use Ollama API to pull model
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": ollama_model_name},
                timeout=300,  # 5 minutes timeout for pull
                stream=True
            )
            
            if response.status_code == 200:
                # Stream the pull progress
                for line in response.iter_lines():
                    if line:
                        try:
                            import json
                            data = json.loads(line)
                            if "status" in data:
                                logger.info(f"Ollama pull: {data['status']}")
                        except:
                            pass
                logger.info(f"Model '{ollama_model_name}' pulled successfully")
            else:
                logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
                raise RuntimeError(f"Failed to pull Ollama model: {ollama_model_name}")
        except requests.exceptions.Timeout:
            logger.error("Timeout while pulling model. Model may be very large.")
            raise
        except Exception as e:
            logger.error(f"Error pulling model from Ollama: {e}")
            raise
    
    def _map_to_ollama_name(self, hf_model_id: str) -> str:
        """Map HuggingFace model ID to Ollama model name."""
        # GPT-OSS-20B: openai/gpt-oss-20b -> gpt-oss:20b
        if "gpt-oss-20b" in hf_model_id.lower():
            return "gpt-oss:20b"
        elif "gpt-oss-120b" in hf_model_id.lower():
            return "gpt-oss:120b"
        else:
            # Default: try to convert openai/gpt-oss-20b -> gpt-oss:20b
            parts = hf_model_id.split("/")
            if len(parts) == 2:
                name = parts[1].replace("-", ":")
                return name
            return hf_model_id
    
    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "model": self.model_id,
            "provider": "ollama",
            "ollama_base_url": self.ollama_base_url
        }


def get_ollama_models() -> List[str]:
    """Get available Ollama models (hardcoded for now)."""
    return ["openai/gpt-oss-20b"]


def create_ollama_chat(config, session_name: str) -> AssistantChatOllama:
    """Create an Ollama chat model instance."""
    model = config.model
    ollama_base_url = getattr(config, "ollama_base_url", "http://localhost:11434")
    
    logger.info(f"Creating Ollama chat model: {model} for session: {session_name}")
    
    kwargs = {
        "model": model,
        "ollama_base_url": ollama_base_url,
        "session_name": session_name,
    }
    
    if hasattr(config, "max_tokens"):
        kwargs["max_tokens"] = config.max_tokens
    
    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature
    
    return AssistantChatOllama(**kwargs)

