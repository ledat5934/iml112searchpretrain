# src/iML/llm/vllm_chat.py
import logging
import os
import subprocess
import time
import signal
import requests
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from openai import OpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatVLLM(ChatOpenAI, BaseAssistantChat):
    """vLLM server with OpenAI-compatible API and LangGraph support."""
    
    def __init__(
        self,
        model: str,
        vllm_port: int = 8000,
        tensor_parallel_size: int = 2,
        max_model_len: Optional[int] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        session_name: str = "default_session",
        **kwargs
    ):
        self.model_id = model
        self.vllm_port = vllm_port
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.server_process = None
        self.base_url = f"http://localhost:{vllm_port}/v1"
        
        # Start vLLM server
        self._start_vllm_server()
        
        # Wait for server to be ready
        self._wait_for_server()
        
        # Initialize OpenAI client pointing to vLLM server
        kwargs.update({
            "model_name": model,  # vLLM will use the model name from server
            "openai_api_key": "dummy-key",  # vLLM doesn't require real key
            "openai_api_base": self.base_url,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "session_name": session_name,
        })
        
        super().__init__(**kwargs)
        self.initialize_conversation(self)
    
    def _start_vllm_server(self):
        """Start vLLM server in background."""
        logger.info(f"Starting vLLM server for model: {self.model_id}")
        logger.info(f"Using {self.tensor_parallel_size} GPUs (tensor_parallel_size={self.tensor_parallel_size})")
        
        # Check if vLLM is installed
        try:
            import vllm
            logger.info(f"vLLM version: {vllm.__version__}")
        except ImportError:
            logger.error("vLLM is not installed. Please install it:")
            logger.error("  pip install --pre vllm==0.10.1+gptoss --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128")
            raise ImportError("vLLM is not installed")
        
        # Build vLLM command
        cmd = [
            "vllm", "serve",
            self.model_id,
            "--port", str(self.vllm_port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
        ]
        
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        
        # Add memory optimization for multi-GPU
        # gpu-memory-utilization: use 90% of GPU memory (reserve 10% for buffer)
        cmd.extend(["--gpu-memory-utilization", "0.9"])
        
        # For GPT-OSS-20B with MXFP4, vLLM should handle it automatically
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group
            )
            logger.info(f"vLLM server started with PID: {self.server_process.pid}")
        except FileNotFoundError:
            logger.error("vLLM command not found. Please ensure vLLM is installed and in PATH.")
            raise
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            raise
    
    def _wait_for_server(self, timeout: int = 600):
        """Wait for vLLM server to be ready."""
        logger.info("Waiting for vLLM server to be ready...")
        logger.info(f"Timeout: {timeout} seconds (model loading may take several minutes)")
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        last_log_time = 0
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.server_process and self.server_process.poll() is not None:
                # Process exited - read output
                stdout, _ = self.server_process.communicate()
                logger.error(f"vLLM server process exited unexpectedly.")
                logger.error(f"Server output (last 1000 chars): {stdout[-1000:] if stdout else 'No output'}")
                raise RuntimeError("vLLM server failed to start. Check logs above for details.")
            
            try:
                # Check if server is ready by calling health endpoint
                response = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=2)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(f"vLLM server is ready! (took {elapsed:.1f} seconds)")
                    return
            except requests.exceptions.ConnectionError:
                # Server not ready yet, continue waiting
                pass
            except Exception as e:
                logger.debug(f"Health check error (expected during startup): {e}")
            
            elapsed = time.time() - start_time
            # Log every 30 seconds
            if int(elapsed) - last_log_time >= 30:
                logger.info(f"Still waiting for vLLM server... ({int(elapsed)}s elapsed)")
                last_log_time = int(elapsed)
            
            time.sleep(check_interval)
        
        # Timeout reached
        if self.server_process:
            logger.error(f"vLLM server did not become ready within {timeout} seconds")
            logger.error("Server may still be loading the model. Check server logs for details.")
        raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")
    
    def __del__(self):
        """Cleanup: stop vLLM server when object is destroyed."""
        self.cleanup()
    
    def cleanup(self):
        """Stop vLLM server."""
        if self.server_process:
            try:
                logger.info(f"Stopping vLLM server (PID: {self.server_process.pid})...")
                # Kill process group
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Error stopping vLLM server: {e}")
                # Fallback: just terminate
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                except:
                    self.server_process.kill()
            finally:
                self.server_process = None
                logger.info("vLLM server stopped")
    
    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "model": self.model_id,
            "provider": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "port": self.vllm_port
        }


def get_vllm_models() -> List[str]:
    """Get available vLLM models (hardcoded for now)."""
    return ["openai/gpt-oss-20b"]


def create_vllm_chat(config, session_name: str) -> AssistantChatVLLM:
    """Create a vLLM chat model instance."""
    model = config.model
    vllm_port = getattr(config, "vllm_port", 8000)
    tensor_parallel_size = getattr(config, "tensor_parallel_size", 2)
    max_model_len = getattr(config, "max_model_len", None)
    
    logger.info(f"Creating vLLM chat model: {model} for session: {session_name}")
    
    kwargs = {
        "model": model,
        "vllm_port": vllm_port,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "session_name": session_name,
    }
    
    if hasattr(config, "max_tokens"):
        kwargs["max_tokens"] = config.max_tokens
    
    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature
    
    return AssistantChatVLLM(**kwargs)

