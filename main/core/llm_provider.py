# core/llm_provider_gpu.py - Enhanced LLM provider with proper GPU support
import gc
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# Import llama-cpp-python
try:
    from llama_cpp import Llama

    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None

logger = logging.getLogger(__name__)


class LLMProvider:
    """Enhanced LLM provider with proper GPU support"""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "local_gguf")
        self.model_path = self._get_model_path()
        self.llm = None

        # GPU-optimized settings
        self.n_ctx = int(os.getenv("LLAMA_N_CTX", "4096"))
        self.n_threads = int(os.getenv("LLAMA_THREADS", "-1"))

        # GPU SETTINGS - KEY CHANGE
        self.n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "32"))  # Default to high GPU usage
        self.use_gpu = os.getenv("LLAMA_USE_GPU", "true").lower() == "true"

        # Performance settings
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        self.top_k = int(os.getenv("TOP_K", "40"))
        self.repeat_penalty = float(os.getenv("REPEAT_PENALTY", "1.1"))

        # Advanced GPU settings
        self.n_batch = int(os.getenv("LLAMA_N_BATCH", "512"))
        self.use_mmap = os.getenv("LLAMA_USE_MMAP", "true").lower() == "true"
        self.use_mlock = os.getenv("LLAMA_USE_MLOCK", "false").lower() == "true"
        self.f16_kv = os.getenv("LLAMA_F16_KV", "true").lower() == "true"

        # Auto-detect GPU if not explicitly disabled
        if self.use_gpu:
            self._detect_optimal_gpu_settings()

        # Initialize model
        self._load_model()

    def _detect_optimal_gpu_settings(self):
        """Auto-detect optimal GPU settings"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name(0)

                logger.info(f"🎮 Detected GPU: {gpu_name} ({gpu_memory:.1f}GB)")

                # Auto-adjust GPU layers based on available memory
                if gpu_memory >= 8.0:  # 8GB+
                    suggested_layers = 32
                elif gpu_memory >= 6.0:  # 6-8GB
                    suggested_layers = 24
                elif gpu_memory >= 4.0:  # 4-6GB
                    suggested_layers = 16
                else:  # <4GB
                    suggested_layers = 8

                # Only override if not explicitly set
                if os.getenv("LLAMA_N_GPU_LAYERS") is None:
                    self.n_gpu_layers = suggested_layers
                    logger.info(f"🔧 Auto-set GPU layers: {self.n_gpu_layers}")

            else:
                logger.warning("⚠️  CUDA not available - disabling GPU")
                self.use_gpu = False
                self.n_gpu_layers = 0

        except ImportError:
            logger.warning("⚠️  PyTorch not available for GPU detection")
        except Exception as e:
            logger.warning(f"⚠️  GPU detection failed: {e}")

    def _get_model_path(self) -> str:
        """Find GGUF file"""
        # Check environment variable first
        model_path = os.getenv("LLAMA_MODEL_PATH")
        if model_path and Path(model_path).exists():
            logger.info(f"Using model from environment: {model_path}")
            return model_path

        # Get directories
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent.parent

        possible_models_dirs = [
            current_dir / "models",
            current_dir / "main" / "models",
            script_dir / "models",
            script_dir.parent / "models",
            Path("models"),
            Path("../models"),
            Path("./models"),
            Path("main/models"),
        ]

        logger.info("Searching for GGUF models...")
        for models_dir in possible_models_dirs:
            try:
                if not models_dir.exists():
                    continue

                # Find GGUF files
                gguf_files = list(models_dir.glob("*.gguf"))
                if gguf_files:
                    model_path = str(gguf_files[0].resolve())
                    logger.info(f"✅ Found GGUF model: {model_path}")
                    return model_path

            except Exception as e:
                logger.warning(f"Error checking {models_dir}: {e}")

        logger.error("❌ No GGUF files found!")
        return ""

    def _load_model(self):
        """Load GGUF model with GPU support"""
        if not HAS_LLAMA_CPP:
            logger.error("llama-cpp-python not installed")
            return

        if not self.model_path or not Path(self.model_path).exists():
            logger.error(f"GGUF model not found: {self.model_path}")
            return

        try:
            logger.info(f"Loading GGUF model: {self.model_path}")

            # GPU configuration
            if self.use_gpu and self.n_gpu_layers > 0:
                logger.info(f"🎮 GPU acceleration enabled")
                logger.info(f"🔧 GPU layers: {self.n_gpu_layers}")
            else:
                logger.info(f"🖥️  CPU-only mode")
                self.n_gpu_layers = 0

            # Detect chat format
            chat_format = self._detect_chat_format()
            logger.info(f"📝 Chat format: {chat_format}")

            # Load model with optimal settings
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,  # KEY: GPU acceleration
                verbose=False,
                use_mmap=self.use_mmap,
                use_mlock=self.use_mlock,
                chat_format=chat_format,
                seed=-1,
                n_batch=self.n_batch,
                f16_kv=self.f16_kv,
                # Additional GPU optimizations
                split_mode=1,  # Split layers across GPU and CPU
                main_gpu=0,  # Use first GPU
                tensor_split=None,  # Let llama.cpp decide
            )

            logger.info("✅ GGUF model loaded successfully")

            # Test the model
            try:
                test_response = self.llm("Hello", max_tokens=5, echo=False)
                logger.info("✅ Model test successful")

                # Log GPU usage if available
                if self.use_gpu and self.n_gpu_layers > 0:
                    logger.info("🎮 GPU acceleration active")
                else:
                    logger.info("🖥️  Running on CPU")

            except Exception as e:
                logger.warning(f"Model test failed: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            logger.error("Solutions:")
            logger.error("1. Check if the GGUF file is valid")
            logger.error("2. Try reducing n_gpu_layers if GPU issues")
            logger.error("3. Install GPU-enabled llama-cpp-python:")
            logger.error("   pip uninstall llama-cpp-python")
            logger.error(
                "   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            self.llm = None

    def _detect_chat_format(self) -> str:
        """Detect chat format from model name"""
        model_name = Path(self.model_path).name.lower()

        if "llama-2" in model_name or "llama2" in model_name:
            return "llama-2"
        elif "llama-3" in model_name or "llama3" in model_name or "llama-3.2" in model_name:
            return "llama-3"
        elif "mistral" in model_name:
            return "mistral-instruct"
        elif "vicuna" in model_name:
            return "vicuna"
        elif "alpaca" in model_name:
            return "alpaca"
        elif "qwen" in model_name:
            return "chatml"
        else:
            return "chatml"

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Generate response"""
        if not self.llm:
            return self._get_model_not_loaded_message()

        try:
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature

            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get('top_p', self.top_p),
                top_k=kwargs.get('top_k', self.top_k),
                repeat_penalty=kwargs.get('repeat_penalty', self.repeat_penalty),
                stop=kwargs.get('stop', ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:"]),
                echo=False,
                stream=False
            )

            if isinstance(response, dict):
                text = response["choices"][0]["text"] if "choices" in response else str(response)
            else:
                text = str(response)

            return text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"❌ Error generating response: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        if not self.llm:
            return self._get_model_not_loaded_message()

        try:
            if hasattr(self.llm, 'create_chat_completion'):
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    stop=kwargs.get('stop', ["</s>", "<|end|>", "<|eot_id|>"]),
                )
                return response['choices'][0]['message']['content']
            else:
                prompt = self._format_chat_prompt(messages)
                return self.generate(prompt, **kwargs)

        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return f"❌ Error generating chat response: {str(e)}"

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """Create chat completion in OpenAI format"""
        if not self.llm:
            return {"choices": [{"message": {"content": self._get_model_not_loaded_message()}}]}

        try:
            if hasattr(self.llm, 'create_chat_completion'):
                return self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    stop=kwargs.get('stop', ["</s>", "<|end|>", "<|eot_id|>"]),
                )
            else:
                prompt = self._format_chat_prompt(messages)
                response_text = self.generate(prompt, **kwargs)
                return {
                    "choices": [{
                        "message": {"content": response_text, "role": "assistant"}
                    }]
                }

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {"choices": [{"message": {"content": f"❌ Error: {str(e)}", "role": "assistant"}}]}

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into prompt string"""
        chat_format = self._detect_chat_format()

        if chat_format == "llama-3":
            prompt_parts = ["<|begin_of_text|>"]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(prompt_parts)
        else:
            # Fallback format
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            prompt_parts.append("<|im_start|>assistant\n")
            return "".join(prompt_parts)

    def _get_model_not_loaded_message(self) -> str:
        """Error message when model not loaded"""
        return f"""❌ GGUF model not loaded.

**Current settings:**
- Model path: `{self.model_path or 'Not found'}`
- GPU layers: {self.n_gpu_layers}
- GPU enabled: {self.use_gpu}

**To enable GPU acceleration:**
```
set LLAMA_N_GPU_LAYERS=32
set LLAMA_USE_GPU=true
```

**Or run the GPU configuration tool:**
```
python gpu_config.py
```"""

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": self.provider,
            "model_path": self.model_path,
            "is_loaded": self.llm is not None,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "use_gpu": self.use_gpu,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_format": self._detect_chat_format() if self.model_path else None,
            "gpu_acceleration": "Enabled" if self.use_gpu and self.n_gpu_layers > 0 else "Disabled",
        }

    def reload_model(self):
        """Reload the model"""
        if self.llm:
            del self.llm
            gc.collect()
        self._load_model()


# Factory function
def get_llm_provider():
    """Factory function for LLM provider"""
    return LLMProvider()
