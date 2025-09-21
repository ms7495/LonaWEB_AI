# core/embeddings.py - Updated for local models
import logging
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSystem:
    """Enhanced embedding system with local model support"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _find_local_model(self, model_name: str) -> str:
        """Find local model in models/embeddings/ folder"""

        # Check if it's already a local path
        if Path(model_name).exists():
            return model_name

        # Extract model name from huggingface path
        if "/" in model_name:
            model_short_name = model_name.split("/")[-1]
        else:
            model_short_name = model_name

        # Look for local model in different locations
        possible_paths = [
            f"models/embeddings/{model_name}",  # Full HF path
            f"models/embeddings/{model_short_name}",  # Short name
            f"./models/embeddings/{model_name}",
            f"./models/embeddings/{model_short_name}",
            f"../models/embeddings/{model_name}",  # If running from subdirectory
            f"../models/embeddings/{model_short_name}",
        ]

        for path in possible_paths:
            if Path(path).exists() and (Path(path) / "config.json").exists():
                logger.info(f"Found local embedding model at: {path}")
                return str(Path(path).absolute())

        return None

    def _load_model(self):
        """Load the embedding model - prefer local, fallback to download"""
        try:
            # First try to find local model
            local_path = self._find_local_model(self.model_name)

            if local_path:
                logger.info(f"Loading local embedding model from: {local_path}")
                self.model = SentenceTransformer(local_path)
                self.model_path = local_path
                self.is_local = True
            else:
                logger.info(f"Local model not found, downloading: {self.model_name}")
                logger.info("To use local models, download to models/embeddings/ folder")
                self.model = SentenceTransformer(self.model_name)
                self.model_path = self.model_name
                self.is_local = False

            # Test the model
            _ = self.model.encode(["test"], normalize_embeddings=True)
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Local: {self.is_local}")
            logger.info(f"   Path: {self.model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode single text"""
        return self.encode([text], normalize=normalize)[0]

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        test_embedding = self.encode(["test"])
        return test_embedding.shape[1]

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": getattr(self, 'model_path', 'Unknown'),
            "is_local": getattr(self, 'is_local', False),
            "dimension": self.get_dimension(),
            "device": str(self.model.device) if hasattr(self.model, 'device') else 'Unknown'
        }
