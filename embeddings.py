"""Image and text embeddings using SigLIP (768-dim)."""
import io
import logging
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipModel, SiglipTokenizer

logger = logging.getLogger(__name__)

# Model: google/siglip-base-patch16-384 outputs 768-dim
MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

_model = None
_image_processor = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return _device


def _load_model():
    global _model, _image_processor, _tokenizer
    if _model is None:
        logger.info("Loading SigLIP model %s...", MODEL_NAME)
        _image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        _tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
        _model = SiglipModel.from_pretrained(MODEL_NAME)
        _model.to(_get_device())
        _model.eval()
    return _model, _image_processor, _tokenizer


def get_image_embedding(image_url: str) -> Optional[list[float]]:
    """
    Generate 768-dim embedding for an image using SigLIP.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.zara.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        resp = requests.get(image_url, timeout=15, headers=headers)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image %s: %s", image_url, e)
        return None

    model, image_processor, _ = _load_model()
    device = _get_device()

    try:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # SigLIP returns BaseModelOutputWithPooling; extract pooled tensor
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        embedding = emb_tensor.cpu().float().numpy().flatten().tolist()
        if len(embedding) != EMBEDDING_DIM:
            logger.warning("Unexpected embedding dim %d", len(embedding))
        return embedding
    except Exception as e:
        logger.warning("Failed to embed image: %s", e)
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    """
    Generate 768-dim embedding for text using SigLIP text encoder.
    """
    if not text or not text.strip():
        return None

    model, _, tokenizer = _load_model()
    device = _get_device()

    try:
        inputs = tokenizer(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        # SigLIP returns BaseModelOutputWithPooling; extract pooled tensor
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        embedding = emb_tensor.cpu().float().numpy().flatten().tolist()
        return embedding
    except Exception as e:
        logger.warning("Failed to embed text: %s", e)
        return None
