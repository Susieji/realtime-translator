"""Opus-MT translation using MarianMT from Hugging Face."""

import os
import logging
import torch
from typing import Dict, Tuple

import config
from translator.translation.base import BaseTranslator

logger = logging.getLogger(__name__)


class OpusMTTranslator(BaseTranslator):
    """Opus-MT translation model."""

    def __init__(self):
        self._models: Dict[Tuple[str, str], object] = {}
        self._tokenizers: Dict[Tuple[str, str], object] = {}

    def load(self, src_lang: str = "zh", tgt_lang: str = "en"):
        """Load translation model for a specific language pair."""
        from transformers import MarianMTModel, MarianTokenizer

        os.environ["HF_HUB_OFFLINE"] = "1"

        model_key = (src_lang, tgt_lang)
        if model_key in self._models:
            return

        model_name = config.TRANSLATION_MODELS.get(model_key)
        if not model_name:
            raise ValueError(f"No model for {src_lang} -> {tgt_lang}")

        logger.info(f"Loading Opus-MT: {model_name}")

        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
        model = model.to("cpu")
        model.eval()

        self._tokenizers[model_key] = tokenizer
        self._models[model_key] = model
        logger.info(f"Opus-MT loaded: {model_key}")

    def translate(self, text: str, src_lang: str = "zh", tgt_lang: str = "en") -> str:
        """Translate text between languages."""
        if not text or not text.strip():
            return ""

        model_key = (src_lang, tgt_lang)
        if model_key not in self._models:
            self.load(src_lang, tgt_lang)

        model = self._models[model_key]
        tokenizer = self._tokenizers[model_key]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
