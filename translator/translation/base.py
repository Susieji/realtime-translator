"""Base interface for translation models."""

from abc import ABC, abstractmethod


class BaseTranslator(ABC):
    """Translation model interface."""

    @abstractmethod
    def load(self, src_lang: str, tgt_lang: str):
        """Load model for a language pair."""
        ...

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text."""
        ...
