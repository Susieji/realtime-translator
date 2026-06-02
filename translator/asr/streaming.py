"""Streaming ASR using Paraformer Online + Offline fallback."""

import numpy as np
import threading
from typing import Optional

import config
from translator.messages import AsrResult, SentenceAudio


class StreamingAsr:
    """Paraformer Online for streaming partial + Offline as fallback for final."""

    def __init__(self):
        self._online_model = None
        self._offline_model = None
        self._cache = {}
        self._chunk_buffer = np.array([], dtype=np.float32)
        self._accumulated_text = ""
        self._sentence_start_ms = 0
        self._lock = threading.Lock()

    def load(self):
        """Load Paraformer models."""
        import warnings
        import logging
        warnings.filterwarnings('ignore')
        logging.getLogger('modelscope').setLevel(logging.ERROR)
        logging.getLogger('funasr').setLevel(logging.ERROR)

        from funasr import AutoModel

        self._online_model = AutoModel(
            model=config.PARAFORMER_ONLINE_MODEL,
            model_revision='v2.0.4',
            disable_update=True,
            disable_log=True,
        )

        self._offline_model = AutoModel(
            model=config.PARAFORMER_OFFLINE_MODEL,
            disable_update=True,
            disable_log=True,
        )

    def start_sentence(self, timestamp_ms: int):
        """Signal the start of a new sentence."""
        with self._lock:
            self._cache = {}
            self._chunk_buffer = np.array([], dtype=np.float32)
            self._accumulated_text = ""
            self._sentence_start_ms = timestamp_ms

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> Optional[AsrResult]:
        """Feed audio for streaming partial output."""
        with self._lock:
            self._chunk_buffer = np.concatenate([self._chunk_buffer, samples])

            if len(self._chunk_buffer) < config.PARAFORMER_CHUNK_SIZE:
                return None

            chunk = self._chunk_buffer[:config.PARAFORMER_CHUNK_SIZE]
            self._chunk_buffer = self._chunk_buffer[config.PARAFORMER_CHUNK_SIZE:]

        result = self._online_model.generate(
            input=chunk,
            cache=self._cache,
            is_final=False,
            chunk_size=[0, 10, 5],
        )

        text = self._extract_text(result)
        if text:
            self._accumulated_text += text
            return AsrResult(
                text=self._accumulated_text,
                is_final=False,
                start_ms=self._sentence_start_ms,
                end_ms=timestamp_ms,
            )
        return None

    def end_sentence(self, timestamp_ms: int) -> Optional[AsrResult]:
        """End streaming session. Returns accumulated partial text.

        Flushes any residual audio (shorter than one streaming chunk) through
        the online model with is_final=True so the tail of the sentence is not
        dropped. Without this, up to PARAFORMER_CHUNK_MS of trailing audio per
        sentence would never be transcribed.
        """
        with self._lock:
            residual = self._chunk_buffer
            self._chunk_buffer = np.array([], dtype=np.float32)

        if len(residual) > 0:
            result = self._online_model.generate(
                input=residual,
                cache=self._cache,
                is_final=True,
                chunk_size=[0, 10, 5],
            )
            tail_text = self._extract_text(result)
            if tail_text:
                self._accumulated_text += tail_text

        self._cache = {}
        final_text = self._accumulated_text
        self._accumulated_text = ""
        if final_text:
            return AsrResult(
                text=final_text,
                is_final=False,
                start_ms=self._sentence_start_ms,
                end_ms=timestamp_ms,
            )
        return None

    def finalize_offline(self, sentence: SentenceAudio) -> Optional[AsrResult]:
        """Run Paraformer Offline on complete sentence. Used as Qwen3 fallback."""
        from translator.utils.audio import normalize_audio
        samples = normalize_audio(sentence.samples)
        result = self._offline_model.generate(input=samples)
        text = self._extract_text(result)
        if text:
            return AsrResult(
                text=text,
                is_final=True,
                start_ms=sentence.start_ms,
                end_ms=sentence.end_ms,
            )
        return None

    def _extract_text(self, result) -> str:
        if not result:
            return ""
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                return item.get("text", "")
            if hasattr(item, 'text'):
                return item.text
        return ""
