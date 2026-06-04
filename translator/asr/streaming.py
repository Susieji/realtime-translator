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
        # Punctuation restorer for the offline fallback's raw output. Lazily
        # loaded on first use; see translator/asr/punctuation.py.
        from translator.asr.punctuation import Punctuator
        self._punctuator = Punctuator()

    def load(self):
        """Load the streaming (online) Paraformer model.

        Only the online model — needed for live partials — is loaded eagerly.
        The offline model is a fallback used solely when Qwen3-ASR is
        unavailable, so it is loaded lazily on first use (see
        ``finalize_offline``). On machines where Qwen3-ASR is present (the
        common case) this avoids holding a second ~1GB Paraformer in memory and
        shortens startup.
        """
        import warnings
        import logging
        warnings.filterwarnings('ignore')
        logging.getLogger('modelscope').setLevel(logging.ERROR)
        logging.getLogger('funasr').setLevel(logging.ERROR)

        from funasr import AutoModel
        from translator.utils.models import resolve_local_model, is_local_path

        # Prefer the local cache dir so startup doesn't stall on ModelScope's
        # revision check (and works offline once the model is cached). Only
        # pass model_revision when loading by ID — a local path has none.
        online_ref = resolve_local_model(config.PARAFORMER_ONLINE_MODEL)
        online_kw = dict(model=online_ref, disable_update=True, disable_log=True)
        if not is_local_path(online_ref):
            online_kw["model_revision"] = "v2.0.4"
        self._online_model = AutoModel(**online_kw)

        # Warm the punctuation model on a background thread so live partials get
        # punctuated as soon as possible, without blocking startup or the
        # capture loop. Until it's ready, partials show as raw text.
        self._punctuator.preload()

    def _ensure_offline(self):
        """Lazily load the offline Paraformer model on first fallback use."""
        if self._offline_model is not None:
            return
        with self._lock:
            if self._offline_model is not None:
                return
            from funasr import AutoModel
            from translator.utils.models import resolve_local_model
            self._offline_model = AutoModel(
                model=resolve_local_model(config.PARAFORMER_OFFLINE_MODEL),
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
            chunk_size=config.PARAFORMER_CHUNK_LOOK,
            encoder_chunk_look_back=config.PARAFORMER_ENCODER_LOOK_BACK,
            decoder_chunk_look_back=config.PARAFORMER_DECODER_LOOK_BACK,
        )

        text = self._extract_text(result)
        if text:
            self._accumulated_text += text
            # Punctuate the live partial. Non-blocking: until the ct-punc model
            # finishes loading this returns the raw text, so the hot path is
            # never stalled by the model download/load.
            display = self._punctuator.restore(self._accumulated_text)
            return AsrResult(
                text=display,
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
                chunk_size=config.PARAFORMER_CHUNK_LOOK,
                encoder_chunk_look_back=config.PARAFORMER_ENCODER_LOOK_BACK,
                decoder_chunk_look_back=config.PARAFORMER_DECODER_LOOK_BACK,
            )
            tail_text = self._extract_text(result)
            if tail_text:
                self._accumulated_text += tail_text

        self._cache = {}
        final_text = self._accumulated_text
        self._accumulated_text = ""
        if final_text:
            # Final partial of the sentence: punctuate (non-blocking) for a
            # clean last update before the accurate-ASR result replaces it.
            final_text = self._punctuator.restore(final_text)
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
        self._ensure_offline()
        samples = normalize_audio(sentence.samples)
        result = self._offline_model.generate(input=samples)
        text = self._extract_text(result)
        if text:
            # Paraformer returns unpunctuated text; restore punctuation so the
            # fallback reads like a finished sentence (Qwen3 already does this).
            # block=True: this runs on Thread 2 (off the capture loop), so it
            # can afford to wait for the ct-punc model to finish loading.
            text = self._punctuator.restore(text, block=True)
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
