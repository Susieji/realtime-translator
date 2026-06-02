"""Voice Activity Detection using FunASR FSMN-VAD for endpoint detection."""

import enum
import numpy as np
import torch
from typing import Optional, List

import config
from translator.messages import SentenceAudio


class VadState(enum.IntEnum):
    SILENCE = 0
    SPEECH = 1
    TRAILING_SILENCE = 2


class VadEngine:
    """FunASR FSMN-VAD with built-in endpoint detection.

    Uses a neural network (Deep-FSMN) trained on speech/silence classification,
    providing more intelligent sentence boundary detection than simple energy
    or probability thresholds.
    """

    def __init__(self):
        from funasr import AutoModel
        self._model = AutoModel(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            disable_update=True,
            disable_log=True,
        )
        self._cache = {}
        self._is_speaking = False

    def process_chunk(self, samples: np.ndarray, is_final: bool = False) -> List[List[int]]:
        """Process audio chunk through FSMN-VAD.

        Returns list of segments: [[start_ms, end_ms], ...]
        - [start_ms, -1] means speech started but not yet ended
        - [-1, end_ms] means speech ended (continuation of previous start)
        - [start_ms, end_ms] means complete segment detected
        - [] means no event
        """
        result = self._model.generate(
            input=samples,
            cache=self._cache,
            is_final=is_final,
            chunk_size=config.FSMN_VAD_CHUNK_MS,
            max_end_silence_time=config.FSMN_VAD_MAX_END_SILENCE_MS,
        )
        if result and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                segments = item.get("value", [])
            elif hasattr(item, "value"):
                segments = item.value
            else:
                segments = []
            return segments if segments else []
        return []

    def reset(self):
        """Reset VAD state for new session."""
        self._cache = {}
        self._is_speaking = False


class SentenceManager:
    """Collects audio chunks and detects sentence boundaries using FSMN-VAD endpoint detection.

    Unlike the previous SileroVAD approach (fixed 600ms silence threshold),
    FSMN-VAD uses a trained neural model to detect speech start/end points,
    providing more semantically aware sentence boundaries.
    """

    # ~320ms of audio kept before the detected speech onset. FSMN-VAD reports
    # a start point that can lag the true onset; a wider pre-buffer avoids
    # clipping the first syllable, which Paraformer is especially sensitive to.
    _PRE_BUFFER_CHUNKS = 10

    def __init__(self, vad: VadEngine):
        self._vad = vad
        self._state = VadState.SILENCE
        self._sentence_chunks: List[np.ndarray] = []
        self._pre_buffer: List[np.ndarray] = []
        self._sentence_start_ms = 0
        self._current_ms = 0
        self._speech_chunk_count = 0

    @property
    def state(self) -> VadState:
        return self._state

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> Optional[SentenceAudio]:
        """Feed a 32ms audio chunk. Returns SentenceAudio when endpoint detected."""
        self._current_ms = timestamp_ms
        segments = self._vad.process_chunk(samples)

        if self._state == VadState.SILENCE:
            self._pre_buffer.append(samples.copy())
            if len(self._pre_buffer) > self._PRE_BUFFER_CHUNKS:
                self._pre_buffer.pop(0)

            for seg in segments:
                start_ms, end_ms = seg[0], seg[1]
                if start_ms >= 0:
                    self._state = VadState.SPEECH
                    self._sentence_chunks = list(self._pre_buffer)
                    self._sentence_start_ms = start_ms
                    self._pre_buffer = []
                    self._speech_chunk_count = len(self._sentence_chunks)

                    if end_ms >= 0:
                        return self._flush(end_ms)
            return None

        elif self._state == VadState.SPEECH:
            self._sentence_chunks.append(samples.copy())
            self._speech_chunk_count += 1

            for seg in segments:
                start_ms, end_ms = seg[0], seg[1]
                if end_ms >= 0:
                    return self._flush(end_ms)
                if start_ms >= 0 and end_ms < 0:
                    pass
            return None

        elif self._state == VadState.TRAILING_SILENCE:
            self._sentence_chunks.append(samples.copy())

            for seg in segments:
                start_ms, end_ms = seg[0], seg[1]
                if end_ms >= 0:
                    return self._flush(end_ms)
            return None

        return None

    def _flush(self, end_ms: int) -> Optional[SentenceAudio]:
        """Flush accumulated audio as a complete sentence."""
        speech_ms = self._speech_chunk_count * config.BLOCK_MS
        if speech_ms < config.VAD_MIN_SPEECH_MS:
            self._reset()
            return None

        all_samples = np.concatenate(self._sentence_chunks)
        result = SentenceAudio(
            samples=all_samples,
            start_ms=self._sentence_start_ms,
            end_ms=end_ms,
        )
        self._reset()
        return result

    def _reset(self):
        self._state = VadState.SILENCE
        self._sentence_chunks = []
        self._pre_buffer = []
        self._speech_chunk_count = 0
        self._vad.reset()
