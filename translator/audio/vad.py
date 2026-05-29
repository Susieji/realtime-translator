"""Voice Activity Detection using SileroVAD."""

import enum
import numpy as np
import torch
from typing import Optional

import config
from translator.messages import SentenceAudio


class VadState(enum.IntEnum):
    SILENCE = 0
    SPEECH = 1
    TRAILING_SILENCE = 2


class VadEngine:
    """SileroVAD wrapper. Feed 32ms chunks, get speech probability."""

    def __init__(self):
        self._model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self._model.eval()

    def process_chunk(self, samples: np.ndarray) -> float:
        tensor = torch.from_numpy(samples).float()
        with torch.no_grad():
            prob = self._model(tensor, config.SAMPLE_RATE).item()
        return prob

    def reset(self):
        self._model.reset_states()


class SentenceManager:
    """Collects audio chunks and detects sentence boundaries using VAD."""

    _PRE_BUFFER_CHUNKS = 5

    def __init__(self, vad: VadEngine):
        self._vad = vad
        self._state = VadState.SILENCE
        self._sentence_chunks = []
        self._pre_buffer = []
        self._silence_count = 0
        self._speech_count = 0
        self._sentence_start_ms = 0
        self._current_ms = 0

    @property
    def state(self) -> VadState:
        return self._state

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> Optional[SentenceAudio]:
        """Feed a 32ms audio chunk. Returns SentenceAudio when sentence boundary detected."""
        prob = self._vad.process_chunk(samples)
        is_speech = prob >= config.VAD_THRESHOLD
        self._current_ms = timestamp_ms

        if self._state == VadState.SILENCE:
            if is_speech:
                self._state = VadState.SPEECH
                self._sentence_chunks = list(self._pre_buffer) + [samples.copy()]
                self._speech_count = 1
                self._sentence_start_ms = timestamp_ms - len(self._pre_buffer) * config.BLOCK_MS
                self._pre_buffer = []
            else:
                self._pre_buffer.append(samples.copy())
                if len(self._pre_buffer) > self._PRE_BUFFER_CHUNKS:
                    self._pre_buffer.pop(0)
            return None

        elif self._state == VadState.SPEECH:
            self._sentence_chunks.append(samples.copy())
            if is_speech:
                self._speech_count += 1
            else:
                self._state = VadState.TRAILING_SILENCE
                self._silence_count = 1
            return None

        elif self._state == VadState.TRAILING_SILENCE:
            self._sentence_chunks.append(samples.copy())
            if is_speech:
                self._state = VadState.SPEECH
                self._silence_count = 0
            else:
                self._silence_count += 1
                silence_ms = self._silence_count * config.BLOCK_MS
                if silence_ms >= config.VAD_MIN_SILENCE_MS:
                    return self._flush()
            return None

        return None

    def _flush(self) -> Optional[SentenceAudio]:
        speech_ms = self._speech_count * config.BLOCK_MS
        if speech_ms < config.VAD_MIN_SPEECH_MS:
            self._reset()
            return None

        all_samples = np.concatenate(self._sentence_chunks)
        end_ms = self._current_ms + config.BLOCK_MS
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
        self._silence_count = 0
        self._speech_count = 0
        self._vad.reset()
