"""Live translation pipeline - main orchestrator.

Architecture:
  Mic → VAD → Paraformer Online (partial) → WebSocket
                └→ Qwen3-ASR (final) → WebSocket → Opus-MT → WebSocket → MeloTTS → WebSocket
"""

import sys
import time
import threading
import queue
import logging
import numpy as np
from pathlib import Path

import config
from translator.audio.capture import AudioCapture
from translator.audio.vad import VadEngine, SentenceManager, VadState
from translator.asr.streaming import StreamingAsr
from translator.asr.accurate import AccurateAsr
from translator.translation.opus_mt import OpusMTTranslator
from translator.tts.melotts import MeloTTSModel
from translator.server.ws_server import WsServer
from translator.messages import SentenceAudio, AsrResult
from translator.utils.audio import save_audio

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class LivePipeline:
    """Main orchestrator. Wires audio → ASR → translation → TTS with threading."""

    def __init__(self, src_lang="zh", tgt_lang="en", enable_ws=True):
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._enable_ws = enable_ws

        # Components
        self._capture = AudioCapture()
        self._vad = VadEngine()
        self._sentence_mgr = SentenceManager(self._vad)
        self._streaming_asr = StreamingAsr()
        self._accurate_asr = AccurateAsr()
        self._translator = OpusMTTranslator()
        self._tts = MeloTTSModel()
        self._ws_server = WsServer() if enable_ws else None

        # Queues for thread communication
        self._sentence_queue = queue.Queue()  # ASR final processing
        self._translation_queue = queue.Queue()  # Translation
        self._tts_queue = queue.Queue()  # TTS

        # State
        self._running = False
        self._in_sentence = False
        self._current_partial = ""
        self._sentence_count = 0
        self._chunks_buf = []
        self._chunks_lock = threading.Lock()

    def load_models(self):
        """Load all models."""
        print("  [1/5] Loading FSMN-VAD (endpoint detection)...")
        # VAD loaded in __init__ via FunASR AutoModel
        print("        [OK]")

        print("  [2/5] Loading Paraformer streaming (CPU)...")
        self._streaming_asr.load()
        print("        [OK]")

        print("  [3/5] Loading Qwen3-ASR (iGPU)...")
        try:
            self._accurate_asr.load()
            print("        [OK]")
        except Exception as e:
            print(f"        [SKIP] {e}")
            print("        Will use Paraformer Offline as fallback")

        print(f"  [4/5] Loading Opus-MT ({self._src_lang}→{self._tgt_lang})...")
        try:
            self._translator.load(self._src_lang, self._tgt_lang)
            print("        [OK]")
        except Exception as e:
            print(f"        [FAIL] {e}")
            raise

        print(f"  [5/5] Loading MeloTTS ({self._tgt_lang})...")
        try:
            self._tts.load(self._tgt_lang)
            print("        [OK]")
        except Exception as e:
            print(f"        [SKIP] {e}")
            self._tts = None

    def start(self):
        """Start all threads."""
        self._running = True

        if self._ws_server:
            self._ws_server.start()

        # Background threads
        threading.Thread(target=self._sentence_processor_loop, daemon=True).start()
        threading.Thread(target=self._translation_loop, daemon=True).start()
        threading.Thread(target=self._tts_loop, daemon=True).start()

        # Audio capture
        self._capture.add_listener(self._on_audio)
        self._capture.start()

        self._main_loop()

    def stop(self):
        self._running = False
        self._capture.stop()
        if self._ws_server:
            self._ws_server.stop()

    def _on_audio(self, samples: np.ndarray, timestamp_ms: int):
        with self._chunks_lock:
            self._chunks_buf.append((samples.copy(), timestamp_ms))

    def _main_loop(self):
        """Main thread: VAD + Paraformer Online streaming."""
        try:
            while self._running:
                to_process = []
                with self._chunks_lock:
                    to_process = self._chunks_buf[:]
                    self._chunks_buf.clear()

                for samples, ts in to_process:
                    self._process_chunk(samples, ts)

                if not to_process:
                    time.sleep(0.005)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _process_chunk(self, samples: np.ndarray, ts: int):
        """Process a single audio chunk through VAD + streaming ASR."""
        sentence_audio = self._sentence_mgr.feed(samples, ts)
        state = self._sentence_mgr.state

        if state == VadState.SPEECH or state == VadState.TRAILING_SILENCE:
            if not self._in_sentence:
                self._in_sentence = True
                self._streaming_asr.start_sentence(ts)
                self._current_partial = ""

            asr_result = self._streaming_asr.feed(samples, ts)
            if asr_result and asr_result.text:
                self._current_partial = asr_result.text
                self._display_partial(asr_result.text)
                if self._ws_server:
                    self._ws_server.push_asr(asr_result.text, is_final=False)

        if sentence_audio is not None:
            final_partial = self._streaming_asr.end_sentence(ts)
            if final_partial and final_partial.text:
                self._current_partial = final_partial.text

            self._sentence_count += 1
            self._in_sentence = False

            self._sentence_queue.put((sentence_audio, self._current_partial, self._sentence_count))
            self._current_partial = ""

    def _sentence_processor_loop(self):
        """Background: Qwen3-ASR (or Paraformer Offline fallback) → push final."""
        while self._running:
            try:
                item = self._sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            sentence_audio, partial_text, sentence_num = item
            final_text = partial_text

            # Try Qwen3-ASR first (most accurate)
            if self._accurate_asr.available:
                result = self._accurate_asr.transcribe(sentence_audio)
                if result and result.text:
                    final_text = result.text
            else:
                # Fallback: Paraformer Offline
                result = self._streaming_asr.finalize_offline(sentence_audio)
                if result and result.text:
                    final_text = result.text

            # Push final ASR to UI
            self._display_final(final_text, sentence_num)
            if self._ws_server:
                self._ws_server.push_asr(final_text, is_final=True)

            # Send to translation
            self._translation_queue.put((final_text, sentence_num))

    def _translation_loop(self):
        """Background: Opus-MT translation."""
        while self._running:
            try:
                item = self._translation_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, sentence_num = item
            try:
                translated = self._translator.translate(text, self._src_lang, self._tgt_lang)
                if translated:
                    self._display_translation(translated, sentence_num)
                    if self._ws_server:
                        self._ws_server.push_translation(
                            translated, self._src_lang, self._tgt_lang, sentence_num
                        )
                    # Send to TTS
                    self._tts_queue.put((translated, sentence_num))
            except Exception as e:
                logger.error(f"Translation failed: {e}")

    def _tts_loop(self):
        """Background: MeloTTS synthesis."""
        while self._running:
            try:
                item = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, sentence_num = item
            if self._tts is None:
                continue

            try:
                audio = self._tts.synthesize(text, language=self._tgt_lang, speed=config.TTS_SPEED)
                if len(audio) > 0:
                    filename = f"sentence_{sentence_num:04d}.wav"
                    output_path = str(OUTPUT_DIR / filename)
                    save_audio(audio, output_path, self._tts.sample_rate)
                    if self._ws_server:
                        self._ws_server.push_tts(filename, sentence_num)
                    self._display_tts(output_path, sentence_num)
            except Exception as e:
                logger.error(f"TTS failed: {e}")

    def _display_partial(self, text: str):
        sys.stdout.write(f"\r  [partial] {text}    ")
        sys.stdout.flush()

    def _display_final(self, text: str, sentence_num: int):
        sys.stdout.write(f"\r  #{sentence_num} [ASR] {text}\n")
        sys.stdout.flush()

    def _display_translation(self, text: str, sentence_num: int):
        sys.stdout.write(f"  #{sentence_num} [TRN] {text}\n")
        sys.stdout.flush()

    def _display_tts(self, path: str, sentence_num: int):
        sys.stdout.write(f"  #{sentence_num} [TTS] {path}\n")
        sys.stdout.flush()
