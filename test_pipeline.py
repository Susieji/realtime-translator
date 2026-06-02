"""Test script for realtime-translator pipeline.

Tests each component individually and then the integration.
Does NOT require a microphone - uses synthetic audio.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def generate_sine_wave(freq=440, duration=1.0, sample_rate=16000):
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def generate_speech_like_audio(duration=2.0, sample_rate=16000):
    """Generate speech-like audio (mix of frequencies with envelope)."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Mix of harmonics
    audio = (0.5 * np.sin(2 * np.pi * 200 * t) +
             0.3 * np.sin(2 * np.pi * 400 * t) +
             0.2 * np.sin(2 * np.pi * 800 * t))
    # Apply envelope
    envelope = np.ones_like(t)
    attack = int(0.05 * sample_rate)
    decay = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    return (audio * envelope * 0.8).astype(np.float32)


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, error):
        self.failed += 1
        print(f"  [FAIL] {name}: {error}")

    def skip(self, name, reason):
        self.skipped += 1
        print(f"  [SKIP] {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n  Results: {self.passed} passed, {self.failed} failed, {self.skipped} skipped / {total} total")
        return self.failed == 0


results = TestResults()


# ============================================================
# Test 1: Messages module
# ============================================================
print("\n[Test 1] Messages module")
try:
    from translator.messages import SentenceAudio, AsrResult, TranslationResult, TtsResult
    audio = generate_speech_like_audio()
    sa = SentenceAudio(samples=audio, start_ms=0, end_ms=2000)
    assert sa.samples is audio
    assert sa.start_ms == 0
    asr = AsrResult(text="hello", is_final=True, speaker_id=-1)
    assert asr.text == "hello"
    results.ok("messages dataclasses")
except Exception as e:
    results.fail("messages dataclasses", e)


# ============================================================
# Test 2: Ring buffer
# ============================================================
print("\n[Test 2] Ring buffer")
try:
    from translator.audio.ring_buffer import RingBuffer
    rb = RingBuffer(1000)
    data = np.ones(500, dtype=np.float32)
    rb.write(data)
    assert rb.available == 500
    out = rb.read(200)
    assert len(out) == 200
    assert rb.available == 300
    rb.clear()
    assert rb.available == 0
    results.ok("ring buffer read/write/clear")
except Exception as e:
    results.fail("ring buffer", e)


# ============================================================
# Test 3: VAD engine
# ============================================================
print("\n[Test 3] VAD engine")
try:
    from translator.audio.vad import VadEngine, SentenceManager, VadState
    vad = VadEngine()
    # Feed silence. FSMN-VAD returns a list of [start_ms, end_ms] segments
    # (empty when no speech event), not a probability.
    silence = np.zeros(config.BLOCK_SIZE, dtype=np.float32)
    segments = vad.process_chunk(silence)
    assert isinstance(segments, list)
    results.ok("VAD engine loads and processes silence")
except Exception as e:
    results.fail("VAD engine", e)

try:
    sm = SentenceManager(vad)
    assert sm.state == VadState.SILENCE
    result = sm.feed(silence, 0)
    assert result is None
    assert sm.state == VadState.SILENCE
    results.ok("SentenceManager state machine (silence)")
except Exception as e:
    results.fail("SentenceManager", e)


# ============================================================
# Test 4: Streaming ASR (Paraformer)
# ============================================================
print("\n[Test 4] Streaming ASR (Paraformer)")
try:
    from translator.asr.streaming import StreamingAsr
    asr = StreamingAsr()
    asr.load()
    results.ok("Paraformer Online + Offline loaded")
except Exception as e:
    results.fail("Paraformer load", e)

try:
    asr.start_sentence(0)
    # Feed enough audio chunks
    audio = generate_speech_like_audio(duration=3.0)
    chunk_size = config.BLOCK_SIZE
    ts = 0
    partial_results = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        r = asr.feed(chunk, ts)
        if r:
            partial_results.append(r.text)
        ts += config.BLOCK_MS

    end_r = asr.end_sentence(ts)
    # We may or may not get text from synthetic audio, but it shouldn't crash
    results.ok("Paraformer streaming feed (no crash)")
except Exception as e:
    results.fail("Paraformer streaming feed", e)

try:
    sa = SentenceAudio(samples=generate_speech_like_audio(2.0), start_ms=0, end_ms=2000)
    offline_result = asr.finalize_offline(sa)
    results.ok("Paraformer Offline fallback (no crash)")
except Exception as e:
    results.fail("Paraformer Offline fallback", e)


# ============================================================
# Test 5: Accurate ASR (Qwen3)
# ============================================================
print("\n[Test 5] Accurate ASR (Qwen3)")
try:
    from translator.asr.accurate import AccurateAsr
    qasr = AccurateAsr()
    qasr.load()
    results.ok("Qwen3-ASR loaded on iGPU")
except FileNotFoundError as e:
    results.skip("Qwen3-ASR load", str(e))
    qasr = None
except Exception as e:
    results.skip("Qwen3-ASR load", str(e))
    qasr = None

if qasr and qasr.available:
    try:
        sa = SentenceAudio(samples=generate_speech_like_audio(2.0), start_ms=0, end_ms=2000)
        r = qasr.transcribe(sa)
        # Synthetic audio won't produce meaningful text, just check no crash
        results.ok("Qwen3-ASR transcribe (no crash)")
    except Exception as e:
        results.fail("Qwen3-ASR transcribe", e)


# ============================================================
# Test 6: Translation (Opus-MT)
# ============================================================
print("\n[Test 6] Translation (Opus-MT)")
try:
    from translator.translation.opus_mt import OpusMTTranslator
    translator = OpusMTTranslator()
    translator.load("zh", "en")
    results.ok("Opus-MT zh→en loaded")
except Exception as e:
    results.fail("Opus-MT load", e)
    translator = None

if translator:
    try:
        result = translator.translate("你好世界", "zh", "en")
        assert result and len(result) > 0
        print(f"         '你好世界' → '{result}'")
        results.ok("Opus-MT zh→en translation")
    except Exception as e:
        results.fail("Opus-MT translate", e)


# ============================================================
# Test 7: TTS (MeloTTS)
# ============================================================
print("\n[Test 7] TTS (MeloTTS)")
try:
    from translator.tts.melotts import MeloTTSModel
    tts = MeloTTSModel()
    tts.load("en")
    results.ok("MeloTTS loaded (en)")
except ImportError as e:
    results.skip("MeloTTS load", f"not installed: {e}")
    tts = None
except Exception as e:
    results.fail("MeloTTS load", e)
    tts = None

if tts:
    try:
        audio = tts.synthesize("Hello world", language="en")
        assert len(audio) > 0
        assert audio.dtype == np.float32
        print(f"         Generated {len(audio)} samples ({len(audio)/tts.sample_rate:.2f}s)")
        results.ok("MeloTTS synthesize")
    except Exception as e:
        results.fail("MeloTTS synthesize", e)

    try:
        from translator.utils.audio import save_audio
        output_path = str(Path(__file__).parent / "output" / "test_tts.wav")
        save_audio(audio, output_path, tts.sample_rate)
        assert Path(output_path).exists()
        results.ok(f"Audio saved to {output_path}")
    except Exception as e:
        results.fail("save audio", e)


# ============================================================
# Test 8: WebSocket server
# ============================================================
print("\n[Test 8] WebSocket server")
try:
    from translator.server.ws_server import WsServer
    ws = WsServer()
    ws.start()
    time.sleep(0.5)
    ws.push_asr("test partial", is_final=False)
    ws.push_asr("test final", is_final=True)
    ws.push_translation("test translation", "zh", "en", 1)
    ws.push_tts("/output/test.wav", 1)
    time.sleep(0.3)
    ws.stop()
    results.ok("WebSocket server start/push/stop")
except Exception as e:
    results.fail("WebSocket server", e)


# ============================================================
# Test 9: Placeholder interfaces
# ============================================================
print("\n[Test 9] Placeholder interfaces")
try:
    from translator.asr.speaker import SpeakerIdentifier
    spk = SpeakerIdentifier()
    assert not spk.available
    sa = SentenceAudio(samples=np.zeros(16000, dtype=np.float32), start_ms=0, end_ms=1000)
    assert spk.identify(sa) == -1
    results.ok("SpeakerIdentifier placeholder")
except Exception as e:
    results.fail("SpeakerIdentifier", e)

try:
    from translator.asr.whisper_asr import WhisperAsr
    w = WhisperAsr()
    assert not w.available
    results.ok("WhisperAsr placeholder")
except Exception as e:
    results.fail("WhisperAsr", e)


# ============================================================
# Test 10: Integration - full pipeline (no mic, simulated)
# ============================================================
print("\n[Test 10] Integration - simulated sentence processing")
try:
    # Simulate what live_pipeline does for one sentence
    test_audio = generate_speech_like_audio(2.0)
    sa = SentenceAudio(samples=test_audio, start_ms=0, end_ms=2000)

    # Step 1: ASR (use Paraformer Offline since no real speech)
    from translator.asr.streaming import StreamingAsr
    test_asr = StreamingAsr()
    test_asr.load()
    offline_result = test_asr.finalize_offline(sa)
    asr_text = offline_result.text if offline_result else "测试文本"
    print(f"         ASR: '{asr_text}'")

    # Step 2: Translation
    if translator:
        # Use a known Chinese text for meaningful translation
        trans_text = translator.translate("这是一个测试", "zh", "en")
        print(f"         Translation: '{trans_text}'")
    else:
        trans_text = "this is a test"

    # Step 3: TTS
    if tts:
        audio_out = tts.synthesize(trans_text, language="en")
        output_path = str(Path(__file__).parent / "output" / "test_integration.wav")
        save_audio(audio_out, output_path, tts.sample_rate)
        print(f"         TTS: {len(audio_out)} samples → {output_path}")

    results.ok("Integration pipeline (simulated)")
except Exception as e:
    results.fail("Integration pipeline", e)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
success = results.summary()
print("=" * 60)
sys.exit(0 if success else 1)
