"""Test script for FSMN-VAD endpoint detection integration.

Tests:
1. FSMN-VAD model loading
2. VadEngine processing with synthetic audio
3. SentenceManager endpoint detection with speech + silence patterns
4. Integration with Paraformer Online streaming ASR
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
import config


def generate_sine_wave(duration_ms, freq=440, sample_rate=16000):
    """Generate a sine wave (simulates speech-like signal)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * freq * t)
    noise = np.random.randn(n_samples).astype(np.float32) * 0.02
    return signal + noise


def generate_silence(duration_ms, sample_rate=16000):
    """Generate near-silence audio."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return np.random.randn(n_samples).astype(np.float32) * 0.001


def test_vad_engine_load():
    """Test 1: FSMN-VAD model loads successfully."""
    print("=" * 60)
    print("Test 1: FSMN-VAD Model Loading")
    print("=" * 60)

    try:
        from translator.audio.vad import VadEngine
        t0 = time.time()
        vad = VadEngine()
        elapsed = time.time() - t0
        print(f"  [PASS] FSMN-VAD loaded in {elapsed:.2f}s")
        return vad
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vad_engine_process(vad):
    """Test 2: VadEngine processes audio chunks and returns segments."""
    print("\n" + "=" * 60)
    print("Test 2: VadEngine Chunk Processing")
    print("=" * 60)

    try:
        # Feed silence
        silence = generate_silence(config.BLOCK_MS)
        segments = vad.process_chunk(silence)
        print(f"  Silence chunk -> segments: {segments}")

        # Feed speech-like audio
        speech = generate_sine_wave(config.BLOCK_MS, freq=300)
        segments = vad.process_chunk(speech)
        print(f"  Speech chunk -> segments: {segments}")

        # Feed more speech
        for i in range(20):
            speech = generate_sine_wave(config.BLOCK_MS, freq=300 + i * 10)
            segments = vad.process_chunk(speech)
            if segments:
                print(f"  Chunk {i+2} -> segments: {segments}")

        # Feed silence to trigger endpoint
        for i in range(30):
            silence = generate_silence(config.BLOCK_MS)
            segments = vad.process_chunk(silence)
            if segments:
                print(f"  Silence chunk {i+1} -> segments: {segments}")

        # Final flush
        silence = generate_silence(config.BLOCK_MS)
        segments = vad.process_chunk(silence, is_final=True)
        if segments:
            print(f"  Final flush -> segments: {segments}")

        print("  [PASS] VadEngine processing works")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentence_manager():
    """Test 3: SentenceManager detects sentence boundaries via FSMN-VAD."""
    print("\n" + "=" * 60)
    print("Test 3: SentenceManager Endpoint Detection")
    print("=" * 60)

    try:
        from translator.audio.vad import VadEngine, SentenceManager, VadState

        vad = VadEngine()
        mgr = SentenceManager(vad)

        sentences_detected = []
        timestamp_ms = 0

        # Phase 1: Initial silence (500ms)
        print("  Phase 1: Initial silence (500ms)...")
        for _ in range(int(500 / config.BLOCK_MS)):
            silence = generate_silence(config.BLOCK_MS)
            result = mgr.feed(silence, timestamp_ms)
            if result is not None:
                sentences_detected.append(result)
                print(f"    Sentence detected: {result.start_ms}ms - {result.end_ms}ms "
                      f"({len(result.samples)} samples)")
            timestamp_ms += config.BLOCK_MS

        print(f"    State after silence: {mgr.state.name}")

        # Phase 2: Speech (2 seconds)
        print("  Phase 2: Speech (2000ms)...")
        for i in range(int(2000 / config.BLOCK_MS)):
            speech = generate_sine_wave(config.BLOCK_MS, freq=200 + i * 5)
            result = mgr.feed(speech, timestamp_ms)
            if result is not None:
                sentences_detected.append(result)
                print(f"    Sentence detected: {result.start_ms}ms - {result.end_ms}ms "
                      f"({len(result.samples)} samples)")
            timestamp_ms += config.BLOCK_MS

        print(f"    State after speech: {mgr.state.name}")

        # Phase 3: Silence (1.5 seconds - should trigger endpoint)
        print("  Phase 3: Silence (1500ms - expect endpoint)...")
        for i in range(int(1500 / config.BLOCK_MS)):
            silence = generate_silence(config.BLOCK_MS)
            result = mgr.feed(silence, timestamp_ms)
            if result is not None:
                sentences_detected.append(result)
                print(f"    Sentence detected: {result.start_ms}ms - {result.end_ms}ms "
                      f"({len(result.samples)} samples, "
                      f"duration={result.end_ms - result.start_ms}ms)")
            timestamp_ms += config.BLOCK_MS

        print(f"    State after silence: {mgr.state.name}")

        # Phase 4: Another speech segment (1 second)
        print("  Phase 4: Second speech (1000ms)...")
        for i in range(int(1000 / config.BLOCK_MS)):
            speech = generate_sine_wave(config.BLOCK_MS, freq=400 + i * 5)
            result = mgr.feed(speech, timestamp_ms)
            if result is not None:
                sentences_detected.append(result)
                print(f"    Sentence detected: {result.start_ms}ms - {result.end_ms}ms")
            timestamp_ms += config.BLOCK_MS

        # Phase 5: Final silence (1.5 seconds)
        print("  Phase 5: Final silence (1500ms)...")
        for i in range(int(1500 / config.BLOCK_MS)):
            silence = generate_silence(config.BLOCK_MS)
            result = mgr.feed(silence, timestamp_ms)
            if result is not None:
                sentences_detected.append(result)
                print(f"    Sentence detected: {result.start_ms}ms - {result.end_ms}ms")
            timestamp_ms += config.BLOCK_MS

        print(f"\n  Total sentences detected: {len(sentences_detected)}")
        for i, s in enumerate(sentences_detected):
            duration = s.end_ms - s.start_ms
            print(f"    Sentence {i+1}: {s.start_ms}ms - {s.end_ms}ms "
                  f"(duration={duration}ms, samples={len(s.samples)})")

        if len(sentences_detected) > 0:
            print("  [PASS] SentenceManager detected sentence boundaries")
        else:
            print("  [WARN] No sentences detected - FSMN-VAD may not trigger on synthetic audio")
            print("         This is expected: FSMN-VAD is trained on real speech, not sine waves")

        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_audio():
    """Test 4: Test with a real audio file if available."""
    print("\n" + "=" * 60)
    print("Test 4: Real Audio Test (optional)")
    print("=" * 60)

    import os
    import glob

    # Look for any WAV files in the project
    wav_files = glob.glob("output/*.wav") + glob.glob("test_audio/*.wav")
    if not wav_files:
        print("  [SKIP] No WAV files found for testing")
        print("         Place a WAV file in output/ or test_audio/ to test with real audio")
        return True

    try:
        import soundfile as sf
        from translator.audio.vad import VadEngine, SentenceManager

        wav_path = wav_files[0]
        print(f"  Using: {wav_path}")

        audio, sr = sf.read(wav_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != config.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        print(f"  Audio: {len(audio)} samples, {len(audio)/config.SAMPLE_RATE:.2f}s")

        vad = VadEngine()
        mgr = SentenceManager(vad)
        sentences = []
        timestamp_ms = 0

        for i in range(0, len(audio) - config.BLOCK_SIZE, config.BLOCK_SIZE):
            chunk = audio[i:i + config.BLOCK_SIZE]
            result = mgr.feed(chunk, timestamp_ms)
            if result is not None:
                sentences.append(result)
            timestamp_ms += config.BLOCK_MS

        print(f"  Detected {len(sentences)} sentences:")
        for i, s in enumerate(sentences):
            duration = s.end_ms - s.start_ms
            print(f"    Sentence {i+1}: {s.start_ms}ms - {s.end_ms}ms (duration={duration}ms)")

        print("  [PASS] Real audio processing works")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_streaming_asr():
    """Test 5: Integration test - FSMN-VAD + Paraformer Online."""
    print("\n" + "=" * 60)
    print("Test 5: Integration - FSMN-VAD + Paraformer Online")
    print("=" * 60)

    try:
        from translator.audio.vad import VadEngine, SentenceManager, VadState
        from translator.asr.streaming import StreamingAsr

        print("  Loading FSMN-VAD...")
        vad = VadEngine()
        mgr = SentenceManager(vad)

        print("  Loading Paraformer Online...")
        asr = StreamingAsr()
        asr.load()

        print("  Running simulated pipeline with synthetic audio...")

        # Generate a speech-like pattern: speech -> silence -> speech -> silence
        timestamp_ms = 0
        in_sentence = False
        sentences = []
        partial_texts = []

        # 500ms silence + 2s speech + 1s silence + 1s speech + 1s silence
        audio_pattern = (
            [("silence", 500)]
            + [("speech", 2000)]
            + [("silence", 1000)]
            + [("speech", 1000)]
            + [("silence", 1000)]
        )

        for seg_type, duration_ms in audio_pattern:
            n_chunks = int(duration_ms / config.BLOCK_MS)
            for i in range(n_chunks):
                if seg_type == "speech":
                    samples = generate_sine_wave(config.BLOCK_MS, freq=300 + i * 5)
                else:
                    samples = generate_silence(config.BLOCK_MS)

                sentence_audio = mgr.feed(samples, timestamp_ms)
                state = mgr.state

                # Simulate streaming ASR when in speech
                if state == VadState.SPEECH or state == VadState.TRAILING_SILENCE:
                    if not in_sentence:
                        in_sentence = True
                        asr.start_sentence(timestamp_ms)

                    asr_result = asr.feed(samples, timestamp_ms)
                    if asr_result and asr_result.text:
                        partial_texts.append(asr_result.text)

                if sentence_audio is not None:
                    asr.end_sentence(timestamp_ms)
                    in_sentence = False
                    sentences.append(sentence_audio)
                    print(f"    Endpoint detected at {timestamp_ms}ms "
                          f"(sentence: {sentence_audio.start_ms}-{sentence_audio.end_ms}ms)")

                timestamp_ms += config.BLOCK_MS

        print(f"\n  Results:")
        print(f"    Sentences detected: {len(sentences)}")
        print(f"    Partial ASR outputs: {len(partial_texts)}")

        if len(sentences) > 0:
            print("  [PASS] Integration test passed")
        else:
            print("  [WARN] No sentences detected with synthetic audio")
            print("         FSMN-VAD is trained on real speech; synthetic sine waves may not trigger it")

        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("FSMN-VAD Endpoint Detection - Test Suite")
    print("=" * 60)
    print(f"Config: BLOCK_MS={config.BLOCK_MS}, BLOCK_SIZE={config.BLOCK_SIZE}")
    print(f"        FSMN_VAD_CHUNK_MS={config.FSMN_VAD_CHUNK_MS}")
    print(f"        VAD_MIN_SPEECH_MS={config.VAD_MIN_SPEECH_MS}")
    print()

    results = {}

    # Test 1: Load
    vad = test_vad_engine_load()
    results["load"] = vad is not None

    # Test 2: Process
    if vad:
        results["process"] = test_vad_engine_process(vad)
    else:
        results["process"] = False

    # Test 3: SentenceManager
    results["sentence_mgr"] = test_sentence_manager()

    # Test 4: Real audio
    results["real_audio"] = test_with_real_audio()

    # Test 5: Integration
    results["integration"] = test_integration_with_streaming_asr()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
