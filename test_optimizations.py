"""Tests for ASR optimization fixes.

Each optimization gets its own test so they can be run and validated
independently:

  Fix 1: end_sentence flushes residual tail audio (no dropped sentence tails)
  Fix 2: audio normalization preprocessing
  Fix 3: FSMN-VAD max_end_silence_ms wiring + wider pre-buffer

Uses real FunASR models (cached locally) where possible.
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
import config


PASS, FAIL = 0, 0


def ok(name, extra=""):
    global PASS
    PASS += 1
    print(f"  [PASS] {name} {extra}".rstrip())


def fail(name, err):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}: {err}")


def gen_speech(duration_ms, freq=300, amp=0.3, sample_rate=16000):
    n = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, dtype=np.float32)
    sig = amp * np.sin(2 * np.pi * freq * t)
    sig += amp * 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    noise = np.random.randn(n).astype(np.float32) * 0.01
    return (sig + noise).astype(np.float32)


# ============================================================
# Fix 1: end_sentence flushes residual tail audio
# ============================================================
def test_end_sentence_flushes_tail():
    print("\n[Fix 1] end_sentence flushes residual tail audio")
    from translator.asr.streaming import StreamingAsr

    asr = StreamingAsr()
    asr.load()

    # Feed an amount of audio that is NOT a multiple of the chunk size, so a
    # residual remains in _chunk_buffer when the sentence ends.
    chunk = config.PARAFORMER_CHUNK_SIZE
    total = chunk * 2 + chunk // 2  # 2.5 chunks -> 0.5 chunk residual
    audio = gen_speech(int(total / config.SAMPLE_RATE * 1000), freq=220)

    asr.start_sentence(0)
    step = config.BLOCK_SIZE
    ts = 0
    for i in range(0, len(audio) - step, step):
        asr.feed(audio[i:i + step], ts)
        ts += config.BLOCK_MS

    # There must be a residual smaller than one chunk left over.
    residual_len = len(asr._chunk_buffer)
    if not (0 < residual_len < chunk):
        fail("residual present before end_sentence",
             f"residual_len={residual_len}, expected 0<x<{chunk}")
        return
    ok("residual buffer present before end", f"({residual_len} samples)")

    # end_sentence must consume the residual (buffer empty afterwards) and
    # must not raise.
    asr.end_sentence(ts)
    if len(asr._chunk_buffer) != 0:
        fail("end_sentence clears buffer", f"len={len(asr._chunk_buffer)}")
        return
    ok("end_sentence consumes residual without error")

    # Cache must be reset for the next sentence.
    if asr._cache != {}:
        fail("cache reset after end_sentence", asr._cache)
        return
    ok("cache reset after end_sentence")


# ============================================================
# Fix 2: audio normalization preprocessing
# ============================================================
def test_normalization():
    print("\n[Fix 2] audio normalization")
    from translator.utils.audio import normalize_audio

    # Quiet signal should be amplified toward the target peak.
    quiet = gen_speech(500, amp=0.02)
    norm = normalize_audio(quiet)
    peak = float(np.max(np.abs(norm)))
    if not (0.9 <= peak <= 1.0):
        fail("quiet signal amplified to target peak", f"peak={peak:.3f}")
        return
    ok("quiet signal amplified", f"peak {float(np.max(np.abs(quiet))):.3f} -> {peak:.3f}")

    # Output must not clip.
    if peak > 1.0:
        fail("no clipping", f"peak={peak}")
        return
    ok("no clipping")

    # Pure silence must stay silent (no divide-by-zero blowup).
    silence = np.zeros(8000, dtype=np.float32)
    norm_sil = normalize_audio(silence)
    if np.any(np.isnan(norm_sil)) or float(np.max(np.abs(norm_sil))) > 1e-6:
        fail("silence stays silent", f"max={float(np.max(np.abs(norm_sil)))}")
        return
    ok("silence handled (no NaN, stays silent)")

    # dtype preserved.
    if norm.dtype != np.float32:
        fail("dtype preserved", norm.dtype)
        return
    ok("dtype float32 preserved")


# ============================================================
# Fix 3: FSMN-VAD max_end_silence_ms + wider pre-buffer
# ============================================================
def test_vad_config_wired():
    print("\n[Fix 3] FSMN-VAD endpoint config + pre-buffer")
    import inspect
    from translator.audio import vad as vad_mod

    # max_end_silence_ms must actually be passed to generate().
    src = inspect.getsource(vad_mod.VadEngine.process_chunk)
    if "max_end_silence_time" not in src:
        fail("max_end_silence_time passed to generate", "not found in process_chunk")
        return
    ok("max_end_silence_time wired into generate()")

    # Pre-buffer should cover ~320ms (>= 10 chunks of 32ms).
    pre = vad_mod.SentenceManager._PRE_BUFFER_CHUNKS
    if pre < 10:
        fail("pre-buffer widened", f"_PRE_BUFFER_CHUNKS={pre}, expected >=10")
        return
    ok("pre-buffer widened", f"({pre} chunks = {pre * config.BLOCK_MS}ms)")

    # VAD must still load and process a chunk without error.
    vad = vad_mod.VadEngine()
    seg = vad.process_chunk(gen_speech(config.FSMN_VAD_CHUNK_MS))
    ok("VadEngine processes chunk after change", f"(segments={seg})")


def main():
    print("ASR Optimization Test Suite")
    print("=" * 60)
    tests = [
        test_end_sentence_flushes_tail,
        test_normalization,
        test_vad_config_wired,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            fail(t.__name__, e)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
