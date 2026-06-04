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


# ============================================================
# P3: FSMN-VAD chunk-size alignment (accumulate to one chunk)
# ============================================================
def test_vad_chunk_accumulation():
    print("\n[P3] FSMN-VAD chunk accumulation")
    from translator.audio import vad as vad_mod

    vad = vad_mod.VadEngine()

    # Stub the heavy model with a call counter; we only test the buffering.
    calls = {"n": 0, "last_len": 0}

    class _Stub:
        def generate(self, input, **kw):
            calls["n"] += 1
            calls["last_len"] = len(input)
            return [{"value": []}]

    vad._model = _Stub()

    chunk_samples = vad._CHUNK_SAMPLES
    small = config.BLOCK_SIZE  # 512 (32ms)
    # Number of 32ms feeds needed to reach (>=) one full chunk.
    feeds_per_chunk = -(-chunk_samples // small)  # ceil division

    # Feed (feeds_per_chunk - 1) small chunks: must NOT call the model yet.
    for _ in range(feeds_per_chunk - 1):
        vad.process_chunk(np.zeros(small, dtype=np.float32))
    if calls["n"] != 0:
        fail("no forward before a full chunk", f"calls={calls['n']}")
        return
    ok("no model call before one full chunk", f"({feeds_per_chunk - 1} feeds buffered)")

    # One more feed crosses the threshold -> exactly one forward pass.
    vad.process_chunk(np.zeros(small, dtype=np.float32))
    if calls["n"] != 1:
        fail("one forward when chunk full", f"calls={calls['n']}")
        return
    ok("one model call once chunk full", f"(block={calls['last_len']} samples)")

    if calls["last_len"] < chunk_samples:
        fail("forward receives >= one chunk", f"len={calls['last_len']} < {chunk_samples}")
        return
    ok("forward block >= configured chunk size")

    # is_final must flush residual even if below a full chunk.
    vad.process_chunk(np.zeros(small, dtype=np.float32))  # buffer a residual
    before = calls["n"]
    vad.process_chunk(np.zeros(small, dtype=np.float32), is_final=True)
    if calls["n"] != before + 1:
        fail("is_final flushes residual", f"calls went {before}->{calls['n']}")
        return
    ok("is_final flushes residual buffer")

    # reset clears the buffer (no forward, no leftover).
    vad.process_chunk(np.zeros(small, dtype=np.float32))
    vad.reset()
    if vad._buf_len != 0 or len(vad._buf) != 0:
        fail("reset clears buffer", f"buf_len={vad._buf_len}")
        return
    ok("reset clears accumulation buffer")


# ============================================================
# P-perf2: Qwen3-ASR transcribes in-memory (no temp-WAV round-trip)
# ============================================================
def test_accurate_no_tempfile():
    print("\n[P-perf2] Qwen3-ASR in-memory transcribe (no disk round-trip)")
    import tempfile, glob, os
    from translator.asr.accurate import AccurateAsr
    from translator.messages import SentenceAudio

    asr = AccurateAsr()

    # Stub the heavy OV model; record what audio form transcribe() passes in.
    seen = {"audio": None}

    class _Stub:
        def transcribe(self, audio=None, **kw):
            seen["audio"] = audio
            return [{"text": "  你好  "}]  # whitespace -> must be stripped

    asr._model = _Stub()

    # Snapshot temp dir so we can assert nothing new (no *.wav) was written.
    tmpdir = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmpdir, "*.wav")))

    sa = SentenceAudio(samples=gen_speech(1000, amp=0.02), start_ms=100, end_ms=1100)
    res = asr.transcribe(sa)

    after = set(glob.glob(os.path.join(tmpdir, "*.wav")))
    if after - before:
        fail("no temp wav written", f"new files: {after - before}")
        return
    ok("no temporary .wav created during transcribe")

    # transcribe() must hand the model a (ndarray, sample_rate) tuple, not a path.
    a = seen["audio"]
    if not (isinstance(a, tuple) and len(a) == 2
            and isinstance(a[0], np.ndarray) and a[1] == config.SAMPLE_RATE):
        fail("model receives (ndarray, sr) tuple", f"got {type(a)}: {a!r}"[:120])
        return
    ok("model fed in-memory (ndarray, 16000) tuple")

    # Audio must be normalized before hand-off (quiet input boosted toward peak).
    peak = float(np.max(np.abs(a[0])))
    if not (0.9 <= peak <= 1.0):
        fail("audio normalized before transcribe", f"peak={peak:.3f}")
        return
    ok("audio normalized before hand-off", f"(peak={peak:.3f})")

    # Result text must be stripped and carry the sentence timestamps.
    if not (res and res.text == "你好" and res.is_final
            and res.start_ms == 100 and res.end_ms == 1100):
        fail("result text stripped + timestamps preserved", repr(res))
        return
    ok("result text stripped, timestamps preserved")


# ============================================================
# P-perf3: offline Paraformer loaded lazily (not at load())
# ============================================================
def test_offline_lazy_load():
    print("\n[P-perf3] offline Paraformer lazy load")
    from translator.asr.streaming import StreamingAsr

    asr = StreamingAsr()
    asr.load()  # loads ONLINE only

    if asr._offline_model is not None:
        fail("offline NOT loaded by load()", "offline model present after load()")
        return
    ok("load() leaves offline model unloaded")
    if asr._online_model is None:
        fail("online loaded by load()", "online model missing after load()")
        return
    ok("load() loads online model eagerly")

    # _ensure_offline must build it once, then be idempotent.
    asr._ensure_offline()
    if asr._offline_model is None:
        fail("_ensure_offline builds offline model", "still None")
        return
    ok("_ensure_offline loads offline model on demand")

    first = asr._offline_model
    asr._ensure_offline()  # second call must not rebuild
    if asr._offline_model is not first:
        fail("_ensure_offline idempotent", "model rebuilt on second call")
        return
    ok("_ensure_offline idempotent (no rebuild)")


# ============================================================
# Q3 (C2): empty / punctuation-only / ultra-short sentence filtering
# ============================================================
def test_meaningful_filter():
    print("\n[Q3] empty/short/punct-only sentence filtering")
    from translator.utils.text import is_meaningful, content_char_count

    # Must be DROPPED (not meaningful).
    drop = ["", "   ", "\n\t", "。", "，。！", "...", "?!", " 、 ", "a", "我"]
    for s in drop:
        if is_meaningful(s):
            fail("drops non-meaningful text", f"kept {s!r}")
            return
    ok("drops empty / whitespace / punct-only / single-char", f"({len(drop)} cases)")

    # Must be KEPT (meaningful).
    keep = ["你好", "hi there", "好的。", "我们走", "OK!", "1+1=2", "嗯嗯嗯"]
    for s in keep:
        if not is_meaningful(s):
            fail("keeps meaningful text", f"dropped {s!r}")
            return
    ok("keeps real sentences", f"({len(keep)} cases)")

    # content_char_count ignores punctuation/whitespace.
    if content_char_count(" 你好 ， 世界 ！ ") != 4:
        fail("content_char_count ignores punct/space",
             f"got {content_char_count(' 你好 ， 世界 ！ ')}")
        return
    ok("content_char_count counts only real chars")

    # Pipeline actually skips non-meaningful finals (guard is wired in).
    import inspect
    from translator import live_pipeline
    src = inspect.getsource(live_pipeline.LivePipeline._sentence_processor_loop)
    if "is_meaningful" not in src:
        fail("filter wired into pipeline", "is_meaningful not used in _sentence_processor_loop")
        return
    ok("is_meaningful guard wired into _sentence_processor_loop")


# ============================================================
# C5: P3 chunk-accumulation did NOT regress sentence segmentation
# (endpoint detected within acceptable delay; consecutive sentences
#  both detected after reset). Uses real FSMN-VAD.
# ============================================================
def test_vad_segmentation_no_regression():
    print("\n[C5] P3 segmentation regression check (real VAD)")
    from translator.audio.vad import VadEngine, SentenceManager

    def tone(ms, f=300):
        n = int(16000 * ms / 1000)
        t = np.linspace(0, ms / 1000, n, dtype=np.float32)
        sig = 0.3 * np.sin(2 * np.pi * f * t) + 0.12 * np.sin(2 * np.pi * 2 * f * t)
        return (sig + 0.01 * np.random.randn(n)).astype(np.float32)

    def sil(ms):
        n = int(16000 * ms / 1000)
        return (0.001 * np.random.randn(n)).astype(np.float32)

    block = config.BLOCK_SIZE

    def run(stream):
        vad = VadEngine()
        sm = SentenceManager(vad)
        ts, out = 0, []
        for i in range(0, len(stream) - block, block):
            res = sm.feed(stream[i:i + block], ts)
            if res is not None:
                out.append((res.start_ms, res.end_ms, ts))
            ts += config.BLOCK_MS
        return out

    # Trailing silence must exceed the endpoint threshold for the endpoint to
    # fire; derive it from config so this test tracks FSMN_VAD_MAX_END_SILENCE_MS
    # instead of assuming a fixed 800ms.
    end_sil = config.FSMN_VAD_MAX_END_SILENCE_MS + 600  # threshold + margin

    # Single sentence: 0.5s sil + 1.5s speech + trailing sil. Endpoint must
    # fire, and not be delayed far beyond speech-end + max_end_silence + chunk.
    one = run(np.concatenate([sil(500), tone(1500), sil(end_sil)]))
    if len(one) != 1:
        fail("single sentence detected exactly once", f"got {len(one)}: {one}")
        return
    flush_ts = one[0][2]
    # speech ends ~2000ms; endpoint = +max_end_silence = ~2000+threshold.
    # Allow up to 2 VAD chunks (400ms) of extra latency from accumulation.
    budget = 2000 + config.FSMN_VAD_MAX_END_SILENCE_MS + 2 * config.FSMN_VAD_CHUNK_MS
    if flush_ts > budget:
        fail("endpoint within acceptable delay", f"flush@{flush_ts}ms > {budget}ms budget")
        return
    ok("single sentence endpoint within delay budget", f"(flush@{flush_ts}ms <= {budget}ms)")

    # Two consecutive same-character sentences: reset must allow the 2nd to be
    # detected (P3's accumulation buffer must not swallow the next sentence).
    # The inter-sentence gap must exceed the endpoint threshold to split them.
    two = run(np.concatenate([sil(400), tone(1200), sil(end_sil),
                              tone(1200), sil(end_sil)]))
    if len(two) != 2:
        fail("two consecutive sentences both detected", f"got {len(two)}: {two}")
        return
    ok("consecutive sentences both detected after reset", f"({len(two)} sentences)")


# ============================================================
# Q4 (C1): conservative ASR text post-processing
# ============================================================
def test_clean_asr_text():
    print("\n[Q4] ASR text cleanup (conservative)")
    from translator.utils.text import clean_asr_text

    # --- Must CLEAN (noise/hallucination) ---
    cases = [
        ("  你好世界  ", "你好世界"),          # trim
        ("你好    世界", "你好 世界"),          # collapse internal whitespace
        ("啊啊啊啊啊啊", "啊啊"),               # long single-char stutter -> 2
        ("好的好的好的好的", "好的"),            # phrase repeated 4x -> 1
        ("我们我们我们走吧", "我们走吧"),        # 3x phrase prefix collapses
    ]
    for raw, want in cases:
        got = clean_asr_text(raw)
        if got != want:
            fail("cleans noise", f"{raw!r} -> {got!r}, want {want!r}")
            return
    ok("cleans stutters/repeats/whitespace", f"({len(cases)} cases)")

    # --- Must NOT corrupt legitimate text (the dangerous direction) ---
    keep = [
        "谢谢",        # legit doubling
        "慢慢来",      # legit doubling
        "好好学习",    # legit doubling
        "哈哈哈",      # legit tripling (below run threshold of 4)
        "你好世界",    # normal sentence
        "我想想",      # legit doubling
        "Hello world", # english untouched
        "看看这个",    # legit doubling
    ]
    for s in keep:
        got = clean_asr_text(s)
        if got != s:
            fail("preserves legitimate text", f"{s!r} -> {got!r} (corrupted)")
            return
    ok("preserves legit doublings/triplings/sentences", f"({len(keep)} cases)")

    # Empty/whitespace stays empty (no crash).
    if clean_asr_text("") != "" or clean_asr_text("   ").strip() != "":
        fail("empty input handled", repr(clean_asr_text("   ")))
        return
    ok("empty/whitespace input handled")

    # Pipeline wires cleanup before the meaningful filter.
    import inspect
    from translator import live_pipeline
    src = inspect.getsource(live_pipeline.LivePipeline._sentence_processor_loop)
    if "clean_asr_text" not in src:
        fail("cleanup wired into pipeline", "clean_asr_text not used")
        return
    if src.index("clean_asr_text") > src.index("is_meaningful"):
        fail("cleanup runs before filter", "clean_asr_text appears after is_meaningful")
        return
    ok("clean_asr_text wired before is_meaningful in pipeline")


# ============================================================
# Q5 (root cause): streaming Paraformer cross-chunk look-back
# ============================================================
def test_streaming_lookback():
    """The dominant streaming-WER fix.

    FunASR's online Paraformer defaults encoder/decoder ``*_look_back`` to 0,
    decoding each 600ms chunk with no cross-chunk context — which inflates WER
    and produces tail hallucinations. This asserts (a) the look-back params are
    actually passed into both streaming generate() calls, and (b) on the
    bundled Chinese example the streaming path now matches the offline result
    (CER ~0), whereas without look-back it diverges.
    """
    print("\n[Q5] streaming cross-chunk look-back (root-cause fix)")
    import inspect, os
    from translator.asr import streaming as st_mod

    # (a) params wired into BOTH streaming generate() calls (feed + end_sentence).
    feed_src = inspect.getsource(st_mod.StreamingAsr.feed)
    end_src = inspect.getsource(st_mod.StreamingAsr.end_sentence)
    for name, src in (("feed", feed_src), ("end_sentence", end_src)):
        if ("encoder_chunk_look_back" not in src
                or "decoder_chunk_look_back" not in src):
            fail("look-back wired into streaming generate()",
                 f"missing in {name}()")
            return
    ok("look-back params wired into feed() and end_sentence()")

    # Config carries sane (non-zero) defaults.
    if config.PARAFORMER_ENCODER_LOOK_BACK <= 0 or config.PARAFORMER_DECODER_LOOK_BACK <= 0:
        fail("non-zero look-back configured",
             f"enc={config.PARAFORMER_ENCODER_LOOK_BACK}, "
             f"dec={config.PARAFORMER_DECODER_LOOK_BACK}")
        return
    ok("non-zero look-back in config",
       f"(enc={config.PARAFORMER_ENCODER_LOOK_BACK}, dec={config.PARAFORMER_DECODER_LOOK_BACK})")

    # (b) Behavioral check on the bundled example, if it is cached locally.
    import glob
    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope", "hub", "models",
        "iic", "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "example", "asr_example.wav")
    hits = glob.glob(pattern)
    if not hits:
        ok("behavioral CER check skipped (example wav not cached)")
        return

    import soundfile as sf
    audio, sr = sf.read(hits[0], dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    from translator.asr.streaming import StreamingAsr
    asr = StreamingAsr()
    asr.load()
    asr.start_sentence(0)
    # This test measures look-back CHARACTER accuracy vs the offline model.
    # Disable punctuation restoration so the streaming output is comparable to
    # the offline (unpunctuated) ground truth; otherwise the added period would
    # register as a spurious "error" here. Punctuation has its own test (Q8).
    _punc = config.PUNC_ENABLED
    config.PUNC_ENABLED = False
    try:
        ts, step, last = 0, config.BLOCK_SIZE, ""
        for i in range(0, len(audio), step):
            r = asr.feed(audio[i:i + step], ts)
            if r and r.text:
                last = r.text
            ts += config.BLOCK_MS
        fin = asr.end_sentence(ts)
        final_text = fin.text if fin else last
    finally:
        config.PUNC_ENABLED = _punc

    # Ground-truth proxy: the offline model on the same clip.
    asr._ensure_offline()
    gt = asr._extract_text(asr._offline_model.generate(input=audio))

    def cer(ref, hyp):
        r, h = list(ref), list(hyp)
        d = np.zeros((len(r) + 1, len(h) + 1), int)
        for i in range(len(r) + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
        return d[len(r)][len(h)] / max(1, len(r))

    c = cer(gt, final_text)
    # With look-back the streaming output should track offline closely.
    if c > 0.05:
        fail("streaming tracks offline with look-back",
             f"CER={c:.3f} (gt={gt!r}, got={final_text!r})")
        return
    ok("streaming matches offline on example clip", f"(CER={c:.3f})")


# ============================================================
# Q6: VAD onset pre-buffer fed to streaming ASR (no dropped first syllable)
# ============================================================
def test_onset_seeds_streaming():
    """The leading-character fix.

    The VAD captures a ~320ms onset pre-buffer (so Qwen3 gets the full
    sentence), but historically that onset was NEVER fed to the streaming
    Paraformer — the live partial therefore lost the first syllable spoken
    before speech was confirmed. This asserts (a) the onset is exposed via
    take_onset and consumed once, and (b) the pipeline feeds it.
    """
    print("\n[Q6] VAD onset pre-buffer seeds streaming ASR")
    import inspect
    from translator.audio.vad import VadEngine, SentenceManager, VadState

    # (a) take_onset returns the onset once, then None until the next sentence.
    vad = VadEngine()
    sm = SentenceManager(vad)

    def tone(ms, f=300):
        n = int(16000 * ms / 1000)
        t = np.linspace(0, ms / 1000, n, dtype=np.float32)
        return (0.3 * np.sin(2 * np.pi * f * t)
                + 0.12 * np.sin(2 * np.pi * 2 * f * t)
                + 0.01 * np.random.randn(n)).astype(np.float32)

    def sil(ms):
        n = int(16000 * ms / 1000)
        return (0.001 * np.random.randn(n)).astype(np.float32)

    if sm.take_onset() is not None:
        fail("no onset before any speech", "take_onset non-None at start")
        return
    ok("take_onset is None before speech starts")

    stream = np.concatenate([sil(300), tone(1500), sil(1800)])
    block = config.BLOCK_SIZE
    ts, onset_seen = 0, None
    for i in range(0, len(stream) - block, block):
        sm.feed(stream[i:i + block], ts)
        if sm.state in (VadState.SPEECH, VadState.TRAILING_SILENCE) and onset_seen is None:
            onset_seen = sm.take_onset()
            break
        ts += config.BLOCK_MS

    if not onset_seen:
        fail("onset captured at speech start", f"got {onset_seen!r}")
        return
    ok("onset pre-buffer captured at speech start", f"({len(onset_seen)} chunks)")

    if sm.take_onset() is not None:
        fail("onset consumed once", "second take_onset returned non-None")
        return
    ok("onset cleared after first read (consumed once)")

    # (b) Pipeline actually wires take_onset into _process_chunk.
    from translator import live_pipeline
    src = inspect.getsource(live_pipeline.LivePipeline._process_chunk)
    if "take_onset" not in src:
        fail("onset wired into pipeline", "take_onset not used in _process_chunk")
        return
    ok("take_onset wired into _process_chunk")


# ============================================================
# Q9: onset recovers the sentence head despite large VAD detection latency
# ============================================================
def test_onset_recovers_head_under_latency():
    """Root-cause fix for the dropped/garbled first syllable.

    FSMN-VAD's detection latency (true onset -> start event) can exceed 1s, and
    the start_ms it reports lags the true onset too. The old fixed 320ms
    pre-buffer was far shorter than that gap, so the sentence head was discarded
    before speech was confirmed. The fix keeps a ~2s rolling buffer and slices
    it back to the reported start_ms (minus a margin).

    Uses a STUB VAD so the latency and reported start_ms are controlled exactly,
    rather than relying on a model's timing on a particular clip.
    """
    print("\n[Q9] onset recovers head under large VAD detection latency")
    from translator.audio.vad import SentenceManager, VadState

    # Pre-buffer must be long enough to cover a >1s detection latency.
    pre_ms = SentenceManager._PRE_BUFFER_CHUNKS * config.BLOCK_MS
    if pre_ms < 1000:
        fail("pre-buffer covers >1s latency", f"pre-buffer only {pre_ms}ms")
        return
    ok("rolling pre-buffer covers large detection latency", f"({pre_ms}ms)")

    # Stub VAD: stays silent until a chosen detection chunk, then reports a
    # start_ms that points far earlier (the true onset). No model needed.
    DETECT_AT_MS = 1600      # event arrives at t=1600ms
    REPORTED_START_MS = 1000  # but VAD says speech started at 1000ms

    class _StubVad:
        def __init__(self):
            self.fired = False

        def process_chunk(self, samples, is_final=False):
            return []  # default: no event

        def reset(self):
            self.fired = False

    stub = _StubVad()
    sm = SentenceManager(stub)
    block = config.BLOCK_SIZE
    ts = 0
    onset = None
    n = int(2.4 * 16000)  # 2.4s of feed
    idx = 0
    while idx < n:
        # Inject the start event exactly at DETECT_AT_MS.
        if ts >= DETECT_AT_MS and not stub.fired:
            stub.fired = True
            stub.process_chunk = lambda s, is_final=False: [[REPORTED_START_MS, -1]]
        else:
            stub.process_chunk = lambda s, is_final=False: []
        sm.feed(np.zeros(block, dtype=np.float32), ts)
        if sm.state in (VadState.SPEECH, VadState.TRAILING_SILENCE) and onset is None:
            onset = sm.take_onset()
            break
        ts += config.BLOCK_MS
        idx += block

    if not onset:
        fail("onset recovered at speech start", "no onset returned")
        return

    onset_ms = len(onset) * config.BLOCK_MS
    # The recovered head should reach back to ~start_ms (minus margin), i.e.
    # roughly (detection_ts - start_ms) + margin, NOT the old fixed 320ms.
    expected_min = (DETECT_AT_MS - REPORTED_START_MS)  # 600ms of head, at least
    if onset_ms < expected_min:
        fail("onset reaches back to reported start_ms",
             f"recovered {onset_ms}ms < expected >= {expected_min}ms")
        return
    ok("onset recovers head back to reported start_ms",
       f"({onset_ms}ms recovered, vs old fixed 320ms)")

    # And it must be materially more than the old 320ms fixed pre-buffer.
    if onset_ms <= 320:
        fail("recovers more than the old 320ms", f"only {onset_ms}ms")
        return
    ok("recovers materially more than the old 320ms pre-buffer")


# ============================================================
# Q7: endpoint fires on the configured silence threshold
# ============================================================
def test_endpoint_silence_threshold():
    """The endpoint is acoustic: a sentence ends after
    FSMN_VAD_MAX_END_SILENCE_MS of continuous silence.

    Verifies the configured threshold drives segmentation as intended: a pause
    comfortably below it does NOT split an utterance, while a pause comfortably
    above it does. With the 600ms setting, a sub-600ms pause keeps one
    sentence and a clearly-longer pause produces two.
    """
    print("\n[Q7] endpoint fires on configured silence threshold")

    thr = config.FSMN_VAD_MAX_END_SILENCE_MS
    ok("endpoint threshold configured", f"({thr}ms)")

    from translator.audio.vad import VadEngine, SentenceManager

    def tone(ms, f=300):
        n = int(16000 * ms / 1000)
        t = np.linspace(0, ms / 1000, n, dtype=np.float32)
        return (0.3 * np.sin(2 * np.pi * f * t)
                + 0.12 * np.sin(2 * np.pi * 2 * f * t)
                + 0.01 * np.random.randn(n)).astype(np.float32)

    def sil(ms):
        n = int(16000 * ms / 1000)
        return (0.001 * np.random.randn(n)).astype(np.float32)

    def count(stream):
        vad = VadEngine()
        sm = SentenceManager(vad)
        block, ts, n = config.BLOCK_SIZE, 0, 0
        for i in range(0, len(stream) - block, block):
            if sm.feed(stream[i:i + block], ts) is not None:
                n += 1
            ts += config.BLOCK_MS
        return n

    # A pause well BELOW the threshold must NOT split the utterance.
    short_pause = max(100, thr - 350)
    n_short = count(np.concatenate([sil(300), tone(1000), sil(short_pause),
                                    tone(1000), sil(thr + 1200)]))
    if n_short != 1:
        fail("sub-threshold pause stays one sentence",
             f"pause={short_pause}ms split into {n_short}")
        return
    ok("sub-threshold pause kept as one sentence", f"({short_pause}ms pause)")

    # A pause well ABOVE the threshold must split into two sentences.
    long_pause = thr + 500
    n_long = count(np.concatenate([sil(300), tone(1000), sil(long_pause),
                                   tone(1000), sil(thr + 1200)]))
    if n_long != 2:
        fail("supra-threshold pause splits sentence",
             f"pause={long_pause}ms produced {n_long} sentences")
        return
    ok("supra-threshold pause splits into two sentences", f"({long_pause}ms pause)")


# ============================================================
# Q8: punctuation restoration for the LIVE streaming partial subtitles
# ============================================================
def test_punctuation_restore():
    """FunASR's CER gap vs Qwen3 was mostly missing punctuation, not wrong
    characters. The live partial subtitles now run through ct-punc.

    Since the partials are produced on the capture/VAD/ASR hot loop, the
    punctuator must NEVER block it on the model download/load. This verifies,
    without downloading the model:
      (a) no model is built at construction;
      (b) non-blocking restore returns RAW text immediately while loading, and
          kicks off a background load (preload);
      (c) graceful degradation — unchanged text when disabled, on empty input,
          on load failure, or on inference error;
      (d) when a model IS present (stubbed), restore() uses its output;
      (e) punctuation is wired into BOTH feed() and end_sentence() (the partial
          path), and the offline fallback uses block=True.
    """
    print("\n[Q8] punctuation restoration (live partial)")
    import inspect
    from translator.asr.punctuation import Punctuator

    # (a) nothing loaded at construction.
    p = Punctuator()
    if p.loaded:
        fail("punctuator lazy", "model present at construction")
        return
    ok("punctuator does not load model at construction")

    orig = config.PUNC_ENABLED
    try:
        # (b) disabled -> identity, and must NOT start a background load.
        config.PUNC_ENABLED = False
        if p.restore("你好世界") != "你好世界":
            fail("disabled is identity", "text changed while PUNC_ENABLED=False")
            return
        p.preload()
        if p._loading:
            fail("disabled does not preload", "_loading set while disabled")
            return
        ok("PUNC_ENABLED=False is identity and does not load")

        config.PUNC_ENABLED = True
        # empty / whitespace unchanged, no crash.
        if p.restore("") != "" or p.restore("   ") != "   ":
            fail("empty input unchanged", repr(p.restore("   ")))
            return
        ok("empty/whitespace input returned unchanged")

        # (b) non-blocking restore while model not ready -> RAW text now.
        p_nb = Punctuator()
        out = p_nb.restore("你好世界", block=False)
        if out != "你好世界":
            fail("non-blocking returns raw while loading", repr(out))
            return
        ok("non-blocking restore returns raw text immediately (no stall)")

        # (c) load failure -> identity.
        p_fail = Punctuator()
        p_fail._load_failed = True
        if p_fail.restore("你好世界") != "你好世界":
            fail("load-failure is identity", "text changed after load failure")
            return
        ok("load failure degrades to identity (no crash)")

        # (d) stub model -> punctuated output used.
        p2 = Punctuator()

        class _Stub:
            def generate(self, input):
                return [{"text": input + "。"}]

        p2._model = _Stub()
        if p2.restore("你好世界") != "你好世界。":
            fail("uses model output", repr(p2.restore("你好世界")))
            return
        ok("restore() applies model punctuation when available")

        # stub that raises -> identity.
        class _BadStub:
            def generate(self, input):
                raise RuntimeError("boom")

        p2._model = _BadStub()
        if p2.restore("你好世界") != "你好世界":
            fail("inference error is identity", repr(p2.restore("你好世界")))
            return
        ok("inference error degrades to identity")
    finally:
        config.PUNC_ENABLED = orig

    # (e) wiring: partial path (feed + end_sentence) punctuates non-blocking;
    #     offline fallback punctuates blocking; load() preloads.
    from translator.asr import streaming as st_mod
    feed_src = inspect.getsource(st_mod.StreamingAsr.feed)
    end_src = inspect.getsource(st_mod.StreamingAsr.end_sentence)
    off_src = inspect.getsource(st_mod.StreamingAsr.finalize_offline)
    load_src = inspect.getsource(st_mod.StreamingAsr.load)

    if "_punctuator.restore" not in feed_src:
        fail("partial punctuation wired into feed()", "restore not called in feed")
        return
    if "_punctuator.restore" not in end_src:
        fail("partial punctuation wired into end_sentence()", "restore not called")
        return
    ok("live partial punctuation wired into feed() + end_sentence()")

    if "block=True" not in off_src:
        fail("offline fallback uses blocking restore", "block=True not found")
        return
    ok("offline fallback restores punctuation with block=True")

    if "preload" not in load_src:
        fail("punctuator preloaded in load()", "preload() not called")
        return
    ok("punctuation model preloaded (background) in load()")

    # (f) real model check, only if it happens to be cached/available.
    real = Punctuator()
    real._load()  # synchronous; sets _load_failed if unavailable
    if real.loaded:
        out = real.restore("你好世界今天天气很好", block=True)
        content = "".join(ch for ch in out if ch not in "，。！？、；：")
        if content != "你好世界今天天气很好":
            fail("real ct-punc preserves characters", f"{out!r}")
            return
        ok("real ct-punc preserves characters + adds punctuation", f"({out!r})")
    else:
        ok("real ct-punc not cached — skipped behavioral check")


def main():
    print("ASR Optimization Test Suite")
    print("=" * 60)
    tests = [
        test_end_sentence_flushes_tail,
        test_normalization,
        test_vad_config_wired,
        test_vad_chunk_accumulation,
        test_accurate_no_tempfile,
        test_offline_lazy_load,
        test_meaningful_filter,
        test_vad_segmentation_no_regression,
        test_clean_asr_text,
        test_streaming_lookback,
        test_onset_seeds_streaming,
        test_onset_recovers_head_under_latency,
        test_endpoint_silence_threshold,
        test_punctuation_restore,
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
