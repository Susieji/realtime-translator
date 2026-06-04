"""Global configuration for realtime-translator."""

# Audio capture
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_MS = 32
BLOCK_SIZE = 512  # 32ms @ 16kHz

# VAD (FunASR FSMN-VAD with endpoint detection)
VAD_MIN_SPEECH_MS = 250
FSMN_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FSMN_VAD_CHUNK_MS = 200  # chunk size for streaming VAD (ms)
# Silence (ms) the VAD must observe before it declares a sentence endpoint.
# This is an ACOUSTIC, not semantic, boundary: a sentence ends once this much
# continuous silence is seen, regardless of whether the utterance is
# semantically complete. Lower values segment sooner (lower latency) at the
# risk of splitting on mid-sentence pauses; higher values tolerate pauses but
# delay the final transcription. Set to 800ms: a pause is only recognized after
# 800ms of continuous silence.
FSMN_VAD_MAX_END_SILENCE_MS = 800  # max silence before endpoint (ms)
# Rolling pre-buffer kept before speech is confirmed, so the sentence head can
# be recovered once FSMN-VAD finally fires. FSMN-VAD's detection latency from
# true onset to the start event can exceed 1s, so this must be generous (the
# old fixed 320ms clipped the first syllable). On speech-start the buffer is
# sliced back to the reported start_ms minus VAD_ONSET_MARGIN_MS (a cushion for
# the VAD under-reporting the onset).
VAD_PRE_BUFFER_MS = 2000   # rolling pre-speech audio retained (ms)
VAD_ONSET_MARGIN_MS = 200  # extra lead-in kept before reported start_ms (ms)

# Paraformer streaming ASR
PARAFORMER_ONLINE_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
PARAFORMER_OFFLINE_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PARAFORMER_CHUNK_MS = 600
PARAFORMER_CHUNK_SIZE = int(SAMPLE_RATE * PARAFORMER_CHUNK_MS / 1000)  # 9600
# Streaming chunk config: [left_lookback, center, right_lookahead] in 60ms units.
# [0, 10, 5] == 600ms center chunk with 300ms lookahead.
PARAFORMER_CHUNK_LOOK = [0, 10, 5]
# History the streaming encoder/decoder may attend to across chunks. FunASR
# defaults BOTH to 0 (no cross-chunk context), which makes each 600ms chunk be
# decoded in isolation and inflates streaming WER (and produces tail
# hallucinations). The official online-Paraformer example uses 4 / 1, which on
# the bundled asr_example.wav cuts streaming CER from ~0.16 to ~0.00.
PARAFORMER_ENCODER_LOOK_BACK = 4
PARAFORMER_DECODER_LOOK_BACK = 1

# Punctuation restoration (FunASR ct-punc) for the Paraformer-Offline fallback.
# Paraformer emits raw, unpunctuated text; on benchmarks its CER gap vs Qwen3
# was almost entirely punctuation, not wrong characters. When Qwen3 is
# unavailable we run the offline result through ct-punc so the fallback output
# reads like a finished sentence (commas / periods / question marks). Qwen3
# already punctuates, so its path is left untouched.
#   - The small zh-cn model (~290MB) is enough for Chinese punctuation and is
#     fast on CPU; the large cn-en model (~1GB) is overkill here.
#   - Loaded lazily on first use, so it costs nothing unless the fallback path
#     actually runs. Set PUNC_ENABLED=False to disable entirely.
PUNC_ENABLED = True
PUNC_MODEL = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

# Qwen3-ASR (iGPU via OpenVINO)
QWEN_MODEL_ID = "snake7gun/Qwen3-ASR-0.6B-fp16-ov"
QWEN_DEVICE = "GPU"

# Speaker ID (CAM++) - reserved for future use
SPEAKER_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"
SPEAKER_ENABLED = False

# Whisper (English ASR) - reserved for future use
WHISPER_MODEL = "base"
WHISPER_DEVICE = "CPU"
WHISPER_ENABLED = False

# Translation (Opus-MT)
TRANSLATION_MODELS = {
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
}

# TTS (MeloTTS)
TTS_SAMPLE_RATE = 44100
TTS_SPEED = 1.0

# WebSocket server
WS_HOST = "127.0.0.1"
WS_PORT = 8765
