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
FSMN_VAD_MAX_END_SILENCE_MS = 800  # max silence before endpoint (ms)

# Paraformer streaming ASR
PARAFORMER_ONLINE_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
PARAFORMER_OFFLINE_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PARAFORMER_CHUNK_MS = 600
PARAFORMER_CHUNK_SIZE = int(SAMPLE_RATE * PARAFORMER_CHUNK_MS / 1000)  # 9600

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
