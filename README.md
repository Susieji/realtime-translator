# Realtime Translator

Real-time speech-to-speech translation system. Captures microphone audio, transcribes in real-time, translates, and synthesizes target language speech.

```
Mic → VAD → Streaming ASR (partial) → UI real-time display
              └→ Accurate ASR (final) → Translation → TTS → Audio playback
```

## Features

- **Real-time ASR**: Paraformer Online streams partial text as you speak
- **Accurate ASR**: Qwen3-ASR (OpenVINO, iGPU) provides high-accuracy final transcription
- **Auto fallback**: If Qwen3-ASR is unavailable, Paraformer Offline (CPU) takes over
- **Translation**: Opus-MT for zh↔en translation
- **TTS**: MeloTTS generates target language audio
- **Web UI**: Live display via WebSocket at http://127.0.0.1:8766
- **Speaker ID**: Interface reserved for future CAM++ integration
- **English ASR**: Interface reserved for future Whisper integration

## Architecture

```
Thread 1 (Main)       : Mic → VAD → Paraformer Online → WS push partial
Thread 2 (Processor)  : Qwen3-ASR (or Paraformer Offline fallback) → WS push final
Thread 3 (Translator) : Opus-MT → WS push translation
Thread 4 (TTS)        : MeloTTS → save WAV → WS push audio URL
Thread 5 (WS Server)  : WebSocket broadcast to clients
Thread 6 (HTTP)       : Serve Web UI + audio files
```

## Models

| Model | Task | Device | RTF | OpenVINO |
|-------|------|--------|-----|----------|
| SileroVAD | Voice activity detection | CPU | <0.01 | No |
| Paraformer Online | Streaming ASR (partial) | CPU | ~0.15 | No |
| Paraformer Offline | Sentence ASR (fallback) | CPU | ~0.09 | No |
| Qwen3-ASR-0.6B | Accurate ASR (final) | iGPU | ~0.3-0.5 | Yes |
| Opus-MT | Translation zh↔en | CPU | ~0.05 | No |
| MeloTTS | Speech synthesis | CPU | ~0.3-0.5 | No |

## Quick Start

### Install

```bash
python install.py
```

This downloads all dependencies and models (~5-10 minutes on first run).

### Run

```bash
# Chinese → English (default)
python main.py

# English → Chinese
python main.py --from en --to zh

# Without WebSocket server
python main.py --no-ws
```

### Access

- **Web UI**: http://127.0.0.1:8766
- **WebSocket**: ws://127.0.0.1:8765

## Project Structure

```
realtime-translator/
├── main.py                 # Entry point
├── config.py               # Configuration
├── install.py              # One-click install script
├── test_pipeline.py        # Component + integration tests
├── requirements.txt        # Pip dependencies
│
├── translator/
│   ├── live_pipeline.py    # Main orchestrator
│   ├── messages.py         # Data types (SentenceAudio, AsrResult, etc.)
│   │
│   ├── audio/              # Audio capture
│   │   ├── capture.py      # Microphone (sounddevice/WASAPI)
│   │   ├── ring_buffer.py  # 30s ring buffer
│   │   └── vad.py          # SileroVAD + sentence boundary detection
│   │
│   ├── asr/                # Speech recognition
│   │   ├── streaming.py    # Paraformer Online + Offline
│   │   ├── accurate.py     # Qwen3-ASR (OpenVINO iGPU)
│   │   ├── speaker.py      # [Reserved] Speaker ID (CAM++)
│   │   └── whisper_asr.py  # [Reserved] Whisper English ASR
│   │
│   ├── translation/        # Translation
│   │   ├── opus_mt.py      # Opus-MT (MarianMT)
│   │   └── base.py         # Interface
│   │
│   ├── tts/                # Text-to-speech
│   │   ├── melotts.py      # MeloTTS
│   │   └── base.py         # Interface
│   │
│   ├── server/             # WebSocket + HTTP
│   │   ├── ws_server.py    # WS broadcast + HTTP file server
│   │   └── web/index.html  # Web UI
│   │
│   └── utils/
│       └── audio.py        # Audio save utility
│
├── output/                 # Generated TTS audio files
└── models/                 # Model cache (if needed)
```

## Requirements

- Python 3.10+
- Windows 10/11 (WASAPI audio capture)
- Microphone
- Intel iGPU (optional, for Qwen3-ASR acceleration)

## Qwen3-ASR Setup (Optional)

For the highest accuracy ASR, Qwen3-ASR runs on Intel iGPU via OpenVINO. It requires a separate setup:

1. Model files at `D:\<username>_openvino\asr\models\Qwen3-ASR-0.6B-fp16-ov\`
2. Engine at `D:\<username>_openvino\asr\asr_engine.py`

Without this, the system automatically falls back to Paraformer Offline (CPU), which is still accurate (RTF ~0.09).

## WebSocket Protocol

Messages are JSON objects pushed to all connected clients:

```json
{"type": "partial", "text": "你好世", "speaker": null}
{"type": "final", "text": "你好世界。", "speaker": "Speaker_1"}
{"type": "translation_final", "text": "Hello world.", "src_lang": "zh", "tgt_lang": "en", "sentence_id": 1}
{"type": "tts_ready", "audio_url": "/audio/sentence_0001.wav", "sentence_id": 1}
```

## Testing

```bash
python test_pipeline.py
```

Tests all components individually and runs an end-to-end integration test with synthetic audio (no microphone required).

## License

MIT
