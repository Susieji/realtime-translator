"""Realtime Translator - Main entry point.

Dual-model ASR with live translation and TTS:
- Paraformer-zh (CPU): real-time streaming partial output
- Qwen3-ASR (iGPU): accurate per-sentence final output
- Opus-MT (CPU): translation zh↔en
- MeloTTS (CPU): speech synthesis

Architecture:
  Mic → VAD → Paraformer (partial) → Display/WebSocket
               └→ Qwen3-ASR (final) → Opus-MT → MeloTTS → Audio
"""

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from translator.live_pipeline import LivePipeline


def main():
    parser = argparse.ArgumentParser(description="Realtime speech translator")
    parser.add_argument("--from", dest="src_lang", default="zh", choices=["zh", "en"],
                        help="Source language (default: zh)")
    parser.add_argument("--to", dest="tgt_lang", default="en", choices=["zh", "en"],
                        help="Target language (default: en)")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    args = parser.parse_args()

    print("=" * 60)
    print("  Realtime Translator")
    print(f"  {args.src_lang} → {args.tgt_lang}")
    print("=" * 60)
    print()

    app = LivePipeline(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        enable_ws=not args.no_ws,
    )

    print("  Loading models...")
    app.load_models()
    print()

    if not args.no_ws:
        print(f"  WebSocket: ws://{config.WS_HOST}:{config.WS_PORT}")
        print(f"  Web UI:    http://{config.WS_HOST}:{config.WS_PORT + 1}")
    print()
    print("  Ready! Speak into the microphone. Press Ctrl+C to stop.")
    print("-" * 60)

    app.start()

    print()
    print("-" * 60)
    print("  Done.")


if __name__ == "__main__":
    main()
