"""WebSocket server for pushing results to clients + HTTP server for web UI."""

import os
import json
import threading
import queue
import asyncio
from pathlib import Path
from typing import Set

os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",127.0.0.1,localhost"

import config

_WEB_DIR = Path(__file__).parent / "web"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


class WsServer:
    """WebSocket server that broadcasts results + HTTP server for web UI."""

    def __init__(self):
        self._output_queue: queue.Queue = queue.Queue()
        self._thread = None
        self._running = False
        self._clients: Set = set()
        self._clients_lock = threading.Lock()

    def push(self, msg: dict):
        """Push a message dict to all connected clients."""
        self._output_queue.put(msg)

    def push_asr(self, text: str, is_final: bool, speaker_id: int = -1,
                 start_ms: int = 0, end_ms: int = 0):
        """Push ASR result."""
        self.push({
            "type": "final" if is_final else "partial",
            "text": text,
            "speaker": f"Speaker_{speaker_id + 1}" if speaker_id >= 0 else None,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    def push_translation(self, text: str, src_lang: str, tgt_lang: str,
                         sentence_id: int = 0):
        """Push translation result."""
        self.push({
            "type": "translation_final",
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "sentence_id": sentence_id,
        })

    def push_tts(self, audio_filename: str, sentence_id: int = 0):
        """Push TTS completion notification. audio_filename is just the filename, not full path."""
        self.push({
            "type": "tts_ready",
            "audio_url": f"/audio/{audio_filename}",
            "sentence_id": sentence_id,
        })

    def start(self):
        """Start the WebSocket server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        import websockets

        async def handler(websocket):
            with self._clients_lock:
                self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                with self._clients_lock:
                    self._clients.discard(websocket)

        async def broadcaster():
            while self._running:
                messages = []
                while True:
                    try:
                        messages.append(self._output_queue.get_nowait())
                    except queue.Empty:
                        break

                if messages:
                    with self._clients_lock:
                        clients_snapshot = list(self._clients)
                    for msg in messages:
                        payload = json.dumps(msg, ensure_ascii=False)
                        for ws in clients_snapshot:
                            try:
                                await ws.send(payload)
                            except Exception:
                                with self._clients_lock:
                                    self._clients.discard(ws)

                await asyncio.sleep(0.02)

        server = await websockets.serve(handler, config.WS_HOST, config.WS_PORT)
        await broadcaster()
        server.close()

    def _run_http(self):
        """HTTP server for web UI + audio files."""
        import http.server
        import urllib.parse

        web_dir = str(_WEB_DIR)
        output_dir = str(_OUTPUT_DIR)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def translate_path(self, path):
                path = urllib.parse.unquote(urllib.parse.urlparse(path).path)
                if path.startswith("/audio/"):
                    filename = path[len("/audio/"):]
                    return os.path.join(output_dir, filename)
                # Default: serve from web dir
                if path == "/":
                    path = "/index.html"
                return os.path.join(web_dir, path.lstrip("/"))

            def log_message(self, format, *args):
                pass  # suppress noisy logs

        port = config.WS_PORT + 1
        httpd = http.server.HTTPServer((config.WS_HOST, port), Handler)
        httpd.timeout = 1
        while self._running:
            httpd.handle_request()
