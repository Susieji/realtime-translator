"""Accurate ASR using Qwen3-ASR on Intel iGPU via OpenVINO."""

import sys
import os
import json
import string
import threading
import importlib.util
import numpy as np
from pathlib import Path
from typing import Optional

import config
from translator.messages import SentenceAudio, AsrResult


def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class AccurateAsr:
    """Qwen3-ASR running on iGPU for accurate per-sentence transcription."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load(self):
        """Load Qwen3-ASR OpenVINO model."""
        model_dir = self._find_model_dir()
        if model_dir is None:
            raise FileNotFoundError("Qwen3-ASR model not found.")

        engine_dir = self._find_engine_dir()
        if engine_dir is None:
            raise FileNotFoundError("asr_engine.py not found.")

        qwen_asr_pkg = engine_dir / "Qwen3-ASR"
        utils_file = qwen_asr_pkg / "qwen_asr" / "inference" / "utils.py"
        processor_file = qwen_asr_pkg / "qwen_asr" / "core" / "transformers_backend" / "processing_qwen3_asr.py"

        if utils_file.exists():
            _import_from_file("qwen_asr_utils_standalone", utils_file)

        if processor_file.exists():
            proc_mod = _import_from_file("qwen_asr_processor_standalone", processor_file)
            import types
            fake_pkg = types.ModuleType("qwen_asr")
            fake_pkg.core = types.ModuleType("qwen_asr.core")
            fake_pkg.core.transformers_backend = types.ModuleType("qwen_asr.core.transformers_backend")
            fake_pkg.core.transformers_backend.processing_qwen3_asr = proc_mod
            fake_pkg.inference = types.ModuleType("qwen_asr.inference")

            if "qwen_asr_utils_standalone" in sys.modules:
                fake_pkg.inference.utils = sys.modules["qwen_asr_utils_standalone"]
            else:
                fake_pkg.inference.utils = types.ModuleType("qwen_asr.inference.utils")

            sys.modules.setdefault("qwen_asr", fake_pkg)
            sys.modules.setdefault("qwen_asr.core", fake_pkg.core)
            sys.modules.setdefault("qwen_asr.core.transformers_backend", fake_pkg.core.transformers_backend)
            sys.modules.setdefault("qwen_asr.core.transformers_backend.processing_qwen3_asr", proc_mod)
            sys.modules.setdefault("qwen_asr.inference", fake_pkg.inference)
            sys.modules.setdefault("qwen_asr.inference.utils", fake_pkg.inference.utils)

        if str(engine_dir) not in sys.path:
            sys.path.insert(0, str(engine_dir))

        from asr_engine import OVQwen3ASRModel
        self._model = OVQwen3ASRModel.from_pretrained(str(model_dir), device=config.QWEN_DEVICE)
        self._available = True

    def transcribe(self, sentence: SentenceAudio) -> Optional[AsrResult]:
        """Transcribe a complete sentence. Thread-safe."""
        if self._model is None:
            return None

        import tempfile
        import soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, sentence.samples, config.SAMPLE_RATE)

            with self._lock:
                results = self._model.transcribe(audio=tmp_path)

            if results and len(results) > 0:
                item = results[0]
                if isinstance(item, dict):
                    text = item.get('text', '')
                elif hasattr(item, 'text'):
                    text = item.text
                else:
                    text = str(item)
                text = text.strip()
                if text:
                    return AsrResult(
                        text=text,
                        is_final=True,
                        start_ms=sentence.start_ms,
                        end_ms=sentence.end_ms,
                    )
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return None

    def _find_model_dir(self) -> Optional[Path]:
        username = os.environ.get("USERNAME", "user").lower()
        model_name = "Qwen3-ASR-0.6B-fp16-ov"

        state = self._find_state()
        if state and "MODEL_DIR" in state:
            p = Path(state["MODEL_DIR"]) / model_name
            if p.exists() and (p / "config.json").exists():
                return p

        for d in string.ascii_uppercase:
            p = Path(f"{d}:\\{username}_openvino") / "asr" / "models" / model_name
            if p.exists() and (p / "config.json").exists():
                return p
        return None

    def _find_engine_dir(self) -> Optional[Path]:
        username = os.environ.get("USERNAME", "user").lower()

        state = self._find_state()
        if state and "ASR_DIR" in state:
            p = Path(state["ASR_DIR"])
            if (p / "asr_engine.py").exists():
                return p

        for d in string.ascii_uppercase:
            p = Path(f"{d}:\\{username}_openvino") / "asr"
            if (p / "asr_engine.py").exists():
                return p
        return None

    def _find_state(self) -> Optional[dict]:
        username = os.environ.get("USERNAME", "user").lower()
        for d in string.ascii_uppercase:
            sf_path = Path(f"{d}:\\{username}_openvino") / "asr" / "state.json"
            if sf_path.exists():
                return json.loads(sf_path.read_text(encoding="utf-8"))
        return None
