"""
Realtime Translator - One-click install script.

Run this on a fresh machine to install all dependencies and download all models.
Requirements: Python 3.10+ with pip.

Usage:
    python install.py

What it does:
    1. Install Python packages (pip)
    2. Download FunASR models (Paraformer Online + Offline) from ModelScope
    3. Download SileroVAD from torch.hub
    4. Download Opus-MT translation models from HuggingFace
    5. Download MeloTTS model + BERT models from HuggingFace
    6. Download NLTK data (for MeloTTS English g2p)
    7. Patch MeloTTS for Windows compatibility (Japanese MeCab DLL issue)
    8. Verify all components

Note on Qwen3-ASR:
    Qwen3-ASR requires a separate setup (asr_engine.py + model files on D:\\<user>_openvino\\asr).
    If not present, the system will fallback to Paraformer Offline automatically.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run(cmd, desc, check=True):
    """Run a shell command with description."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  [WARNING] Command returned non-zero exit code: {result.returncode}")
        return False
    return True


def pip_install(packages, desc=None):
    """Install pip packages."""
    if isinstance(packages, str):
        packages = [packages]
    pkg_str = " ".join(packages)
    run(f"{sys.executable} -m pip install {pkg_str}", desc or f"Installing {pkg_str}")


# ============================================================
# Step 1: Core Python packages
# ============================================================
print("\n" + "=" * 60)
print("  REALTIME TRANSLATOR - INSTALLATION")
print("=" * 60)

print("\n[Step 1/7] Installing Python packages...")

# PyTorch (CPU version - sufficient for our pipeline)
pip_install(
    ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"],
    "Installing PyTorch (CPU)"
)

# Core dependencies
pip_install([
    "numpy",
    "scipy",
    "sounddevice",
    "soundfile",
    "librosa",
    "websockets",
], "Installing audio + server packages")

# ASR dependencies
pip_install([
    "funasr",
    "modelscope",
], "Installing FunASR + ModelScope")

# Translation dependencies
pip_install([
    "transformers",
    "sentencepiece",
    "protobuf",
    "huggingface_hub",
], "Installing Transformers + HuggingFace")

# OpenVINO (for Qwen3-ASR on iGPU)
pip_install([
    "openvino>=2025.4",
], "Installing OpenVINO")

# MeloTTS and its dependencies
pip_install([
    "git+https://github.com/myshell-ai/MeloTTS.git",
    "--no-deps",
], "Installing MeloTTS (no-deps to avoid numpy conflict)")

pip_install([
    "cn2an",
    "pypinyin",
    "jieba",
    "pykakasi",
    "num2words",
    "cached_path",
    "nltk",
    "g2p-en",
], "Installing MeloTTS text processing dependencies")

# ============================================================
# Step 2: Download FunASR models (Paraformer)
# ============================================================
print("\n[Step 2/7] Downloading FunASR models (Paraformer Online + Offline)...")
print("  Source: ModelScope (https://www.modelscope.cn)")
print("  Models:")
print("    - iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
print("    - iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

run(
    f'{sys.executable} -c "'
    'import warnings; warnings.filterwarnings(\"ignore\"); '
    'import logging; logging.getLogger(\"modelscope\").setLevel(logging.ERROR); '
    'logging.getLogger(\"funasr\").setLevel(logging.ERROR); '
    'from funasr import AutoModel; '
    'print(\"  Downloading Paraformer Online...\"); '
    'AutoModel(model=\"iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online\", '
    'model_revision=\"v2.0.4\", disable_update=True, disable_log=True); '
    'print(\"  Downloading Paraformer Offline...\"); '
    'AutoModel(model=\"iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\", '
    'disable_update=True, disable_log=True); '
    'print(\"  [OK] FunASR models downloaded\")'
    '"',
    "Downloading Paraformer models from ModelScope",
    check=False,
)

# ============================================================
# Step 3: Download SileroVAD
# ============================================================
print("\n[Step 3/7] Downloading SileroVAD...")
print("  Source: torch.hub (https://github.com/snakers4/silero-vad)")

run(
    f'{sys.executable} -c "'
    'import torch; '
    'model, utils = torch.hub.load(repo_or_dir=\"snakers4/silero-vad\", '
    'model=\"silero_vad\", trust_repo=True); '
    'print(\"  [OK] SileroVAD downloaded\")'
    '"',
    "Downloading SileroVAD from torch.hub",
    check=False,
)

# ============================================================
# Step 4: Download Opus-MT translation models
# ============================================================
print("\n[Step 4/7] Downloading Opus-MT translation models...")
print("  Source: HuggingFace (https://huggingface.co)")
print("  Models:")
print("    - Helsinki-NLP/opus-mt-zh-en")
print("    - Helsinki-NLP/opus-mt-en-zh")

run(
    f'{sys.executable} -c "'
    'from transformers import MarianMTModel, MarianTokenizer; '
    'print(\"  Downloading opus-mt-zh-en...\"); '
    'MarianTokenizer.from_pretrained(\"Helsinki-NLP/opus-mt-zh-en\"); '
    'MarianMTModel.from_pretrained(\"Helsinki-NLP/opus-mt-zh-en\"); '
    'print(\"  Downloading opus-mt-en-zh...\"); '
    'MarianTokenizer.from_pretrained(\"Helsinki-NLP/opus-mt-en-zh\"); '
    'MarianMTModel.from_pretrained(\"Helsinki-NLP/opus-mt-en-zh\"); '
    'print(\"  [OK] Opus-MT models downloaded\")'
    '"',
    "Downloading Opus-MT from HuggingFace",
    check=False,
)

# ============================================================
# Step 5: Download MeloTTS model + BERT
# ============================================================
print("\n[Step 5/7] Downloading MeloTTS models...")
print("  Source: HuggingFace")
print("  Models:")
print("    - myshell-ai/MeloTTS-English (TTS checkpoint)")
print("    - bert-base-uncased (English BERT for prosody)")

run(
    f'{sys.executable} -c "'
    'import os; os.environ[\"HF_HUB_DISABLE_SYMLINKS_WARNING\"]=\"1\"; '
    'from melo.api import TTS; '
    'print(\"  Loading MeloTTS English (downloads model on first use)...\"); '
    'tts = TTS(language=\"EN\", device=\"cpu\"); '
    'print(\"  [OK] MeloTTS English model ready\")'
    '"',
    "Downloading MeloTTS English model",
    check=False,
)

# ============================================================
# Step 6: Download NLTK data
# ============================================================
print("\n[Step 6/7] Downloading NLTK data...")
print("  Packages: averaged_perceptron_tagger, averaged_perceptron_tagger_eng, cmudict")

run(
    f'{sys.executable} -c "'
    'import nltk; '
    'nltk.download(\"averaged_perceptron_tagger\", quiet=True); '
    'nltk.download(\"averaged_perceptron_tagger_eng\", quiet=True); '
    'nltk.download(\"cmudict\", quiet=True); '
    'print(\"  [OK] NLTK data downloaded\")'
    '"',
    "Downloading NLTK data",
    check=False,
)

# ============================================================
# Step 7: Patch MeloTTS for Windows (Japanese MeCab issue)
# ============================================================
print("\n[Step 7/7] Patching MeloTTS for Windows compatibility...")

try:
    import melo
    melo_dir = Path(melo.__file__).parent / "text"

    # Patch cleaner.py - make japanese/korean/french/spanish imports optional
    cleaner_path = melo_dir / "cleaner.py"
    if cleaner_path.exists():
        content = cleaner_path.read_text(encoding="utf-8")
        if "from . import chinese, japanese, english" in content:
            new_content = '''from . import chinese, english, chinese_mix
from . import cleaned_text_to_sequence
import copy

try:
    from . import japanese
except ImportError:
    japanese = None
try:
    from . import korean
except ImportError:
    korean = None
try:
    from . import french
except ImportError:
    french = None
try:
    from . import spanish
except ImportError:
    spanish = None

language_module_map = {"ZH": chinese, "EN": english, 'ZH_MIX_EN': chinese_mix}
if japanese:
    language_module_map["JP"] = japanese
if korean:
    language_module_map["KR"] = korean
if french:
    language_module_map["FR"] = french
if spanish:
    language_module_map["SP"] = spanish
    language_module_map["ES"] = spanish


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
'''
            cleaner_path.write_text(new_content, encoding="utf-8")
            print("  [OK] Patched cleaner.py")

    # Patch japanese.py - guard MeCab and top-level tokenizer load
    japanese_path = melo_dir / "japanese.py"
    if japanese_path.exists():
        content = japanese_path.read_text(encoding="utf-8")
        needs_patch = False

        # Patch 1: MeCab import
        if "raise ImportError(\"Japanese requires mecab-python3" in content:
            content = content.replace(
                'except ImportError as e:\n    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e',
                'except (ImportError, OSError):\n    MeCab = None'
            )
            needs_patch = True

        # Patch 2: num2words import
        if "from num2words import num2words" in content and "try:" not in content.split("from num2words")[0][-20:]:
            content = content.replace(
                "from num2words import num2words",
                "try:\n    from num2words import num2words\nexcept ImportError:\n    num2words = None"
            )
            needs_patch = True

        # Patch 3: _TAGGER = MeCab.Tagger()
        if "_TAGGER = MeCab.Tagger()" in content:
            content = content.replace(
                "_TAGGER = MeCab.Tagger()",
                "_TAGGER = MeCab.Tagger() if MeCab else None"
            )
            needs_patch = True

        # Patch 4: top-level pykakasi + tokenizer init
        if "from pykakasi import kakasi\n" in content and "_japanese_available" not in content:
            old_block = '''from pykakasi import kakasi
# Initialize kakasi object
kakasi = kakasi()
# Set options for converting Chinese characters to Katakana
kakasi.setMode("J", "K")  # Chinese to Katakana
kakasi.setMode("H", "K")  # Hiragana to Katakana
# Convert Chinese characters to Katakana
conv = kakasi.getConverter()

def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    res = conv.do(res)
    return res


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word'''

            new_block = '''def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


# Guard all Japanese-specific initialization (requires MeCab, fugashi, pykakasi)
_japanese_available = MeCab is not None
_kakasi_conv = None
_jp_tokenizer = None

if _japanese_available:
    try:
        from pykakasi import kakasi as _kakasi_cls
        _kks = _kakasi_cls()
        _kks.setMode("J", "K")
        _kks.setMode("H", "K")
        _kakasi_conv = _kks.getConverter()

        model_id = 'tohoku-nlp/bert-base-japanese-v3'
        _jp_tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        _japanese_available = False


def text_normalize(text):
    if not _japanese_available:
        return text
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    res = _kakasi_conv.do(res)
    return res'''

            if old_block in content:
                content = content.replace(old_block, new_block)
                needs_patch = True

        # Patch 5: tokenizer at bottom
        if "tokenizer = AutoTokenizer.from_pretrained(model_id)" in content and "_jp_tokenizer" not in content:
            content = content.replace(
                "model_id = 'tohoku-nlp/bert-base-japanese-v3'\ntokenizer = AutoTokenizer.from_pretrained(model_id)",
                "# tokenizer loaded above in guarded block"
            )
            needs_patch = True

        # Patch 6: g2p function referencing tokenizer
        if "tokenized = tokenizer.tokenize(norm_text)" in content:
            content = content.replace(
                "def g2p(norm_text):\n\n    tokenized = tokenizer.tokenize(norm_text)",
                "def g2p(norm_text):\n    if not _japanese_available or _jp_tokenizer is None:\n        return [], [], []\n    tokenized = _jp_tokenizer.tokenize(norm_text)"
            )
            needs_patch = True

        if needs_patch:
            japanese_path.write_text(content, encoding="utf-8")
            print("  [OK] Patched japanese.py")

    print("  [OK] MeloTTS patching complete")
except ImportError:
    print("  [SKIP] MeloTTS not installed, skipping patch")
except Exception as e:
    print(f"  [WARNING] Patch failed: {e}")


# ============================================================
# Verification
# ============================================================
print("\n" + "=" * 60)
print("  VERIFICATION")
print("=" * 60)

checks = [
    ("torch", "import torch; print(f'  PyTorch {torch.__version__}')"),
    ("sounddevice", "import sounddevice; print(f'  sounddevice {sounddevice.__version__}')"),
    ("funasr", "import funasr; print(f'  FunASR {funasr.__version__}')"),
    ("openvino", "import openvino; print(f'  OpenVINO {openvino.__version__}')"),
    ("transformers", "import transformers; print(f'  Transformers {transformers.__version__}')"),
    ("MeloTTS", "from melo.api import TTS; print('  MeloTTS OK')"),
    ("websockets", "import websockets; print(f'  websockets {websockets.__version__}')"),
]

all_ok = True
for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        all_ok = False

print()
if all_ok:
    print("  ALL CHECKS PASSED!")
    print()
    print("  To run the translator:")
    print(f"    cd {PROJECT_DIR}")
    print("    python main.py --from zh --to en")
    print()
    print("  Web UI: http://127.0.0.1:8766")
    print("  WebSocket: ws://127.0.0.1:8765")
else:
    print("  Some checks failed. Review the output above.")

print()
print("=" * 60)
print("  MODELS SUMMARY")
print("=" * 60)
print("""
  Model                     | Source        | Device | OpenVINO
  --------------------------+---------------+--------+---------
  SileroVAD                 | torch.hub     | CPU    | No
  Paraformer Online         | ModelScope    | CPU    | No
  Paraformer Offline        | ModelScope    | CPU    | No
  Qwen3-ASR-0.6B (optional)| Manual setup  | iGPU   | Yes
  Opus-MT zh-en / en-zh     | HuggingFace   | CPU    | No
  MeloTTS English           | HuggingFace   | CPU    | No
  bert-base-uncased (BERT)  | HuggingFace   | CPU    | No

  Download locations:
  - torch.hub cache:  ~/.cache/torch/hub/
  - ModelScope cache: ~/.cache/modelscope/hub/models/
  - HuggingFace cache: ~/.cache/huggingface/hub/
  - NLTK data: ~/AppData/Roaming/nltk_data/ (Windows)
               ~/nltk_data/ (Linux/Mac)

  Qwen3-ASR setup (optional, for highest accuracy):
    Requires separate installation at D:\\<username>_openvino\\asr\\
    with asr_engine.py and Qwen3-ASR-0.6B-fp16-ov model.
    Without it, system falls back to Paraformer Offline (still accurate).
""")
print("  Installation complete!")
print("=" * 60)
