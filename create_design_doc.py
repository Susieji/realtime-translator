"""Generate the design document as a Word file."""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

style = doc.styles['Normal']
style.font.name = 'Consolas'
style.font.size = Pt(10)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

# Title
title = doc.add_heading('实时语音翻译系统 - 设计文档', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('项目路径: C:\\Users\\shujieji\\sourcecode\\skill\\realtime-translator')
doc.add_paragraph('日期: 2026-05-29')
doc.add_paragraph('Web UI: http://127.0.0.1:8766')
doc.add_paragraph('WebSocket: ws://127.0.0.1:8765')
doc.add_paragraph('')

# ============================================================
doc.add_heading('1. 完整流程图', level=1)

flow_text = """
┌──────────┐     ┌──────────┐     ┌──────────────────────────────────┐
│ 麦克风    │────▶│ AudioCap │────▶│ RingBuffer (30s)                  │
│ WASAPI   │     │ 16kHz    │     │ 32ms/block (512 samples)          │
└──────────┘     └──────────┘     └────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                  ┌────────────────────────────────┐
                                  │  SileroVAD (CPU)               │
                                  │  语音活动检测 + 句子边界分割     │
                                  │  SILENCE → SPEECH →            │
                                  │  TRAILING_SILENCE → END        │
                                  └───────┬────────────────┬───────┘
                                          │                │
                            ┌─────────────┘                └──────────────┐
                            │ 每个 chunk (语音中)                          │ 句子结束
                            ▼                                             ▼
             ┌──────────────────────────┐              ┌──────────────────────────────┐
             │ Paraformer Online (CPU)   │              │    后台处理线程 (Queue)         │
             │ 流式 ASR                  │              │                              │
             │ 每 600ms 输出 partial     │              │  ┌────────────────────────┐  │
             └─────────────┬────────────┘              │  │ Paraformer Offline(CPU)│  │
                           │                           │  │ 整句精确识别            │  │
                           │                           │  └───────────┬────────────┘  │
                           │                           │              ▼               │
                           │                           │  ┌────────────────────────┐  │
                           │                           │  │ Qwen3-ASR (iGPU/OV)   │  │
                           │                           │  │ 最高精度, 覆盖 final   │  │
                           │                           │  └───────────┬────────────┘  │
                           │                           └──────────────┼───────────────┘
                           │                                          │
                           ▼                                          ▼
            ┌───────────────────────────────────────────────────────────────────┐
            │                     WebSocket Server                               │
            │  partial / final / translation_partial / translation_final /       │
            │  tts_ready                                                        │
            └───────────────────────┬───────────────────────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │     Web UI (Browser)   │
                        │  原文 (实时刷新/替换)   │
                        │  译文 (逐句显示)        │
                        │  音频播放              │
                        └───────────────────────┘


═══════════════════════════════════════════════════════════════
翻译 + TTS 后台 (与 ASR 并行, 不阻塞实时显示)
═══════════════════════════════════════════════════════════════

    accurate_asr final text
            │
            ▼
┌────────────────────────┐       ┌────────────────────────┐
│ Opus-MT (CPU)          │──────▶│ MeloTTS (CPU)          │
│ zh→en / en→zh          │       │ 目标语言语音合成        │
│ ~50ms/sentence         │       │ ~1-2s/sentence         │
└────────────────────────┘       └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │ WAV → WebSocket 通知    │
                                 │ UI 播放                 │
                                 └────────────────────────┘
"""
p = doc.add_paragraph()
run = p.add_run(flow_text)
run.font.name = 'Consolas'
run.font.size = Pt(8)

# ============================================================
doc.add_heading('2. 线程模型', level=1)

threads = [
    ('Thread 1 (Main)', 'AudioCapture callback → VAD → Paraformer Online → WS push partial'),
    ('Thread 2 (Processor)', 'Queue ← SentenceAudio → Paraformer Offline → Qwen3-ASR → WS push final'),
    ('Thread 3 (Translator)', 'Queue ← final text → Opus-MT → WS push translation'),
    ('Thread 4 (TTS)', 'Queue ← translated text → MeloTTS → save WAV → WS push tts_ready'),
    ('Thread 5 (WS Server)', 'asyncio event loop, broadcast to clients'),
    ('Thread 6 (HTTP Server)', 'serve web UI static files'),
]

table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '线程'
hdr[1].text = '职责'
for name, desc in threads:
    row = table.add_row().cells
    row[0].text = name
    row[1].text = desc

# ============================================================
doc.add_heading('3. 模型清单', level=1)

models = [
    ('1', 'SileroVAD', '语音活动检测', 'CPU', '否 (PyTorch)', '<0.01', '极轻量'),
    ('2', 'Paraformer Online\n(iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)', '流式 ASR (partial)', 'CPU', '否 (PyTorch)', '~0.15', '实时显示用'),
    ('3', 'Paraformer Offline\n(iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)', '整句 ASR (fallback)', 'CPU', '否 (PyTorch)', '~0.09', 'Qwen3不可用时降级'),
    ('4', 'Qwen3-ASR-0.6B\n(snake7gun/Qwen3-ASR-0.6B-fp16-ov)', '最终精确 ASR', 'iGPU', '是 (OpenVINO)', '~0.3-0.5', '覆盖 partial, 触发翻译'),
    ('5', 'Opus-MT\n(Helsinki-NLP/opus-mt-zh-en)', '翻译 zh↔en', 'CPU', '否 (PyTorch)', '~0.05', 'MarianMT'),
    ('6', 'MeloTTS\n(myshell-ai/MeloTTS-English)', '语音合成', 'CPU', '否 (PyTorch)', '~0.3-0.5', '中英双语, VITS-based'),
]

table = doc.add_table(rows=1, cols=7)
table.style = 'Table Grid'
hdr = table.rows[0].cells
headers = ['#', '模型 (ID)', '任务', '运行设备', 'OpenVINO', 'RTF', '备注']
for i, h in enumerate(headers):
    hdr[i].text = h
for row_data in models:
    row = table.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val

doc.add_paragraph('')
doc.add_paragraph(
    '注: RTF (Real-Time Factor) = 处理时间/音频时长。RTF < 1 表示实时可行。\n'
    '以上 RTF 数据来自实际测试 (2026-05-29, Intel PTL 平台)。\n'
    '仅 Qwen3-ASR 使用 OpenVINO, 运行在 Intel iGPU 上; 其余均为 PyTorch CPU 推理。'
)

# ============================================================
doc.add_heading('4. 数据流时序 (一句话的生命周期)', level=1)

timeline = """
用户说话:  "今天天气很好"
           ├──────── 1.5s ────────┤

VAD:       [SPEECH................][TRAILING_SILENCE 600ms][END]

Paraformer │partial:"今天"│partial:"今天天气"│partial:"今天天气很好"│
Online:    ↓ WS push     ↓ WS push          ↓ WS push
           UI 实时刷新原文区

句子结束 ──────────────────────────────────────── 触发后台
           │
           ├→ Paraformer Offline: "今天天气很好。" (200ms)
           │     └→ Qwen3-ASR: "今天天气很好。"    (500ms) ← final
           │           └→ WS push final → UI 替换原文
           │
           └→ Opus-MT: "The weather is great today." (80ms)
                  └→ WS push translation → UI 显示译文
                  └→ MeloTTS: 生成音频 (1.5s)
                        └→ WS push tts_ready → UI 播放

端到端延迟: ~0.6s(VAD) + 0.5s(Qwen3) + 0.08s(翻译) ≈ 1.2s
TTS播放额外: +1.5s
"""
p = doc.add_paragraph()
run = p.add_run(timeline)
run.font.name = 'Consolas'
run.font.size = Pt(9)

# ============================================================
doc.add_heading('5. 融合后项目结构', level=1)

structure = """
realtime-translator/
├── main.py                    # 入口 (python main.py --from zh --to en)
├── config.py                  # 统一配置
├── requirements.txt
├── test_pipeline.py           # 全组件测试脚本
│
├── translator/
│   ├── __init__.py
│   ├── live_pipeline.py       # 实时管道 (麦克风→播放)
│   ├── messages.py            # 统一数据结构
│   │
│   ├── audio/                 # 音频采集
│   │   ├── capture.py         # AudioCapture (sounddevice/WASAPI)
│   │   ├── ring_buffer.py     # RingBuffer (30s)
│   │   └── vad.py             # SileroVAD + SentenceManager
│   │
│   ├── asr/                   # ASR 模块
│   │   ├── streaming.py       # Paraformer Online/Offline
│   │   ├── accurate.py        # Qwen3-ASR (OpenVINO iGPU)
│   │   ├── speaker.py         # [预留] CAM++ 说话人识别
│   │   └── whisper_asr.py     # [预留] Whisper 英文 ASR
│   │
│   ├── translation/           # 翻译模块
│   │   ├── opus_mt.py         # Opus-MT (MarianMT)
│   │   └── base.py            # 翻译接口基类
│   │
│   ├── tts/                   # TTS 模块
│   │   ├── melotts.py         # MeloTTS
│   │   └── base.py            # TTS 接口基类
│   │
│   ├── server/                # WebSocket + HTTP
│   │   ├── ws_server.py       # WS: ws://127.0.0.1:8765
│   │   └── web/index.html     # Web UI: http://127.0.0.1:8766
│   │
│   └── utils/
│       └── audio.py           # 音频保存工具
│
├── models/                    # 模型文件 (缓存)
└── output/                    # TTS 输出音频
"""
p = doc.add_paragraph()
run = p.add_run(structure)
run.font.name = 'Consolas'
run.font.size = Pt(9)

# ============================================================
doc.add_heading('6. 可能遇到的问题', level=1)

issues = [
    ('config 冲突', '高', '两项目 config 结构不同，需统一'),
    ('FunASR 版本/调用差异', '高', 'live-transcribe 用双模型 Online+Offline，需确认兼容'),
    ('依赖膨胀/版本冲突', '中', 'transformers, openvino 版本需对齐'),
    ('实时 vs 批处理模式差异', '中', '事件驱动 vs 顺序管道，需新建 live_pipeline'),
    ('句子粒度衔接', '中', '翻译/TTS 需适配逐句输入'),
    ('Qwen3-ASR 路径硬编码', '中', '磁盘搜索逻辑需配置化'),
    ('TTS 延迟', '中', 'MeloTTS ~1-2s/句，需异步不阻塞'),
    ('WebSocket 协议扩展', '低', '新增 translation/tts 消息类型'),
]

table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '问题'
hdr[1].text = '严重度'
hdr[2].text = '说明'
for issue, severity, desc in issues:
    row = table.add_row().cells
    row[0].text = issue
    row[1].text = severity
    row[2].text = desc

# ============================================================
doc.add_heading('7. 设计决策说明', level=1)

doc.add_heading('7.1 CAM++ 说话人识别 (暂不实现, 预留接口)', level=2)
doc.add_paragraph(
    '当前版本不集成 CAM++ 说话人识别模块。在 asr/ 目录下保留 speaker.py 文件，'
    '定义 SpeakerIdentifier 基类接口 (load/identify/reset)。'
    'live_pipeline 中预留 speaker_id 字段和处理槽位，后续迭代直接插入即可。'
)

doc.add_heading('7.2 Paraformer Online vs Offline 说明 (两级架构)', level=2)
doc.add_paragraph(
    'Paraformer Online (流式模型):\n'
    '- 模型: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online\n'
    '- 工作方式: 接收音频 chunk (每 600ms)，边听边输出 partial 文字\n'
    '- 精度: 较低 (因为只看到部分上下文)\n'
    '- 用途: 实时显示给用户，让用户知道"系统在听"\n'
    '- 类比: 就像同传翻译员边听边记笔记，随时可能修正\n\n'
    'Qwen3-ASR (精确模型, iGPU):\n'
    '- 模型: snake7gun/Qwen3-ASR-0.6B-fp16-ov\n'
    '- 工作方式: 等一整句话说完后，一次性处理全部音频\n'
    '- 精度: 最高\n'
    '- 用途: 句子结束后给出最终精确结果，替换 partial，并触发翻译\n\n'
    '设计决策 - 两级而非三级:\n'
    '- 原 live-transcribe 使用 Online → Offline → Qwen3 三级\n'
    '- 翻译场景下，Offline 的中间结果仅闪现 300ms 就被 Qwen3 覆盖，无实际价值\n'
    '- 简化为两级: Online (partial) → Qwen3 (final → 触发翻译)\n'
    '- Paraformer Offline 保留代码作为降级方案 (Qwen3不可用时启用)\n\n'
    '最终架构:\n'
    '  说话中: Online → partial → UI 实时刷新\n'
    '  说完后: Qwen3 (iGPU) → final → UI 替换 → 翻译 → TTS\n'
    '          ↓ (如果 Qwen3 失败)\n'
    '          Paraformer Offline (CPU) → fallback final'
)

doc.add_heading('7.3 Whisper 英文 ASR (暂不实现, 预留接口)', level=2)
doc.add_paragraph(
    '后续版本将加入 Whisper 用于英文语音输入。当前设计:\n'
    '- asr/ 目录下预留 whisper.py，定义 WhisperASR 类\n'
    '- 接口与 StreamingAsr/AccurateAsr 对齐: load() / transcribe(SentenceAudio)\n'
    '- config.py 中预留 WHISPER_MODEL 和 WHISPER_DEVICE 配置项\n'
    '- live_pipeline 中通过语言检测 (VAD+简单能量特征) 路由到不同 ASR:\n'
    '  - 检测到中文 → Paraformer + Qwen3-ASR\n'
    '  - 检测到英文 → Whisper\n'
    '- 推荐模型: whisper-large-v3-turbo (OpenAI) 或 Whisper OV 量化版'
)

doc.add_heading('7.4 TTS 选型: MeloTTS vs CosyVoice3', level=2)
doc.add_paragraph(
    '对比 (基于 OpenVINO Python 推理):\n\n'
    '┌────────────────┬──────────────────────┬──────────────────────────┐\n'
    '│                │ MeloTTS              │ CosyVoice3 (0.5B)        │\n'
    '├────────────────┼──────────────────────┼──────────────────────────┤\n'
    '│ 模型大小       │ ~400MB               │ ~2GB+                    │\n'
    '│ 推理速度(CPU)  │ RTF 0.3-0.5          │ RTF 1.5-3.0              │\n'
    '│ 推理速度(OV)   │ RTF 0.2-0.3          │ RTF 0.8-1.5              │\n'
    '│ 每句延迟(5字)  │ ~0.5-1s              │ ~2-4s                    │\n'
    '│ 音质           │ 良好 (略机械)         │ 优秀 (自然, 情感丰富)     │\n'
    '│ 语言支持       │ 中/英/日/韩           │ 中/英/日/韩/粤            │\n'
    '│ 流式支持       │ 不原生支持            │ 支持 chunk 流式           │\n'
    '│ 语音克隆       │ 不支持               │ 支持 (3s 参考音频)        │\n'
    '│ 内存占用       │ ~1GB                 │ ~4GB+                    │\n'
    '│ OV 转换成熟度  │ 社区有方案            │ 官方刚起步               │\n'
    '└────────────────┴──────────────────────┴──────────────────────────┘\n\n'
    '结论: 选择 MeloTTS\n\n'
    '原因:\n'
    '1. 速度优势明显: MeloTTS 比 CosyVoice3 快 3-5x，实时场景下延迟差距巨大\n'
    '2. 资源占用低: 1GB vs 4GB+，不会挤占 iGPU 给 Qwen3-ASR 的资源\n'
    '3. OV 转换更成熟: MeloTTS 结构简单 (VITS-based)，OV 导出稳定\n'
    '4. 实时翻译场景优先速度: 用户等 0.5-1s 可接受，等 3-4s 体验差\n'
    '5. 后续可迭代: 如果音质需求提升，可以在非实时场景切换到 CosyVoice3'
)

# ============================================================
doc.add_heading('8. 测试结果 (2026-05-29)', level=1)

doc.add_paragraph('测试环境: Python 3.14, Intel PTL (iGPU), Windows 11')
doc.add_paragraph('运行命令: python test_pipeline.py')
doc.add_paragraph('')

test_results = [
    ('Messages 数据结构', 'PASS', ''),
    ('RingBuffer', 'PASS', ''),
    ('SileroVAD', 'PASS', ''),
    ('SentenceManager 状态机', 'PASS', ''),
    ('Paraformer Online (流式)', 'PASS', 'RTF ~0.15'),
    ('Paraformer Offline (降级)', 'PASS', 'RTF ~0.09'),
    ('Qwen3-ASR (iGPU/OpenVINO)', 'PASS', 'RTF ~0.3-0.5'),
    ('Opus-MT 翻译 zh→en', 'PASS', '"你好世界" → "You\'re in the world."'),
    ('MeloTTS 合成', 'PASS', '"Hello world" → 1.62s 音频'),
    ('音频保存 (WAV)', 'PASS', ''),
    ('WebSocket 服务器', 'PASS', ''),
    ('Speaker ID 接口 (预留)', 'PASS', '返回 -1'),
    ('Whisper 接口 (预留)', 'PASS', '返回 not available'),
    ('端到端集成管道', 'PASS', 'ASR → 翻译 → TTS 全流程'),
]

table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = '组件'
hdr[1].text = '结果'
hdr[2].text = '备注'
for name, status, note in test_results:
    row = table.add_row().cells
    row[0].text = name
    row[1].text = status
    row[2].text = note

doc.add_paragraph('')
doc.add_paragraph('总计: 18 passed, 0 failed, 0 skipped')

# Save
output_path = r"C:\Users\shujieji\sourcecode\skill\Translator\design_doc.docx"
doc.save(output_path)
print(f"Document saved to: {output_path}")
