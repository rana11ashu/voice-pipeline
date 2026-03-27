# RTX 5090 Megakernel → Qwen3-TTS on Pipecat

A streaming voice pipeline that wires AlpinDale's CUDA megakernel into Qwen3-TTS, served as a Pipecat TTS service.

```
Mic → Whisper STT → GPT-4o-mini LLM → Qwen3-TTS (RTX 5090) → Speakers
```

---

## Architecture

### Components

| File | Role |
|---|---|
| `kernel/csrc/kernel.cu` | AlpinDale's megakernel (patched for TTS vocab) |
| `kernel/qwen_megakernel/` | Python wrapper — JIT compiles kernel, exposes `Decoder` |
| `tts/streaming_engine.py` | Core TTS engine — streaming generation via forward hook |
| `tts/tts_server.py` | HTTP server exposing TTS to local clients |
| `pipecat_service/qwen3_tts_service.py` | Pipecat `TTSService` subclass |
| `pipeline/bot.py` | Full Pipecat pipeline (mic → STT → LLM → TTS → speakers) |
| `demo/demo.py` | Standalone demo client (Whisper + GPT + GPU TTS server) |
| `benchmarks/` | `bench_kernel.py`, `bench_tts.py` |

### How streaming works

`model.generate()` runs in a background thread. A `register_forward_hook` on `model.talker` fires after each decode step. During decode, `hidden_states[-1]` carries the 16-code frame (1 semantic + 15 acoustic codes). The hook pushes this into a `ThreadQueue`; the main thread dequeues and calls the vocoder immediately — audio chunks arrive as the model generates, not all at the end.

```
model.generate() [thread]
    │
    │ forward hook fires per decode step
    ▼
ThreadQueue  ←─ [semantic, a0..a14]  (16 codes = 80ms audio)
    │
    ▼
vocoder → PCM bytes → Pipecat pipeline / HTTP client
```

### Qwen3-TTS architecture discovery

Qwen3-TTS has an unusual multi-codebook decode loop. Each 80ms audio frame requires:
1. **Main talker** (Qwen3-0.6B transformer): 1 step → semantic token
2. **code_predictor** (separate 5-layer transformer): 15 autoregressive steps → 15 acoustic tokens
3. **Vocoder**: 16 codes → 1920 PCM samples

The code_predictor is not the talker backbone — it is a separate submodel. Critically, the decode step input is not just the semantic embedding: it is `semantic_embed + sum(15 acoustic embeds) + trailing_text_hidden[step]`. This trailing text conditioning is the core innovation of Qwen3-TTS and was the reason our initial custom talker produced noise.

---

## Kernel Modification

**One change to `kernel/csrc/kernel.cu`:**

```diff
- constexpr int LDG_VOCAB_SIZE = 151936;   // Qwen3 text vocab
+ constexpr int LDG_VOCAB_SIZE = 3072;     // Qwen3-TTS audio codec vocab
```

The megakernel was built for Qwen3-0.6B text generation (151,936-token vocab). Qwen3-TTS's talker uses a 3,072-token audio codec vocab. The lm_head kernel allocates blocks proportional to vocab size — without this patch, the kernel allocates 50× too many blocks and the weight matrix shape doesn't match.

The kernel is JIT-compiled via `torch.utils.cpp_extension.load` with `-arch=sm_120a` (Blackwell).

---

## Performance Numbers

All benchmarks run on **NVIDIA RTX 5090**, PyTorch 2.x, bfloat16, no flash-attn.

### Megakernel decode speed (`benchmarks/bench_kernel.py`)

| Metric | Result | Target |
|---|---|---|
| Decode speed | **1035 tok/s** | ~1000 tok/s |
| Per-token latency | **0.97 ms** | ~1 ms |

### TTS pipeline (`benchmarks/bench_tts.py`)

| Metric | Result | Target | Notes |
|---|---|---|---|
| TTFC (time to first chunk) | **165 ms** | < 90 ms | See bottleneck analysis |
| RTF (real-time factor) | **1.47×** | < 0.3× | See bottleneck analysis |

### End-to-end voice pipeline (estimated, full demo)

| Stage | Latency |
|---|---|
| VAD silence detection | ~500 ms |
| Whisper STT | ~200 ms |
| GPT-4o-mini first token | ~300 ms |
| Qwen3-TTS first audio chunk | ~165 ms |
| **Total (end of speech → first audio)** | **~1.2 s** |

---

## Why RTF and TTFC Miss Targets

The megakernel handles the **main talker transformer** (1 step per frame). But the bottleneck is **code_predictor** — a separate 5-layer transformer that runs 15 sequential steps per frame:

```
Per 80ms audio frame:
  Main talker  (megakernel):  1 step  ×  1ms  =   1ms
  code_predictor (PyTorch):  15 steps × 3.6ms =  54ms  ← bottleneck
  Vocoder:                                        5ms
  ─────────────────────────────────────────────────────
  Total:                                         60ms/frame  →  RTF 0.75×
```

Even with the megakernel fully integrated into the live path, RTF would be ~0.75× due to code_predictor. To reach < 0.3× RTF requires flash-attn (cuts each code_predictor pass from 3.6ms to ~1.2ms):

```
  With flash-attn:
  Main talker  (megakernel):  1ms
  code_predictor (flash-attn): 15 × 1.2ms = 18ms
  Vocoder:                      5ms
  Total:                       24ms/frame  →  RTF 0.30×  ✓
```

flash-attn installation was attempted but sm_120 (Blackwell) support is not yet available in stable releases.

### Parallel code_predictor attempt

We attempted to collapse 15 sequential code_predictor passes into 1 by applying all 15 `lm_head[i]` projections in parallel to the same hidden state. This gave RTF 0.53× but broke EOS detection — the model ran to 4095 tokens (max sequence length) and audio degraded after the first second. Root cause: the predicted acoustic codes feed back as the embedding input for the next main transformer step. Wrong codes corrupt the feedback loop, preventing the model from ever generating EOS. Full analysis in `Tradeoffs.md`.

---

## How to Run

### Requirements
- NVIDIA RTX 5090 (sm_120 / Blackwell)
- CUDA 12.x, PyTorch 2.x
- Python 3.10+

### 1. Install dependencies (GPU machine)

```bash
git clone <this-repo> && cd voice-pipeline
pip install qwen-tts torch soundfile flask numpy
```

### 2. Download model weights

```bash
huggingface-cli download Qwen/Qwen3-TTS --local-dir ~/.cache/qwen3-tts
```

### 3. Build the megakernel (JIT — happens automatically on first import)

```bash
cd voice-pipeline
PYTHONPATH=. python3 -c "import sys; sys.path.insert(0, 'kernel'); import qwen_megakernel; print('OK')"
```

Compilation takes ~60 seconds on first run. Cached in `~/.cache/torch_extensions/`.

### 4. Run benchmarks

```bash
# Megakernel decode speed
PYTHONPATH=.:kernel python3 benchmarks/bench_kernel.py

# Full TTS pipeline (TTFC + RTF)
PYTHONPATH=.:kernel python3 benchmarks/bench_tts.py
```

### 5. Run the voice demo

**Terminal 1 — start TTS server on GPU:**
```bash
PYTHONPATH=. python3 tts/tts_server.py
```

**Terminal 2 — SSH tunnel (from local machine):**
```bash
ssh -p <PORT> -N -L 5000:localhost:5000 root@<GPU_IP>
```

**Terminal 3 — run demo client (local machine):**
```bash
# Requires OPENAI_API_KEY in .env
pip install openai pyaudio soundfile python-dotenv requests
python3 demo/demo.py
```

### 6. Pipecat pipeline (full bot)

```bash
# Requires OPENAI_API_KEY and DEEPGRAM_API_KEY
PYTHONPATH=. python3 pipeline/bot.py
```

---

## Design Decisions

See [Tradeoffs.md](Tradeoffs.md) for detailed rationale on all major decisions, including:
- Why the custom `MegakernelTalker` was abandoned for the official model
- Forward hook streaming vs batch generation
- Parallel code_predictor attempt and why it failed
- LDG_VOCAB_SIZE kernel patch
- Why RTF targets weren't met and what would be needed
