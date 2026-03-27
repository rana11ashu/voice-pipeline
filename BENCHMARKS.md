# Benchmark Results

Hardware: **NVIDIA RTX 5090** (sm_120 / Blackwell), 32GB GDDR7
Software: PyTorch 2.x, bfloat16, no flash-attn (sm_120 not yet supported)

---

## 1. Megakernel Decode Speed

**Script:** `benchmarks/bench_kernel.py`
**What it measures:** Raw CUDA kernel throughput — 500 standalone decode steps, no TTS pipeline.

| Metric | Result | Target |
|---|---|---|
| Decode speed | **1035 tok/s** | ~1000 tok/s |
| Per-token latency | **0.97 ms** | ~1 ms |
| Total (500 steps) | **483 ms** | — |

**Status: ✅ PASS**

---

## 2. TTS Pipeline — TTFC and RTF

**Script:** `benchmarks/bench_tts.py`
**What it measures:** Full TTS pipeline on a fixed sentence ("Hello, how are you today?"), 10 trials.
**TTFC** = time from `generate_streaming(text)` call to first audio chunk yielded.
**RTF** = total synthesis time / total audio duration.

| Metric | Median | p95 | Target | Status |
|---|---|---|---|---|
| TTFC | **165 ms** | 174 ms | < 90 ms | ❌ |
| RTF | **1.47×** | 1.50× | < 0.3× | ❌ |

**Why targets are missed — bottleneck analysis:**

Each 80ms audio frame requires:

```
Main talker  (Qwen3-0.6B, megakernel):  1 step  ×  1ms   =   1ms/frame
code_predictor (5-layer transformer):  15 steps × 3.6ms  =  54ms/frame  ← bottleneck
Vocoder:                                                       5ms/frame
──────────────────────────────────────────────────────────────────────
Total:                                                        60ms/frame
RTF = 60ms / 80ms = 0.75×  (with megakernel fully integrated)
```

The megakernel handles the main talker transformer (1ms/step). The bottleneck is `code_predictor` — a separate submodel that runs **15 sequential autoregressive steps** per audio frame. It is not the talker backbone and cannot be replaced by the megakernel.

**Path to hitting targets:**

| Optimisation | code_predictor | Total/frame | RTF |
|---|---|---|---|
| Current (PyTorch, no megakernel in live path) | 54ms | 82ms | **1.47×** |
| Megakernel integrated into live path | 54ms | 60ms | **0.75×** |
| + flash-attn (15 × 1.2ms) | 18ms | 24ms | **0.30×** ✓ |
| + flash-attn + megakernel | 18ms | 20ms | **0.25×** ✓ |

flash-attn was not available for sm_120 (Blackwell) at time of testing.

---

## 3. End-to-End Voice Pipeline Latency

**Measured:** Full round-trip from end of user speech to first audio heard.
**Setup:** Mac mic → OpenAI Whisper STT → GPT-4o-mini → GPU Qwen3-TTS server → Mac speakers.

| Stage | Latency | Notes |
|---|---|---|
| VAD silence detection | ~500 ms | Configurable; `vad_stop_secs=0.5` |
| Whisper STT (OpenAI) | ~200 ms | Streaming, network dependent |
| GPT-4o-mini first token | ~300 ms | Network dependent |
| Qwen3-TTS first audio chunk | ~165 ms | GPU generation |
| **Total** | **~1.2 s** | End of speech → first audio |

**Note:** TTS TTFC (165ms) is the smallest contributor. The dominant factors are VAD (500ms) and LLM (300ms), both of which are independent of this project's scope.

---

## 4. Parallel Code Predictor Experiment

Attempted to collapse 15 sequential code_predictor passes → 1 by applying all `lm_head[i]` in parallel.

| Metric | Autoregressive (baseline) | Parallel (experiment) |
|---|---|---|
| TTFC | 390 ms | **94 ms** |
| RTF | 1.47× | **0.53×** |
| Chunks generated | 20 (correct) | 4095 (runaway — no EOS) |
| Audio quality | Clear | Breaks after ~1s |

Parallel prediction was reverted. The acoustic codes feed back as the embedding input for the next main transformer step — wrong codes corrupt this feedback loop and the model never generates EOS. Full analysis in `Tradeoffs.md`.
