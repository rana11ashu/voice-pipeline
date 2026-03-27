# Design Decisions & Tradeoffs

---

## 1. Official qwen-tts model vs. custom MegakernelTalker

**Decision:** Use the official `Qwen3TTSModel` for generation, not our custom `MegakernelTalker`.

**What we tried first:** Built `tts/megakernel_talker.py` — a hand-rolled transformer that called the megakernel for each decode step. Cleaner integration, should have been faster.

**Why it failed:** Qwen3-TTS has an unusual architecture: at each decode step, the talker's input is not just `embed_weight[semantic_token]`. It's the *sum* of the semantic embedding + 15 acoustic code embeddings (from a separate `code_predictor` submodel) + a slice of `trailing_text_hidden` (remaining text tokens streamed in at each step). Our custom talker used only the semantic embedding, so every frame was garbage noise.

**Tradeoff accepted:** We lose the ability to directly swap in the megakernel for generation. The official model handles all the embedding math correctly and produces clear speech. Speed is lower but correctness is non-negotiable.

---

## 2. Forward hook streaming vs. batch generation

**Decision:** Register a `register_forward_hook` on `model.talker` to intercept each decode step's output codes, yielding audio in real-time. `model.generate()` runs in a background thread.

**Alternative:** Run `model.generate()` to completion, then decode all frames at once.

**Why we chose streaming:** For a voice assistant, perceived latency matters more than throughput. With streaming, the first audio chunk arrives at ~165ms and playback starts while generation continues. With batch, the user waits for the entire response before hearing anything.

**Tradeoff accepted:** Per-frame vocoder decode (18.6ms/frame warm) vs. batch decode (1.3ms/frame). Streaming adds ~13ms overhead per frame. RTF is worse (~1.47x vs. ~0.3x batch), but the user hears audio immediately.

---

## 3. Per-frame vocoder vs. batch vocoder

**Decision:** Decode each 16-code frame through the vocoder immediately as it arrives.

**Alternative:** Collect all frames, then batch-decode with a single vocoder call.

**Measured difference:** Per-frame = 18.6ms/frame; batch = 1.3ms/frame (14× faster).

**Why we chose per-frame:** Streaming. If we batch, we can't send audio until generation finishes — defeating the purpose of the forward hook approach.

**Tradeoff accepted:** Higher RTF (1.47x vs. ~0.3x) in exchange for true per-token audio streaming.

---

## 4. Silent reference voice vs. real voice cloning

**Decision:** Use `x_vector_only_mode=True` with 1 second of silence as the reference audio.

**Alternative:** Use a real speaker recording for voice cloning.

**Why:** We don't have a target voice. `x_vector_only_mode` skips in-context learning and uses only the speaker embedding (x-vector), which produces a consistent neutral voice without needing a real recording.

**Tradeoff accepted:** The output voice is generic/neutral. Adding a real reference voice would give a specific character to the assistant but requires a clean audio recording.

---

## 5. LDG_VOCAB_SIZE patch: 151936 → 3072

**Decision:** Patch `kernel/csrc/kernel.cu` to change `LDG_VOCAB_SIZE` from 151936 (Qwen3 text vocab) to 3072 (TTS codec vocab).

**Why:** The megakernel was built for Qwen3-0.6B text generation with a 151,936-token vocab. Qwen3-TTS's talker has a 3,072-token audio codec vocab. The lm_head projection kernel allocates blocks based on vocab size — using 151,936 wastes GPU memory and bandwidth, and the weight shape doesn't match.

**Tradeoff accepted:** The patched kernel is only compatible with TTS (3072-vocab) decode, not standard Qwen3 text generation.

---

## 6. Megakernel loaded but not integrated into the live generation path

**Decision:** `_mk_decoder` is initialized in `StreamingTTSEngine` but not called during `generate_streaming()`.

**Why:** The official model's `generate()` method controls the decode loop. The megakernel is a standalone decoder with its own KV cache — integrating it would require either (a) intercepting the talker's forward pass and replacing it mid-loop, or (b) rewriting the decode loop from scratch (which broke audio quality previously).

**What the megakernel gives us:** The standalone benchmark shows 1033 tok/s. If fully integrated, the main transformer step would drop from ~23ms/frame to ~1ms/frame. Code_predictor (54ms/frame) would still dominate.

**Tradeoff accepted:** Megakernel available and verified working, but not wired into the hot path. Full integration is a future step requiring careful KV-cache synchronisation.

---

## 7. Thread + asyncio Queue bridge in Pipecat service

**Decision:** Run `generate_streaming()` in a `ThreadPoolExecutor` thread, bridge chunks to the asyncio event loop via `asyncio.Queue` + `call_soon_threadsafe`.

**Why:** `generate_streaming()` is a blocking generator (GPU ops, can't be awaited). Pipecat's pipeline is async. Running blocking code directly on the event loop would freeze the entire pipeline.

**Alternative:** `asyncio.to_thread` — simpler but doesn't give control over the executor (we need `max_workers=1` to serialise GPU access).

**Tradeoff accepted:** Slightly more boilerplate, but GPU work is correctly serialised and the event loop is never blocked.

---

## 8. Text wrapping format

**Decision:** Wrap input text as:
```
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

**Why:** The TTS model expects a chat-formatted prompt. The trailing `<|im_start|>assistant\n` is the generation prefix that cues the model to start producing audio tokens. Without the exact format, the model produces silence or garbage.

---

## 9. code_predictor is the real bottleneck

**Observation:** Profiling shows:
- code_predictor (15 sequential steps): ~54ms/frame
- Main transformer (1 step): ~23ms/frame
- Vocoder: ~5ms/frame

**Implication for targets:** The original RTF < 0.15 target assumed the main transformer was the bottleneck. With code_predictor dominating, even a perfect megakernel integration brings RTF to ~0.75x, not 0.15x. Reaching 0.15x would require flash-attn (cuts code_predictor to ~18ms) and potentially parallelising the 15 acoustic code predictions.

---

## 11. Parallel code_predictor prediction — attempted, reverted

**Idea:** Instead of running code_predictor's transformer 15 times sequentially, run it once and apply all 15 `lm_head[i]` projections to the same hidden state in parallel. Collapses 15 transformer passes → 1.

**Expected speedup:** code_predictor: 54ms → 4ms per frame. RTF: 1.47x → 0.53x.

**What we implemented:** Monkey-patched `code_predictor.generate` with `_parallel_code_generate()` — single transformer forward pass on `(past_hidden, semantic_embed)`, then 15 parallel `lm_head` matmuls. Returns a `SimpleNamespace(sequences=...)` matching the expected interface.

**What actually happened:**
- TTFC improved: 390ms → 94ms (4x faster first chunk)
- RTF improved: 1.47x → 0.53x (generation time)
- But chunk count: 20 → 4095 (hit max sequence length — model never generated EOS)
- Audio: intelligible for the first ~1.5 seconds, then degraded to noise indefinitely

**Root cause:** The talker uses `predictor_result.sequences` to build the combined embedding for the *next* decode step: `inputs_embeds = sum(semantic_embed + 15_acoustic_embeds)`. With parallel codes instead of autoregressive codes, this combined embedding is wrong at every step. The main transformer's context is corrupted frame-by-frame — it never finds the EOS state.

**Why the lookup table idea also doesn't work:** The same embedding feedback issue. Even if codes were precomputed per semantic token, the next-step embedding would still be wrong because `past_hidden` (the talker's rolling context) varies per frame.

**Conclusion:** Any approach that changes the acoustic codes without also fixing the embedding feedback will break EOS. Parallel prediction is fundamentally incompatible with this model's autoregressive feedback loop. The only clean speedup for code_predictor is flash-attn (cuts each of the 15 passes from 3.6ms to ~1.2ms, no quality loss).

---

## 10. Headless GPU — no physical audio

**Decision:** For Step 17, use simulated STT + LLM responses to verify the pipeline plumbing without physical mic/speaker hardware.

**Why:** The Vast.ai RTX 5090 instance has no audio hardware. PulseAudio virtual devices work for local audio I/O but there's no way to "speak into" the server remotely.

**For the real demo (Step 19):** Full end-to-end run requires Deepgram + OpenAI API keys and either SSH audio forwarding or Daily.co WebRTC transport.



Option 3: Write a custom CUDA kernel for CodePredictor
Seq_len=2 is fixed, batch=1, 5 layers. Extremely predictable shape — could fuse the whole thing. Brings 54ms → 5-8ms. This is the right long-term fix but 2-3 days of work.