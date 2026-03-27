"""
Benchmark: isolated megakernel decode speed.

Runs 500 single-token decode steps using qwen_megakernel.Decoder and reports
tokens/sec. Uses CUDA events for accurate GPU-side timing (excludes Python
overhead and CPU↔GPU transfers).

Run on GPU machine (Step 18):
    python benchmarks/bench_kernel.py

Target: ~1000 tokens/sec on RTX 5090.
"""

import sys
import torch

MODEL_DIR   = "/root/.cache/qwen3-tts"   # adjust if different on Vast.ai
WARMUP_STEPS = 50
BENCH_STEPS  = 500
SEED_TOKEN   = 100   # arbitrary valid codec token (0–2047)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on the GPU machine.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Load weights
    print("Loading weights...")
    from tts.weight_bridge import TalkerWeightBridge
    weights = TalkerWeightBridge().load(MODEL_DIR, device="cuda")

    # Build megakernel decoder
    from qwen_megakernel import Decoder
    decoder = Decoder(weights=weights)
    print("Megakernel decoder ready.")

    # Warm-up: let the GPU reach steady state
    print(f"Warming up ({WARMUP_STEPS} steps)...")
    token = SEED_TOKEN
    for _ in range(WARMUP_STEPS):
        token = decoder.step(token)
    torch.cuda.synchronize()

    # Benchmark: CUDA events give GPU-side wall time, no Python overhead
    print(f"Benchmarking ({BENCH_STEPS} steps)...")
    decoder.reset()
    token = SEED_TOKEN

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(BENCH_STEPS):
        token = decoder.step(token)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)   # milliseconds

    tokens_per_sec = BENCH_STEPS / (elapsed_ms / 1000)
    ms_per_token   = elapsed_ms / BENCH_STEPS

    print()
    print("=" * 40)
    print(f"Steps:          {BENCH_STEPS}")
    print(f"Total time:     {elapsed_ms:.1f} ms")
    print(f"Per token:      {ms_per_token:.3f} ms")
    print(f"Tokens/sec:     {tokens_per_sec:.0f}")
    print("=" * 40)
    print(f"Target:         ~1000 tok/s")
    status = "PASS" if tokens_per_sec >= 900 else "BELOW TARGET"
    print(f"Status:         {status}")


if __name__ == "__main__":
    main()
