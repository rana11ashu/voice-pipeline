"""
Benchmark: full TTS pipeline (TTFC and RTF).

Metrics:
  TTFC — Time To First Chunk: from generate_streaming() call to first audio
         chunk yielded. Target: < 60ms.
  RTF  — Real-Time Factor: total_synthesis_time / total_audio_duration.
         Target: < 0.15  (synthesises audio 6.7× faster than real-time).

Runs 10 trials on a fixed sentence and reports median + p95.

Run on GPU machine (Step 18):
    python benchmarks/bench_tts.py

Target: TTFC < 60ms, RTF < 0.15.
"""

import sys
import time
import statistics

import torch

MODEL_DIR   = "/root/.cache/qwen3-tts"
TEST_TEXT   = "Hello, how are you today?"
NUM_TRIALS  = 10
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920   # 1 token = 80ms of audio at 12.5 Hz


def measure_trial(engine) -> tuple[float, float]:
    """
    Returns (ttfc_ms, rtf) for one synthesis run.

    TTFC: time from call start to first chunk.
    RTF:  total synthesis time / total audio duration.
    """
    gen = engine.generate_streaming(TEST_TEXT)

    t_start = time.perf_counter()
    ttfc_ms = None
    n_chunks = 0

    for _pcm, _sr in gen:
        if ttfc_ms is None:
            ttfc_ms = (time.perf_counter() - t_start) * 1000
        n_chunks += 1

    t_end = time.perf_counter()

    total_synthesis_s = t_end - t_start
    total_audio_s     = (n_chunks * SAMPLES_PER_FRAME) / SAMPLE_RATE
    rtf = total_synthesis_s / total_audio_s if total_audio_s > 0 else float("inf")

    return ttfc_ms or 0.0, rtf


def percentile(data: list[float], p: int) -> float:
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on the GPU machine.")
        sys.exit(1)

    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print(f"Text:       '{TEST_TEXT}'")
    print(f"Trials:     {NUM_TRIALS}")
    print()

    from tts.streaming_engine import StreamingTTSEngine
    print("Loading engine (megakernel mode)...")
    engine = StreamingTTSEngine(
        model_dir=MODEL_DIR,
        device="cuda",
        use_megakernel=True,
    )

    # Warm-up trial (not counted)
    print("Warming up...")
    list(engine.generate_streaming(TEST_TEXT))
    torch.cuda.synchronize()

    # Benchmark trials
    ttfc_results = []
    rtf_results  = []

    for i in range(NUM_TRIALS):
        ttfc_ms, rtf = measure_trial(engine)
        ttfc_results.append(ttfc_ms)
        rtf_results.append(rtf)
        print(f"  Trial {i+1:2d}: TTFC={ttfc_ms:6.1f}ms  RTF={rtf:.3f}")

    # Stats
    ttfc_median = statistics.median(ttfc_results)
    ttfc_p95    = percentile(ttfc_results, 95)
    rtf_median  = statistics.median(rtf_results)
    rtf_p95     = percentile(rtf_results, 95)

    print()
    print("=" * 45)
    print(f"{'Metric':<12} {'Median':>10} {'p95':>10}  {'Target':>10}")
    print("-" * 45)
    print(f"{'TTFC (ms)':<12} {ttfc_median:>10.1f} {ttfc_p95:>10.1f}  {'<60ms':>10}")
    print(f"{'RTF':<12} {rtf_median:>10.3f} {rtf_p95:>10.3f}  {'<0.15':>10}")
    print("=" * 45)

    ttfc_pass = ttfc_median < 60
    rtf_pass  = rtf_median  < 0.15
    print(f"TTFC: {'PASS' if ttfc_pass else 'FAIL'}")
    print(f"RTF:  {'PASS' if rtf_pass  else 'FAIL'}")


if __name__ == "__main__":
    main()
