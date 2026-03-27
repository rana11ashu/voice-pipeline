"""Compare autoregressive vs parallel code_predictor — speed and audio output."""

import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "kernel")
from tts.streaming_engine import StreamingTTSEngine

TEXT = "Hello, how are you today?"


def run(label, use_parallel):
    print(f"\n=== {label} ===")
    engine = StreamingTTSEngine(device="cuda", use_parallel_codes=use_parallel)
    chunks = []
    t0 = time.perf_counter()
    for pcm, sr in engine.generate_streaming(TEXT):
        if not chunks:
            print(f"  TTFC : {(time.perf_counter() - t0) * 1000:.0f}ms")
        chunks.append(pcm)
    total = time.perf_counter() - t0
    audio_dur = len(chunks) * 1920 / 24000
    print(f"  RTF  : {total / audio_dur:.3f}x")
    print(f"  Chunks: {len(chunks)}")
    return b"".join(chunks)


if __name__ == "__main__":
    pcm1 = run("BASELINE  (autoregressive)", use_parallel=False)
    pcm2 = run("PARALLEL  (single forward pass)", use_parallel=True)

    sf.write("/tmp/baseline.wav", np.frombuffer(pcm1, dtype=np.int16), 24000)
    sf.write("/tmp/parallel.wav", np.frombuffer(pcm2, dtype=np.int16), 24000)
    print("\nSaved /tmp/baseline.wav and /tmp/parallel.wav")
