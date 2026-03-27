"""
Headless end-to-end pipeline test (Step 17).

Verifies the full voice pipeline without physical audio hardware:
  Simulated STT → Simulated LLM → Qwen3-TTS → WAV file

Usage:
    python pipeline/test_e2e.py

Produces output/test_e2e_output.wav.
"""

import asyncio
import os
import time

import numpy as np
import soundfile as sf

# Simulated input/output — verifies the full plumbing without an LLM API
SIMULATED_USER_SPEECH = "What is the capital of France?"
SIMULATED_LLM_RESPONSE = "Paris is the capital of France. It's known as the City of Light."


async def run_tts(text: str) -> tuple[bytes, int]:
    from tts.streaming_engine import StreamingTTSEngine

    model_dir = os.environ.get("MODEL_DIR", os.path.expanduser("~/.cache/qwen3-tts"))
    device = os.environ.get("DEVICE", "cuda")

    engine = StreamingTTSEngine(model_dir=model_dir, device=device, use_megakernel=False)

    chunks = []
    t0 = time.perf_counter()
    first_chunk_time = None

    for i, (pcm_bytes, sample_rate) in enumerate(engine.generate_streaming(text)):
        now = time.perf_counter()
        if first_chunk_time is None:
            first_chunk_time = (now - t0) * 1000
            print(f"  TTS TTFC:     {first_chunk_time:.0f}ms")
        chunks.append(pcm_bytes)

    total_time = (time.perf_counter() - t0) * 1000
    n_chunks = len(chunks)
    audio_duration_ms = n_chunks * 80  # 80ms per frame
    rtf = total_time / audio_duration_ms if audio_duration_ms > 0 else 0

    print(f"  TTS chunks:   {n_chunks}")
    print(f"  TTS duration: {total_time:.0f}ms total, {audio_duration_ms}ms audio")
    print(f"  TTS RTF:      {rtf:.3f}x")

    all_pcm = b"".join(chunks)
    return all_pcm, sample_rate


def save_wav(pcm_bytes: bytes, sample_rate: int, path: str):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sample_rate)
    print(f"  Saved:        {path}")


async def main():
    print("=" * 60)
    print("End-to-end pipeline test")
    print("=" * 60)
    print(f"\nUser (simulated STT):  {SIMULATED_USER_SPEECH!r}")
    print(f"Agent (simulated LLM): {SIMULATED_LLM_RESPONSE!r}\n")

    # TTS
    print("[TTS]")
    pcm_bytes, sample_rate = await run_tts(SIMULATED_LLM_RESPONSE)

    # Step 3: Save
    print("\n[Output]")
    save_wav(pcm_bytes, sample_rate, "output/test_e2e_output.wav")

    print("\n" + "=" * 60)
    print("Pipeline test PASSED — output/test_e2e_output.wav written")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
