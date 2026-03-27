"""
Voice agent demo — runs on your local machine.

Pipeline:
    [Mic] → OpenAI Whisper STT → OpenAI GPT-4o-mini → [GPU TTS server] → [Speakers]

Requirements:
    pip install openai pyaudio soundfile numpy python-dotenv requests

Setup:
    1. Start TTS server on GPU:
           ssh -p 42575 root@90.224.159.6
           cd ~/voice-pipeline && PYTHONPATH=. python3 tts/tts_server.py

    2. Open SSH tunnel in a separate terminal:
           ssh -p 42575 -N -L 5000:localhost:5000 root@90.224.159.6

    3. Run this script:
           cd voice-pipeline && python3 demo/demo.py

Controls:
    Press ENTER to start recording, press ENTER again to stop.
    Type 'quit' and press ENTER to exit.
"""

import io
import os
import sys
import wave

import numpy as np
import pyaudio
import requests
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TTS_SERVER     = os.environ.get("TTS_SERVER", "http://localhost:5000")

SAMPLE_RATE    = 16000   # Whisper works best at 16kHz
CHANNELS       = 1
CHUNK          = 1024
FORMAT         = pyaudio.paInt16

SYSTEM_PROMPT = (
    "You are a concise, knowledgeable voice assistant. "
    "Keep every response to one or two short sentences — you are speaking aloud, not writing. "
    "No bullet points, no markdown, no special characters. Be direct and confident."
)

client = OpenAI(api_key=OPENAI_API_KEY)
history = [{"role": "system", "content": SYSTEM_PROMPT}]


def record_until_enter() -> bytes:
    """Record mic audio until user presses Enter. Returns raw PCM bytes."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    frames = []
    print("  🎙  Recording... (press ENTER to stop)")
    import threading
    stop_event = threading.Event()

    def _wait():
        input()
        stop_event.set()

    t = threading.Thread(target=_wait, daemon=True)
    t.start()

    while not stop_event.is_set():
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container (for Whisper)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe(pcm: bytes) -> str:
    """Send PCM audio to OpenAI Whisper and return transcript."""
    wav_bytes = pcm_to_wav(pcm)
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
    )
    return result.text.strip()


def llm_response(user_text: str) -> str:
    """Send user text to GPT-4o-mini and return the response."""
    history.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        max_tokens=100,
    )
    text = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": text})
    return text


def speak(text: str):
    """Fetch complete audio from GPU TTS server then play it smoothly."""
    resp = requests.post(f"{TTS_SERVER}/tts", json={"text": text}, timeout=120)
    resp.raise_for_status()

    audio, sr = sf.read(io.BytesIO(resp.content), dtype="int16")

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True)
    try:
        stream.write(audio.tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    print("=" * 50)
    print("  Voice Agent Demo")
    print("  (Press ENTER to speak, ENTER again to stop)")
    print("  Type 'quit' + ENTER to exit")
    print("=" * 50)

    # Verify TTS server is up
    try:
        requests.get(f"{TTS_SERVER}/health", timeout=5).raise_for_status()
        print(f"  TTS server: OK ({TTS_SERVER})\n")
    except Exception as e:
        print(f"\n  ERROR: TTS server not reachable at {TTS_SERVER}")
        print("  Make sure the SSH tunnel is running:")
        print("    ssh -p 42575 -N -L 5000:localhost:5000 root@90.224.159.6\n")
        sys.exit(1)

    while True:
        cmd = input("Press ENTER to speak (or 'quit'): ").strip().lower()
        if cmd == "quit":
            break

        pcm = record_until_enter()

        print("  Transcribing...", end=" ", flush=True)
        transcript = transcribe(pcm)
        if not transcript:
            print("(no speech detected)")
            continue
        print(f"\n  You:   {transcript}")

        print("  Agent thinking...", end=" ", flush=True)
        response = llm_response(transcript)
        print(f"\n  Agent: {response}")

        print("  Speaking...")
        speak(response)
        print()


if __name__ == "__main__":
    main()
