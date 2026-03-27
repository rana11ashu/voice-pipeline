"""
Lightweight TTS HTTP server — runs on the GPU.

POST /tts  {"text": "..."}  → returns WAV bytes (24kHz mono int16)

Start with:
    cd ~/voice-pipeline && PYTHONPATH=. python3 tts/tts_server.py
"""

import io
import os

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_file

from tts.streaming_engine import StreamingTTSEngine

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/.cache/qwen3-tts"))
DEVICE    = os.environ.get("DEVICE", "cuda")

print(f"[server] Loading TTS engine (device={DEVICE})...")
_engine = StreamingTTSEngine(model_dir=MODEL_DIR, device=DEVICE)
print("[server] Ready.")


@app.route("/tts", methods=["POST"])
def tts():
    """Generate full audio then return as WAV — clean playback, no stuttering."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    print(f"[server] Synthesising: {text!r}")
    chunks = []
    for pcm_bytes, _sr in _engine.generate_streaming(text):
        chunks.append(pcm_bytes)

    audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767.0
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name="tts.wav")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)
