"""
Full voice pipeline: mic → Deepgram STT → OpenAI LLM → Qwen3-TTS → speakers.

Run on the GPU machine (Step 17):
    python pipeline/bot.py

Requires environment variables:
    DEEPGRAM_API_KEY
    OPENAI_API_KEY

Optional:
    MODEL_DIR   path to Qwen3-TTS weights  (default: ~/.cache/qwen3-tts)
    DEVICE      cuda or cpu                (default: cuda)
"""

import asyncio
import os

import pyaudio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)

from pipecat_service.qwen3_tts_service import Qwen3MegakernelTTSService

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep your responses short — one or two sentences. "
    "You are being spoken to and will respond with speech, so avoid lists, "
    "markdown, or any formatting."
)


async def main():
    model_dir = os.environ.get("MODEL_DIR", os.path.expanduser("~/.cache/qwen3-tts"))
    device    = os.environ.get("DEVICE", "cuda")

    py_audio = pyaudio.PyAudio()

    # --- Transports (mic + speakers) ---
    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_stop_secs=0.5,
    )
    input_transport  = LocalAudioInputTransport(py_audio, params)
    output_transport = LocalAudioOutputTransport(py_audio, params)

    # --- Services ---
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    )

    tts = Qwen3MegakernelTTSService(
        model_dir=model_dir,
        device=device,
        use_megakernel=(device == "cuda"),
    )

    # --- LLM context (conversation memory) ---
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline ---
    # mic → STT → [user aggregator] → LLM → TTS → speakers → [assistant aggregator]
    pipeline = Pipeline([
        input_transport,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        output_transport,
        context_aggregator.assistant(),
    ])

    task   = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
