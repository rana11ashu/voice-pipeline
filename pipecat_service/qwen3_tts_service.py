"""
Pipecat TTS service wrapping the Qwen3-TTS megakernel streaming engine.

Plugs into a Pipecat pipeline as a drop-in TTS service:
    mic → Deepgram STT → OpenAI LLM → Qwen3MegakernelTTSService → speakers

The streaming engine runs blocking GPU work in a background thread so it
doesn't block Pipecat's asyncio event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from tts.streaming_engine import StreamingTTSEngine


class Qwen3MegakernelTTSService(TTSService):
    """
    Pipecat TTSService that uses Qwen3-TTS with the qwen_megakernel decoder.

    Args:
        model_dir:       path to Qwen3-TTS-12Hz-0.6B-Base download
        device:          'cuda' in production, 'cpu' for local testing
        use_megakernel:  True on GPU after Step 15 wires in the kernel
        **kwargs:        forwarded to TTSService base class
    """

    def __init__(
        self,
        model_dir: str = "~/.cache/qwen3-tts",
        device: str = "cuda",
        use_megakernel: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._engine = StreamingTTSEngine(
            model_dir=model_dir,
            device=device,
            use_megakernel=use_megakernel,
        )
        # Single-thread executor: GPU work must be serialised
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """
        Called by Pipecat with each text chunk from the LLM.
        Runs the streaming engine in a background thread and yields audio
        frames back to the pipeline as they arrive.
        """
        loop = asyncio.get_event_loop()

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Queue bridges the blocking generator (thread) and the async consumer (event loop).
        # sentinel=None signals the producer is done.
        queue: asyncio.Queue = asyncio.Queue()
        first_chunk = True

        def _produce():
            """Runs in executor thread: feeds chunks into the queue as they arrive."""
            try:
                for pcm_bytes, sample_rate in self._engine.generate_streaming(text):
                    loop.call_soon_threadsafe(queue.put_nowait, (pcm_bytes, sample_rate))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        # Kick off production in background — does NOT block the event loop
        loop.run_in_executor(self._executor, _produce)

        # Consume chunks as they arrive — yields each frame immediately
        while True:
            item = await queue.get()
            if item is None:
                break

            pcm_bytes, sample_rate = item

            if first_chunk:
                await self.stop_ttfb_metrics()
                first_chunk = False

            yield TTSAudioRawFrame(
                audio=pcm_bytes,
                sample_rate=sample_rate,
                num_channels=1,
                context_id=context_id,
            )

        yield TTSStoppedFrame()
