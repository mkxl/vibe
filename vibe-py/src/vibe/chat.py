import asyncio
import contextlib
import dataclasses
from typing import Annotated, AsyncIterator, Optional, Self

from typer import Option

from vibe.triton import EotDetection, EotDetectionResult, Transcription
from vibe.utils.audio import Audio
from vibe.utils.logger import Logger
from vibe.utils.microphone import Dtype, Microphone
from vibe.utils.task_queue import TaskQueue
from vibe.utils.utils import Utils
from vibe.vad import Vad

logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class Chat:
    microphone: Microphone
    task_queue: TaskQueue[Optional[str]]
    vad: Vad

    @classmethod
    @contextlib.asynccontextmanager
    async def context(cls, *, device: int, dtype: Dtype) -> AsyncIterator[Self]:
        async with Microphone.context(device=device, dtype=dtype) as microphone:
            task_queue = await TaskQueue.new()
            chat = cls(microphone=microphone, task_queue=task_queue, vad=Vad.new())

            yield chat

    @classmethod
    async def _transcribe_and_eot(cls, *, audio: Audio, user_last: bool) -> tuple[Optional[str], EotDetectionResult]:
        transcription_coro = Transcription(audio=audio).result()
        eot_detection_result_coro = EotDetection(audio=audio, user_last=user_last).result()
        transcription, eot_detection_result = await asyncio.gather(transcription_coro, eot_detection_result_coro)

        return (transcription, eot_detection_result)

    async def _listen(self) -> None:
        async for microphone_input in self.microphone:
            for vad_result in self.vad.add(audio=microphone_input.audio):
                # coro = self._transcribe_and_eot(audio=vad_result.audio, user_last=False)
                coro = Transcription(audio=vad_result.audio).result()

                self.task_queue.append(coro)

                await Utils.yield_now()

    async def _process(self) -> None:
        async for pair in self.task_queue:
            logger.info(pair=pair)

    async def _run(self) -> None:
        await Utils.wait(self._listen(), self._process())

    @classmethod
    @logger.instrument()
    async def chat(
        cls,
        *,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        dtype: Annotated[Dtype, Option()] = Microphone.DEFAULT_DTYPE,
    ) -> None:
        async with cls.context(device=device, dtype=dtype) as chat:
            await chat._run()  # pylint: disable=protected-access
