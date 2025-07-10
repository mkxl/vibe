import asyncio
import contextlib
import dataclasses
from pathlib import Path
from typing import Annotated, AsyncIterator, ClassVar, Optional, Self

import sounddevice
from typer import Option

from vibe.config import Config
from vibe.conversation import Conversation, Role
from vibe.eleven_labs import ElevenLabs
from vibe.language_model import LanguageModel, LanguageModelResponse
from vibe.sambanova import Sambanova
from vibe.triton import EotDetection, EotDetectionResult, Transcription
from vibe.utils.audio import Audio
from vibe.utils.http import Http
from vibe.utils.logger import Logger
from vibe.utils.microphone import Dtype, Microphone
from vibe.utils.queue import Queue, TaskQueue
from vibe.utils.utils import Utils
from vibe.vad import Vad

logger: Logger = Logger.new(__name__)


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass(kw_only=True)
class Chat:
    BLOCK_ON_PLAY_AUDIO: ClassVar[bool] = False

    conversation: Conversation
    eleven_labs: ElevenLabs
    language_model: LanguageModel
    language_model_response_queue: Queue[LanguageModelResponse]
    microphone: Microphone
    play_audio_task_queue: TaskQueue[bytes]
    transcription_task_queue: TaskQueue[Optional[str]]
    vad: Vad

    @classmethod
    @contextlib.asynccontextmanager
    async def context(cls, *, config_filepath: Path, device: int, dtype: Dtype) -> AsyncIterator[Self]:
        config = Config.model_validate_json(config_filepath.read_text())
        conversation = Conversation.new(system_prompt=config.system_prompt)
        eleven_labs_cm = ElevenLabs.context(voice_id=config.eleven_labs_voice_id, api_key=config.eleven_labs_api_key)

        with Microphone.context(device=device, dtype=dtype) as microphone:
            async with Http.context() as http, eleven_labs_cm as eleven_labs:
                language_model = Sambanova(
                    http=http,
                    model=config.sambanova_model,
                    api_key=config.sambanova_api_key,
                    tools=config.sambanova_tools,
                )
                chat = cls(
                    conversation=conversation,
                    eleven_labs=eleven_labs,
                    language_model=language_model,
                    language_model_response_queue=Queue.new(),
                    microphone=microphone,
                    play_audio_task_queue=TaskQueue.new(),
                    transcription_task_queue=TaskQueue.new(),
                    vad=Vad.new(),
                )

                yield chat

    @classmethod
    async def _transcribe_and_eot(cls, *, audio: Audio, user_last: bool) -> tuple[Optional[str], EotDetectionResult]:
        transcription_coro = Transcription(audio=audio).result()
        eot_detection_result_coro = EotDetection(audio=audio, user_last=user_last).result()
        transcription, eot_detection_result = await asyncio.gather(transcription_coro, eot_detection_result_coro)

        return (transcription, eot_detection_result)

    async def _queue_transcriptions(self) -> None:
        async for microphone_input in self.microphone.input_queue:
            for vad_result in self.vad.add(audio=microphone_input.audio):
                # TODO: use self._transcribe_and_eot()
                transcription_coro = Transcription(audio=vad_result.audio).result()

                # NOTE-c8ddfc: call [Utils.yield_now()] bc this is a synchronous for loop
                self.transcription_task_queue.append(transcription_coro)

                await Utils.yield_now()

    async def _queue_language_model_responses(self) -> None:
        async for transcription in self.transcription_task_queue:
            if Utils.is_none_or_empty(text=transcription):
                continue

            self.conversation.append(role=Role.USER, text=transcription)

            language_model_response = self.language_model.respond(conversation=self.conversation)

            self.language_model_response_queue.append(language_model_response)

    async def _process_language_model_response_text_queue(
        self, *, language_model_response: LanguageModelResponse
    ) -> None:
        async for text in language_model_response.text_queue:
            self.conversation.append(role=Role.ASSISTANT, text=text)

            await self.eleven_labs.asend(text)

    async def _process_language_model_response_tool_call_queue(
        self, *, language_model_response: LanguageModelResponse
    ) -> None:
        async for _tool_call in language_model_response.tool_call_queue:
            pass

    async def _process_language_model_responses(self) -> None:
        async for language_model_response in self.language_model_response_queue:
            text_coro = self._process_language_model_response_text_queue(
                language_model_response=language_model_response
            )
            tool_call_coro = self._process_language_model_response_tool_call_queue(
                language_model_response=language_model_response
            )

            await asyncio.gather(text_coro, tool_call_coro)

            logger.info(conversation_chat_messages=self.conversation.chat_messages())

    async def _play_audio(self) -> None:
        async for audio in self.eleven_labs.iter_audio():
            sounddevice.play(data=audio.data, samplerate=audio.sample_rate, blocking=self.BLOCK_ON_PLAY_AUDIO)

    async def _run(self) -> None:
        await Utils.wait(
            self._queue_transcriptions(),
            self._queue_language_model_responses(),
            self._process_language_model_responses(),
            self._play_audio(),
        )

    @classmethod
    @logger.instrument()
    async def chat(
        cls,
        *,
        config_filepath: Annotated[Path, Option("--config")],
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        dtype: Annotated[Dtype, Option()] = Microphone.DEFAULT_DTYPE,
    ) -> None:
        async with cls.context(config_filepath=config_filepath, device=device, dtype=dtype) as chat:
            await chat._run()  # pylint: disable=protected-access
