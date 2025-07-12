import asyncio
import contextlib
import dataclasses
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, ClassVar, Optional, Self, TextIO

from typer import Option

from vibe.config import Config, Secret
from vibe.conversation import Conversation
from vibe.eleven_labs import ElevenLabs
from vibe.language_model import LanguageModel, LanguageModelResponse
from vibe.models import ChatMessage, ConstantImplementation, Implementation, ProcessImplementation, Role, Tool, ToolCall
from vibe.sambanova import Sambanova
from vibe.triton import EotDetection, EotDetectionResult, Transcription
from vibe.tts import Tts
from vibe.utils.audio import Audio
from vibe.utils.http import Http
from vibe.utils.logger import Logger
from vibe.utils.microphone import Dtype, Microphone
from vibe.utils.process import Process
from vibe.utils.queue import Queue, TaskQueue
from vibe.utils.utils import Utils
from vibe.vad import Vad

logger: Logger = Logger.new(__name__)


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass(kw_only=True)
class Chat:
    # NOTE:
    # - "success" from [https://platform.openai.com/docs/guides/function-calling#formatting-results]
    CONFIG_YAML_STR: ClassVar[str] = Utils.read_document("config.yaml")
    CONFIG: ClassVar[Config] = Utils.model_validate_yaml(CONFIG_YAML_STR, base_model_type=Config)
    TOOL_CALL_RESULT_CONTENT_FOR_SUCCESS: ClassVar[str] = "success"
    WRITE_MODE: ClassVar[str] = "w"

    conversation: Conversation
    implementations: dict[str, Implementation]
    language_model: LanguageModel
    language_model_response_queue: Queue[LanguageModelResponse]
    microphone: Microphone
    misc_task_queue: TaskQueue[Any]
    request_chat_message_task_queue: TaskQueue[Optional[ChatMessage]]
    secret: Secret
    tool_call_queue: Queue[ToolCall]
    tts: Tts
    vad: Vad

    @classmethod
    @contextlib.asynccontextmanager
    async def _acontext(cls, *, secret_filepath: Path, device: int, dtype: Dtype) -> AsyncIterator[Self]:
        secret = Utils.model_validate_yaml(secret_filepath.read_text(), base_model_type=Secret)
        conversation = Conversation.new(system_prompt=cls.CONFIG.system_prompt)
        tts_cm = ElevenLabs.acontext(voice_id=cls.CONFIG.eleven_labs_voice_id, api_key=secret.eleven_labs_api_key)
        implementations = cls._implementations(tools=cls.CONFIG.tools)

        with Microphone.context(device=device, dtype=dtype) as microphone:
            async with Http.acontext() as http, tts_cm as tts:
                language_model = Sambanova(
                    http=http,
                    model=cls.CONFIG.sambanova_model,
                    api_key=secret.sambanova_api_key,
                    tools=cls.CONFIG.tools,
                )
                chat = cls(
                    conversation=conversation,
                    implementations=implementations,
                    language_model=language_model,
                    language_model_response_queue=Queue.new(),
                    microphone=microphone,
                    misc_task_queue=TaskQueue.new(),
                    request_chat_message_task_queue=TaskQueue.new(),
                    secret=secret,
                    tool_call_queue=Queue.new(),
                    tts=tts,
                    vad=Vad.new(),
                )

                yield chat

    @staticmethod
    def _implementations(*, tools: Optional[list[Tool]]) -> dict[str, Implementation]:
        return {} if tools is None else {tool.function.name: tool.function.implementation for tool in tools}

    @classmethod
    def _open_file_in_write_mode(cls, filepath: Path) -> TextIO:
        return filepath.open(cls.WRITE_MODE)

    @classmethod
    async def _transcribe_and_eot(cls, *, audio: Audio, user_last: bool) -> tuple[Optional[str], EotDetectionResult]:
        transcription_coro = Transcription(audio=audio).result()
        eot_detection_result_coro = EotDetection(audio=audio, user_last=user_last).result()
        transcription, eot_detection_result = await asyncio.gather(transcription_coro, eot_detection_result_coro)

        return (transcription, eot_detection_result)

    async def _transcribe(self, *, audio: Audio) -> Optional[ChatMessage]:
        transcription = await Transcription(audio=audio).result()
        chat_message = (
            ChatMessage(role=Role.USER, content=transcription)
            if Utils.is_not_none_and_is_nonempty(transcription)
            else None
        )

        return chat_message

    async def _queue_transcribed_user_messages(self) -> None:
        async for microphone_input in self.microphone.input_queue:
            for vad_result in self.vad.add(audio=microphone_input.audio):
                # TODO: use [self._transcribe_and_eot()]
                self.request_chat_message_task_queue.create_task(self._transcribe, audio=vad_result.audio)

                # NOTE-c8ddfc: call [Utils.yield_now()] bc this is a synchronous for loop
                await Utils.yield_now()

    async def _prompt_language_model_and_queue_response(self) -> None:
        async for chat_message in self.request_chat_message_task_queue:
            if chat_message is None:
                continue

            logger.info(request_chat_message=chat_message)

            self.conversation.append_chat_message(chat_message)

            language_model_response = self.language_model.respond(conversation=self.conversation)

            self.language_model_response_queue.append(language_model_response)

    async def _process_language_model_response(self, *, language_model_response: LanguageModelResponse) -> None:
        async for chat_message in language_model_response.chat_message_queue:
            # NOTE: sometimes we receive chat messages that have a content of "" and a tool_calls value of None
            should_append_chat_message = False

            if Utils.is_not_none_and_is_nonempty(chat_message.content):
                await self.tts.asend(chat_message.content)

                should_append_chat_message = True

            if chat_message.tool_calls is not None:
                self.tool_call_queue.extend(chat_message.tool_calls)

                should_append_chat_message = True

            logger.info(received_chat_message=chat_message, should_append_chat_message=should_append_chat_message)

            if should_append_chat_message:
                self.conversation.append_chat_message(chat_message)

    async def _process_language_model_responses(self) -> None:
        async for language_model_response in self.language_model_response_queue:
            await self._process_language_model_response(language_model_response=language_model_response)

            # NOTE: log conversation after processing response to ensure the response was processed correctly
            logger.debug(conversation_chat_messages=self.conversation.chat_messages())

    async def _tool_call_result_content_for_process(
        self, *, process: Process, input_byte_str: Optional[bytes], wait: bool
    ) -> str:
        output_task = self.misc_task_queue.create_task(process.output, input_byte_str=input_byte_str)

        if wait:
            tool_call_result_content = await output_task

            if tool_call_result_content == "":
                tool_call_result_content = self.TOOL_CALL_RESULT_CONTENT_FOR_SUCCESS
        else:
            tool_call_result_content = self.TOOL_CALL_RESULT_CONTENT_FOR_SUCCESS

        return tool_call_result_content

    async def _tool_call_result_content(self, *, tool_call: ToolCall) -> str:
        match self.implementations.get(tool_call.function.name):
            case ConstantImplementation() as constant_implementation:
                tool_call_result_content = constant_implementation.value
            case ProcessImplementation() as process_implementation:
                process = await Process.new(process_implementation.command, *process_implementation.args)
                input_byte_str = process_implementation.input_byte_str(tool_call=tool_call)
                tool_call_result_content = await self._tool_call_result_content_for_process(
                    process=process, input_byte_str=input_byte_str, wait=process_implementation.wait
                )

                logger.debug(
                    command=process_implementation.command,
                    args=process_implementation.args,
                    input_byte_str=input_byte_str,
                    tool_call_result_content=tool_call_result_content,
                )
            case unknown_tool_call_handler:
                raise Utils.value_error(unknown_tool_call_handler=unknown_tool_call_handler, tool_call=tool_call)

        return tool_call_result_content

    async def _process_tool_call(self, *, tool_call: ToolCall) -> ChatMessage:
        content = await self._tool_call_result_content(tool_call=tool_call)
        chat_message = ChatMessage(role=Role.TOOL, tool_call_id=tool_call.id, content=content)

        logger.debug(tool_call_request=tool_call, tool_call_response=chat_message)

        return chat_message

    async def _queue_tool_results(self) -> None:
        async for tool_call in self.tool_call_queue:
            self.request_chat_message_task_queue.create_task(self._process_tool_call, tool_call=tool_call)

    async def _play_audio(self) -> None:
        async for audio in self.tts.iter_audio():
            await audio.play()

    @logger.instrument()
    async def _chat(self) -> None:
        await Utils.wait(
            self._queue_transcribed_user_messages(),
            self._prompt_language_model_and_queue_response(),
            self._process_language_model_responses(),
            self._queue_tool_results(),
            self._play_audio(),
        )

    @classmethod
    async def chat(
        cls,
        *,
        secret_filepath: Annotated[Path, Option("--secret")],
        debug_filepath: Annotated[Optional[Path], Option("--log")] = None,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        dtype: Annotated[Dtype, Option()] = Microphone.DEFAULT_DTYPE,
    ) -> None:
        with Utils.context_map(value=debug_filepath, fn=cls._open_file_in_write_mode) as debug_file:
            Logger.init(debug_file=debug_file)

            async with cls._acontext(secret_filepath=secret_filepath, device=device, dtype=dtype) as chat:
                await chat._chat()  # pylint: disable=protected-access
