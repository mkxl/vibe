import base64
import contextlib
import dataclasses
from typing import Annotated, AsyncIterator, ClassVar, Optional, Self, Union

import websockets.asyncio.client
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from websockets.asyncio.client import ClientConnection

from vibe.tts import Tts
from vibe.utils.audio import Audio, AudioFormat, AudioInfo
from vibe.utils.logger import Logger
from vibe.utils.typing import JsonObject
from vibe.utils.utils import Utils

type OutputMessage = Union["AudioOutput", "FinalOutput"]

logger: Logger = Logger.new(__name__)


class VoiceSettings(BaseModel):
    speed: float


class InputMessage(BaseModel):
    text: str
    voice_settings: Optional[VoiceSettings]
    flush: Optional[bool]


class AudioOutput(BaseModel):
    audio: str

    def as_bytes(self) -> bytes:
        return base64.b64decode(self.audio)


class FinalOutput(BaseModel):
    VALIDATION_ALIAS_IS_FINAL: ClassVar[str] = "isFinal"

    is_final: Annotated[Optional[bool], Field(validation_alias=VALIDATION_ALIAS_IS_FINAL)]


@dataclasses.dataclass(kw_only=True)
class ElevenLabs(Tts):
    AUDIO_FORMAT: ClassVar[Optional[AudioFormat]] = AudioFormat.PCM_16
    AUDIO_INFO: ClassVar[Optional[AudioInfo]] = AudioInfo(sample_rate=16_000, num_channels=1)
    HEADER_NAME_API_KEY: ClassVar[str] = "xi-api-key"
    INACTIVITY_TIMEOUT: ClassVar[int] = 180
    INIT_MESSAGE_FLUSH: ClassVar[Optional[bool]] = None
    INIT_MESSAGE_TEXT: ClassVar[str] = " "
    MAX_SIZE: ClassVar[int] = 2**24
    NORMAL_MESSAGE_FLUSH: ClassVar[Optional[bool]] = True
    OUTPUT_FORMAT: ClassVar[str] = "pcm_16000"
    QUERY_PARAM_NAME_INACTIVITY_TIMEOUT: ClassVar[str] = "inactivity_timeout"
    QUERY_PARAM_NAME_OUTPUT_FORMAT: ClassVar[str] = "output_format"
    SPEED: ClassVar[float] = 1
    OUTPUT_MESSAGE_TYPE_ADAPTER: ClassVar[TypeAdapter] = TypeAdapter(OutputMessage)

    websocket: ClientConnection

    @classmethod
    @contextlib.asynccontextmanager
    async def acontext(cls, *, voice_id: str, api_key: str) -> AsyncIterator[Self]:
        uri = cls._uri(voice_id=voice_id)
        additional_headers = cls._additional_headers(api_key=api_key)
        websocket_cm = websockets.asyncio.client.connect(
            uri, additional_headers=additional_headers, max_size=cls.MAX_SIZE
        )

        async with websocket_cm as websocket:
            eleven_labs = cls(websocket=websocket)

            await eleven_labs._init()

            yield eleven_labs

    @classmethod
    def _uri(cls, *, voice_id: str) -> str:
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        query_params = {
            cls.QUERY_PARAM_NAME_OUTPUT_FORMAT: cls.OUTPUT_FORMAT,
            cls.QUERY_PARAM_NAME_INACTIVITY_TIMEOUT: cls.INACTIVITY_TIMEOUT,
        }
        url = Utils.url(url=url, query_params=query_params)

        return url

    @classmethod
    def _additional_headers(cls, *, api_key: str) -> JsonObject:
        return {cls.HEADER_NAME_API_KEY: api_key}

    async def _asend(self, *, text: str, voice_settings: Optional[VoiceSettings], flush: Optional[bool]) -> None:
        input_message = InputMessage(text=text, voice_settings=voice_settings, flush=flush)

        await self.websocket.send(input_message.model_dump_json())

    async def _init(self) -> None:
        voice_settings = VoiceSettings(speed=self.SPEED)

        await self._asend(text=self.INIT_MESSAGE_TEXT, voice_settings=voice_settings, flush=self.INIT_MESSAGE_FLUSH)

    async def asend(self, text: str) -> None:
        await self._asend(text=text, voice_settings=None, flush=self.NORMAL_MESSAGE_FLUSH)

    @classmethod
    def _output_message(cls, *, json_str: str) -> Union[OutputMessage, str]:
        try:
            return cls.OUTPUT_MESSAGE_TYPE_ADAPTER.validate_json(json_str)
        except ValidationError:
            return json_str

    async def iter_audio_byte_strs(self) -> AsyncIterator[bytes]:
        async for json_str in self.websocket:
            match self._output_message(json_str=json_str):
                case AudioOutput() as audio_output:
                    yield audio_output.as_bytes()
                case FinalOutput() as final_output:
                    logger.debug(final_output=final_output)

                    break
                case unknown_message:
                    logger.warning(unknown_message=unknown_message)

    # pylint: disable=invalid-overridden-method
    async def iter_audio(self) -> AsyncIterator[Audio]:
        async for audio_byte_str in self.iter_audio_byte_strs():
            yield Audio.new(byte_str=audio_byte_str, audio_format=self.AUDIO_FORMAT, audio_info=self.AUDIO_INFO)
