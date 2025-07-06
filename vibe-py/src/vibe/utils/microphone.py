import contextlib
import dataclasses
from typing import AsyncIterator, ClassVar, Self

import _cffi_backend  # ty: ignore[unresolved-import]
from _cffi_backend import _CDataBase as CDataBase  # pylint: disable=no-name-in-module  # ty: ignore[unresolved-import]
from sounddevice import CallbackFlags, RawInputStream

from vibe.utils.audio import Audio, AudioInfo
from vibe.utils.logger import Logger
from vibe.utils.queue import Queue

logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class MicrophoneInput:
    pcm_16_byte_str: bytes
    audio: Audio

    @classmethod
    def new(cls, *, pcm_16_byte_str: bytes, audio_info: AudioInfo) -> Self:
        audio = Audio.from_pcm_16_byte_str(pcm_16_byte_str, audio_info=audio_info)
        microphone_input = cls(pcm_16_byte_str=pcm_16_byte_str, audio=audio)

        return microphone_input


# NOTE: inspired by: [https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html#creating-an-asyncio-generator-for-audio-blocks]  # pylint: disable=line-too-long  # noqa: E501
@dataclasses.dataclass(kw_only=True)
class Microphone:
    DTYPE: ClassVar[str] = "int16"
    DEFAULT_DEVICE: ClassVar[int] = 0
    ZERO_INPUT_CHANNELS_ERROR_MESSAGE: ClassVar[str] = "selected device does not have any input channels"

    queue: Queue[MicrophoneInput]
    audio_info: AudioInfo

    @classmethod
    @contextlib.asynccontextmanager
    async def context(cls, *, device: int = DEFAULT_DEVICE) -> AsyncIterator[Self]:
        queue = await Queue.new()
        audio_info = AudioInfo.from_device(device=device)
        microphone = cls(queue=queue, audio_info=audio_info)

        if audio_info.num_channels == 0:
            raise ValueError(cls.ZERO_INPUT_CHANNELS_ERROR_MESSAGE)

        with microphone._raw_input_stream():
            yield microphone

    def __aiter__(self) -> AsyncIterator[MicrophoneInput]:
        return self.queue

    # NOTE: type annotations gotten from logging
    def _callback(
        self,
        indata: _cffi_backend.buffer,  # pylint: disable=c-extension-no-member
        _frame_count: int,
        _time_info: CDataBase,
        _status: CallbackFlags,
    ) -> None:
        # NOTE: _cffi_backend.buffer returns bytes when sliced
        microphone_input = MicrophoneInput.new(pcm_16_byte_str=indata[:], audio_info=self.audio_info)

        self.queue.event_loop().call_soon_threadsafe(self.queue.append, microphone_input)

    def _raw_input_stream(self) -> RawInputStream:
        return RawInputStream(
            samplerate=self.audio_info.sample_rate,
            channels=self.audio_info.num_channels,
            dtype=self.DTYPE,
            callback=self._callback,
        )
