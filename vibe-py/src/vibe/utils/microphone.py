import contextlib
import dataclasses
import functools
from enum import StrEnum
from typing import AsyncIterator, ClassVar, Self

import _cffi_backend  # ty: ignore[unresolved-import]
from _cffi_backend import _CDataBase as CDataBase  # pylint: disable=no-name-in-module  # ty: ignore[unresolved-import]
from sounddevice import CallbackFlags, RawInputStream

from vibe.utils.audio import Audio, AudioFormat, AudioInfo
from vibe.utils.logger import Logger
from vibe.utils.queue import Queue
from vibe.utils.utils import Utils

logger = Logger.new(__name__)


# TODO-959e20
@dataclasses.dataclass(kw_only=True)
class DtypeInfo:
    name: str
    audio_format: AudioFormat


# TODO-959e20
class Dtype(StrEnum):
    INT_16 = "INT_16"
    FLOAT_32 = "FLOAT_32"

    @functools.cached_property
    def info(self) -> DtypeInfo:
        match self:
            case Dtype.INT_16:
                return DtypeInfo(name="int16", audio_format=AudioFormat.PCM_16)
            case Dtype.FLOAT_32:
                return DtypeInfo(name="float32", audio_format=AudioFormat.FLOAT)
            case _:
                raise Utils.invalid_result(self)


@dataclasses.dataclass(kw_only=True)
class MicrophoneInput:
    byte_str: bytes
    audio: Audio

    @classmethod
    def new(cls, *, byte_str: bytes, audio_info: AudioInfo, dtype: Dtype) -> Self:
        audio = Audio.new(byte_str=byte_str, audio_format=dtype.info.audio_format, audio_info=audio_info)
        microphone_input = cls(byte_str=byte_str, audio=audio)

        return microphone_input


# NOTE: inspired by: [https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html#creating-an-asyncio-generator-for-audio-blocks]  # pylint: disable=line-too-long  # noqa: E501
@dataclasses.dataclass(kw_only=True)
class Microphone:
    DEFAULT_DEVICE: ClassVar[int] = 0
    DEFAULT_DTYPE: ClassVar[Dtype] = Dtype.FLOAT_32
    ZERO_INPUT_CHANNELS_ERROR_MESSAGE: ClassVar[str] = "selected device does not have any input channels"

    audio_info: AudioInfo
    dtype: Dtype
    queue: Queue[MicrophoneInput]

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
        microphone_input = MicrophoneInput.new(byte_str=indata[:], audio_info=self.audio_info, dtype=self.dtype)

        self.queue.event_loop().call_soon_threadsafe(self.queue.append, microphone_input)

    def _raw_input_stream(self) -> RawInputStream:
        return RawInputStream(
            samplerate=self.audio_info.sample_rate,
            channels=self.audio_info.num_channels,
            dtype=self.dtype.info.name,
            callback=self._callback,
        )

    @classmethod
    @contextlib.asynccontextmanager
    async def context(cls, *, device: int = DEFAULT_DEVICE, dtype: Dtype = DEFAULT_DTYPE) -> AsyncIterator[Self]:
        queue = await Queue.new()
        audio_info = AudioInfo.from_device(device=device)
        microphone = cls(audio_info=audio_info, dtype=dtype, queue=queue)

        if audio_info.num_channels == 0:
            raise ValueError(cls.ZERO_INPUT_CHANNELS_ERROR_MESSAGE)

        with microphone._raw_input_stream():
            yield microphone
