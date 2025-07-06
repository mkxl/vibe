import dataclasses
from collections.abc import Buffer
from io import BytesIO
from typing import Annotated, BinaryIO, ClassVar, Iterable, Optional, Self

import sounddevice
import soundfile
import torch
import torchaudio
from pydub import AudioSegment
from torch import Tensor

from vibe.utils.typing import JsonObject
from vibe.utils.utils import Shape


@dataclasses.dataclass(kw_only=True)
class AudioFormat:
    format: str
    subtype: Optional[str]

    def pair(self) -> tuple[str, Optional[str]]:
        return (self.format, self.subtype)


class AudioFormats:
    WAV = AudioFormat(format="WAV", subtype=None)
    PCM_16 = AudioFormat(format="RAW", subtype="PCM_16")
    FLOAT = AudioFormat(format="RAW", subtype="FLOAT")

    @classmethod
    def from_str(cls, name: str) -> AudioFormat:
        return vars(cls)[name.upper()]


@dataclasses.dataclass(kw_only=True)
class AudioInfo:
    sample_rate: int
    num_channels: int

    @classmethod
    def from_device(cls, device: int) -> Self:
        # NOTE: [https://python-sounddevice.readthedocs.io/en/0.5.1/api/checking-hardware.html#sounddevice.query_devices]  # pylint: disable=line-too-long  # noqa: E501
        sound_device = sounddevice.query_devices(device=device)
        sample_rate = int(sound_device["default_samplerate"])
        num_channels = sound_device["max_input_channels"]
        audio_info = cls(sample_rate=sample_rate, num_channels=num_channels)

        return audio_info

    def pair(self) -> tuple[int, int]:
        return (self.sample_rate, self.num_channels)


@dataclasses.dataclass(kw_only=True)
class Audio:
    ALWAYS_2D: ClassVar[bool] = True
    DIM_CHANNELS: ClassVar[int] = 1
    DIM_FRAMES: ClassVar[int] = 0
    SAMPLE_WIDTH_PCM_16: ClassVar[int] = 2

    data: Annotated[Tensor, Shape("F,C")]
    sample_rate: int

    # NOTE: implemented to avoid having to do [type(self)(data=data, sample_rate=sample_rate)]
    @classmethod
    def from_values(cls, *, data: Annotated[Tensor, Shape("F,C")], sample_rate: int) -> Self:
        return cls(data=data, sample_rate=sample_rate)

    @classmethod
    def empty(cls, *, sample_rate: int, num_channels: int) -> Self:
        data = torch.zeros(0, num_channels)
        audio = cls.from_values(data=data, sample_rate=sample_rate)

        return audio

    # pylint: disable=redefined-builtin
    @classmethod
    def from_file(
        cls, *, file: BinaryIO, audio_format: Optional[AudioFormat] = None, audio_info: Optional[AudioInfo] = None
    ) -> Self:
        format, subtype = (None, None) if audio_format is None else audio_format.pair()
        samplerate, channels = (None, None) if audio_info is None else audio_info.pair()
        data_np, sample_rate = soundfile.read(
            file,
            always_2d=cls.ALWAYS_2D,
            samplerate=samplerate,
            channels=channels,
            format=format,
            subtype=subtype,
        )
        data = torch.from_numpy(data_np)
        audio = cls.from_values(data=data, sample_rate=sample_rate)

        return audio

    @classmethod
    def from_pcm_16_byte_str(cls, pcm_16_byte_str: Buffer, *, audio_info: AudioInfo) -> Self:
        file = BytesIO(pcm_16_byte_str)
        audio = cls.from_file(file=file, audio_format=AudioFormats.PCM_16, audio_info=audio_info)

        return audio

    # NOTE-37cb5e: assumes all the audio instances share the same sample rate
    @classmethod
    def cat(cls, audio_iter: Iterable[Self]) -> Self:
        audio_iter = iter(audio_iter)
        first_audio = next(audio_iter)
        data_list = [first_audio.data]

        for audio in audio_iter:
            data_list.append(audio.data)

        data = torch.vstack(data_list)
        audio = cls.from_values(data=data, sample_rate=first_audio.sample_rate)

        return audio

    # NOTE-37cb5e
    def add(self, other: Self) -> None:
        data_list = [self.data, other.data]
        self.data = torch.vstack(data_list)

    def with_data(self, data: Annotated[Tensor, Shape("F,C")]) -> None:
        return self.from_values(data=data, sample_rate=self.sample_rate)

    def slice(self, *, begin: int, end: int) -> Self:
        data = self.data[begin:end, :]
        audio = self.with_data(data)

        return audio

    def resample(self, *, sample_rate: int) -> Self:
        data = torchaudio.functional.resample(waveform=self.data.T, orig_freq=self.sample_rate, new_freq=sample_rate).T
        audio = self.from_values(data=data, sample_rate=sample_rate)

        return audio

    def mean(self, *, num_channels: int) -> Self:
        data = self.data.mean(dim=self.DIM_CHANNELS, keepdim=True).repeat(1, num_channels)
        audio = self.with_data(data)

        return audio

    def split(self, *, num_frames: int) -> tuple[Self, Self]:
        data_1 = self.data[:num_frames, :]
        data_2 = self.data[num_frames:, :]
        audio_1 = self.from_values(data=data_1, sample_rate=self.sample_rate)
        audio_2 = self.from_values(data=data_2, sample_rate=self.sample_rate)

        return (audio_1, audio_2)

    def describe(self) -> JsonObject:
        return {
            "sample_rate": self.sample_rate,
            "shape": tuple(self.data.shape),
        }

    def num_frames(self) -> int:
        return self.data.shape[self.DIM_FRAMES]

    def num_channels(self) -> int:
        return self.data.shape[self.DIM_CHANNELS]

    def info(self) -> AudioInfo:
        return AudioInfo(sample_rate=self.sample_rate, num_channels=self.num_channels())

    def byte_str(self, *, audio_format: AudioFormat) -> bytes:
        bytes_io = BytesIO()
        soundfile.write(
            bytes_io,
            self.data,
            samplerate=self.sample_rate,
            subtype=audio_format.subtype,
            format=audio_format.format,
        )

        return bytes_io.getvalue()

    def segment(self) -> AudioSegment:
        data = self.byte_str(audio_format=AudioFormats.PCM_16)
        audio_segment = AudioSegment(
            data=data, sample_width=self.SAMPLE_WIDTH_PCM_16, frame_rate=self.sample_rate, channels=self.num_channels()
        )

        return audio_segment

    def vad_input(self) -> Annotated[Tensor, Shape("C,F")]:
        return self.data.T.float()
