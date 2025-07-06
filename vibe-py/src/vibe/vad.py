import collections
import dataclasses
from typing import ClassVar, Iterator, Optional, Self

import silero_vad
from silero_vad import VADIterator

from vibe.utils.audio import Audio
from vibe.utils.interval import Interval
from vibe.utils.logger import Logger
from vibe.utils.typing import JsonObject
from vibe.utils.utils import Utils

logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class VadResult:
    interval: Interval[int]
    audio: Audio


# NOTE: [https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#examples]
@dataclasses.dataclass(kw_only=True)
class Vad:
    # NOTE: DEFAULT_* gotten from hume and vad repo
    CHUNK_NUM_FRAMES: ClassVar[int] = 512
    DEFAULT_MIN_SILENCE_DURATION_MS: ClassVar[int] = 96
    DEFAULT_SPEECH_PAD_MS: ClassVar[int] = 300
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5
    INITIAL_CURSOR: ClassVar[int] = 0
    NUM_CHANNELS: ClassVar[int] = 1
    ONNX: ClassVar[bool] = False
    SAMPLE_RATE: ClassVar[int] = 16_000

    audio: Audio
    audio_chunks: collections.deque[Audio]
    audio_cursor: int
    interval_begin: Optional[int]
    iterator: VADIterator

    @classmethod
    def new(
        cls,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
    ) -> Self:
        audio = Audio.empty(sample_rate=cls.SAMPLE_RATE, num_channels=cls.NUM_CHANNELS)
        iterator = cls._iterator(
            threshold=threshold, min_silence_duration_ms=min_silence_duration_ms, speech_pad_ms=speech_pad_ms
        )
        vad = cls(
            audio=audio,
            audio_chunks=collections.deque(),
            audio_cursor=cls.INITIAL_CURSOR,
            interval_begin=None,
            iterator=iterator,
        )

        return vad

    @classmethod
    def _iterator(cls, *, threshold: float, min_silence_duration_ms: int, speech_pad_ms: int) -> VADIterator:
        model = silero_vad.load_silero_vad(onnx=cls.ONNX)
        iterator = VADIterator(
            model=model,
            threshold=threshold,
            sampling_rate=cls.SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        return iterator

    def _iter_speech_dicts_helper(self) -> Iterator[JsonObject]:
        interval_iter = Utils.iter_intervals(
            begin=self.audio_cursor, total=self.audio.num_frames(), chunk_size=self.CHUNK_NUM_FRAMES, exact=True
        )

        for interval in interval_iter:
            self.audio_cursor = interval.end
            audio_chunk = self.audio.slice(begin=interval.begin, end=interval.end)
            speech_dict = self.iterator(audio_chunk.vad_input())

            if speech_dict is not None:
                yield speech_dict

    @classmethod
    def _preprocess_audio(cls, *, audio: Audio) -> Audio:
        if audio.sample_rate != cls.SAMPLE_RATE:
            audio = audio.resample(sample_rate=cls.SAMPLE_RATE)

        return audio.mean(num_channels=cls.NUM_CHANNELS)

    def _iter_speech_dicts(self, *, audio: Audio) -> Iterator[JsonObject]:
        yield from self._iter_speech_dicts_helper()

        audio = self._preprocess_audio(audio=audio)

        self.audio.add(audio)

        yield from self._iter_speech_dicts_helper()

    def add(self, *, audio: Audio) -> Iterator[Audio]:
        for speech_dict in self._iter_speech_dicts(audio=audio):
            match speech_dict:
                case {"start": int(interval_begin)} if self.interval_begin is None:
                    self.interval_begin = interval_begin
                case {"end": int(interval_end)} if self.interval_begin is not None:
                    interval = Interval[int](begin=self.interval_begin, end=interval_end)
                    vad_result_audio = self.audio.slice(begin=interval.begin, end=interval.end)
                    vad_result = VadResult(interval=interval, audio=vad_result_audio)
                    self.interval_begin = None

                    yield vad_result
                case _:
                    raise Utils.value_error(
                        message="invalid state", speech_dict=speech_dict, interval_begin=self.interval_begin
                    )
