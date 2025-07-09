import collections
import dataclasses
from typing import Annotated, ClassVar, Iterator, Optional, Self

import silero_vad
from silero_vad import VADIterator
from torch import Tensor

from vibe.utils.audio import Audio
from vibe.utils.interval import Interval
from vibe.utils.logger import Logger
from vibe.utils.typing import JsonObject
from vibe.utils.utils import Shape, Utils

logger: Logger = Logger.new(__name__)


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
    iterator: VADIterator
    latest_interval_begin: Optional[int]
    results: list[VadResult]

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
            iterator=iterator,
            latest_interval_begin=None,
            results=[],
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

    @staticmethod
    def _vad_input(*, audio: Audio) -> Annotated[Tensor, Shape("C,F")]:
        return audio.data.T.float()

    def _iter_speech_dicts_helper(self) -> Iterator[JsonObject]:
        interval_iter = Utils.iter_intervals(
            begin=self.audio_cursor, total=self.audio.num_frames(), chunk_size=self.CHUNK_NUM_FRAMES, exact=True
        )

        for interval in interval_iter:
            self.audio_cursor = interval.end
            audio_chunk = self.audio.slice(begin=interval.begin, end=interval.end)
            vad_input = self._vad_input(audio=audio_chunk)
            speech_dict = self.iterator(vad_input)

            if speech_dict is not None:
                yield speech_dict

    @classmethod
    def _preprocess_audio(cls, *, audio: Audio) -> Audio:
        return audio.resample(sample_rate=cls.SAMPLE_RATE).mean(num_channels=cls.NUM_CHANNELS)

    def _iter_speech_dicts(self, *, audio: Audio) -> Iterator[JsonObject]:
        yield from self._iter_speech_dicts_helper()

        audio = self._preprocess_audio(audio=audio)

        self.audio.add(audio)

        yield from self._iter_speech_dicts_helper()

    def _on_end(self, *, interval_end: int) -> VadResult:
        interval = Interval[int](begin=self.latest_interval_begin, end=interval_end)
        result_audio = self.audio.slice(begin=interval.begin, end=interval.end)
        result = VadResult(interval=interval, audio=result_audio)
        self.latest_interval_begin = None

        self.results.append(result)

        return result

    def add(self, *, audio: Audio) -> Iterator[VadResult]:
        for speech_dict in self._iter_speech_dicts(audio=audio):
            match speech_dict:
                case {"start": int(latest_interval_begin)} if self.latest_interval_begin is None:
                    self.latest_interval_begin = latest_interval_begin
                case {"end": int(interval_end)} if self.latest_interval_begin is not None:
                    yield self._on_end(interval_end=interval_end)
                case _:
                    raise Utils.value_error(
                        message="invalid state",
                        speech_dict=speech_dict,
                        latest_interval_begin=self.latest_interval_begin,
                    )
