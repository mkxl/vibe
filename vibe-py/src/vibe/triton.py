import dataclasses
from enum import Enum
from typing import Any, ClassVar, Optional, Protocol, Self

import numpy
import torch
from torch import Tensor
from tritonclient.http.aio import InferenceServerClient, InferInput, InferRequestedOutput, InferResult

from vibe.utils.audio import Audio
from vibe.utils.logger import Logger
from vibe.utils.utils import Function

logger: Logger = Logger.new(__name__)


class Dtype(Enum):
    BOOL = torch.bool
    FP16 = torch.float16
    FP32 = torch.float32
    INT32 = torch.int32
    INT64 = torch.int64


@dataclasses.dataclass(kw_only=True)
class TritonResponse:
    infer_result: InferResult

    @classmethod
    def new(cls, *, infer_result: InferResult) -> Self:
        return cls(infer_result=infer_result)

    def to_array(self, *, name: str) -> numpy.ndarray:
        return self.infer_result.as_numpy(name=name)

    def to_tensor(self, *, name: str) -> Tensor:
        array = self.to_array(name=name)
        tensor = torch.from_numpy(array)

        return tensor

    # NOTE: returns an arbitrarily-nested list with the same shape as the result array
    def to_list(self, *, name: str) -> list[Any]:
        return self.to_array(name=name).tolist()

    def map[**P, T](self, *, name: str, fn: Function[P, T]) -> list[Any]:
        array = self.to_array(name=name)
        values = fn(array).tolist()

        return values

    @staticmethod
    @numpy.vectorize
    def optional_string(item: Optional[bytes]) -> Optional[str]:
        return None if item is None else item.decode("utf-8")

    @staticmethod
    @numpy.vectorize
    def bool(item: Any) -> bool:
        return bool(item)


class TritonRequest[T](Protocol):
    MODEL_NAME: ClassVar[str]
    OUTPUT_NAMES: ClassVar[tuple[str, ...]]
    URL: ClassVar[str]

    @staticmethod
    def _infer_input(*, name: str, array: numpy.ndarray, dtype: Dtype) -> InferInput:
        infer_input = InferInput(name=name, shape=array.shape, datatype=dtype.name)

        infer_input.set_data_from_numpy(array)

        return infer_input

    @classmethod
    def _tensor_infer_input(cls, *, name: str, tensor: Tensor, dtype: Dtype) -> InferInput:
        array = tensor.to(dtype=dtype.value).numpy()
        tensor_infer_input = cls._infer_input(name=name, array=array, dtype=dtype)

        return tensor_infer_input

    @classmethod
    def _bool_infer_input(cls, *, name: str, value: bool) -> InferInput:
        array = numpy.array([value])
        bool_infer_input = cls._infer_input(name=name, array=array, dtype=Dtype.BOOL)

        return bool_infer_input

    def _infer_inputs(self) -> list[InferInput]:
        raise NotImplementedError()

    def _result(self, *, triton_response: TritonResponse) -> T:
        raise NotImplementedError

    async def result(self) -> T:
        async with InferenceServerClient(url=self.URL) as client:
            outputs = [InferRequestedOutput(output_name) for output_name in self.OUTPUT_NAMES]
            infer_result = await client.infer(model_name=self.MODEL_NAME, inputs=self._infer_inputs(), outputs=outputs)

        infer_result.get_response()

        triton_response = TritonResponse.new(infer_result=infer_result)
        result = self._result(triton_response=triton_response)

        return result


@dataclasses.dataclass(kw_only=True)
class Transcription(TritonRequest[Optional[str]]):
    DTYPE_AUDIO: ClassVar[Dtype] = Dtype.FP32
    MODEL_NAME: ClassVar[str] = "parakeet"
    NAME_AUDIO: ClassVar[str] = "AUDIO"
    NAME_TEXT = "TEXT"
    NUM_CHANNELS: ClassVar[int] = 1
    OUTPUT_NAMES: ClassVar[tuple[str, ...]] = (NAME_TEXT,)
    SAMPLE_RATE: ClassVar[int] = 16_000
    URL: ClassVar[str] = "triton-parakeet-test-internal"

    audio: Audio

    def _infer_inputs(self) -> list[InferInput]:
        audio = self.audio.resample(sample_rate=self.SAMPLE_RATE).mean(num_channels=self.NUM_CHANNELS)
        tensor = audio.data.squeeze(Audio.DIM_CHANNELS)
        infer_input = self._tensor_infer_input(name=self.NAME_AUDIO, tensor=tensor, dtype=self.DTYPE_AUDIO)

        return [infer_input]

    def _result(self, *, triton_response: TritonResponse) -> Optional[str]:
        return triton_response.map(name=self.NAME_TEXT, fn=TritonResponse.optional_string)[0]


@dataclasses.dataclass(kw_only=True)
class EotDetectionResult:
    eot: bool
    wait: float


@dataclasses.dataclass(kw_only=True)
class EotDetection(TritonRequest[EotDetectionResult]):
    DTYPE_AUDIO: ClassVar[Dtype] = Dtype.FP32
    INPUT_DURATION_SECS: ClassVar[float] = 10.0
    MODEL_NAME: ClassVar[str] = "eot-detection"
    NAME_AUDIO: ClassVar[str] = "AUDIO"
    NAME_END_OF_TURN: ClassVar[str] = "END_OF_TURN"
    NAME_USER_LAST = "USER_LAST"
    NAME_WAIT = "WAIT"
    NUM_CHANNELS: ClassVar[int] = 1
    OUTPUT_NAMES: ClassVar[tuple[str, ...]] = (NAME_END_OF_TURN, NAME_WAIT)
    SAMPLE_RATE: ClassVar[int] = 16_000
    URL: ClassVar[str] = "triton-eot-detection-test-internal"

    audio: Audio
    user_last: bool

    def _infer_inputs(self) -> list[InferInput]:
        end = int(self.INPUT_DURATION_SECS * self.SAMPLE_RATE)
        audio = (
            self.audio.resample(sample_rate=self.SAMPLE_RATE)
            .mean(num_channels=self.NUM_CHANNELS)
            .slice(begin=0, end=end)
        )
        audio_infer_input = self._tensor_infer_input(name=self.NAME_AUDIO, tensor=audio.data.T, dtype=self.DTYPE_AUDIO)
        user_last_infer_input = self._bool_infer_input(name=self.NAME_USER_LAST, value=self.user_last)

        return [audio_infer_input, user_last_infer_input]

    def _result(self, *, triton_response: TritonResponse) -> EotDetectionResult:
        # NOTE: despite [backend/src/humeai_triton/models/eot-detection/config.pbtxt], the eot infer result has dtype
        # fp32
        eot = triton_response.map(name=self.NAME_END_OF_TURN, fn=TritonResponse.bool)[0]
        wait = triton_response.to_list(name=self.NAME_WAIT)[0]
        eot_detection_result = EotDetectionResult(eot=eot, wait=wait)

        return eot_detection_result
