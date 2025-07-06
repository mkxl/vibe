import contextlib
import dataclasses
import uuid
from pathlib import Path
from typing import Annotated, ClassVar, Iterator

from humeai.api.assistant.config import Config, Secret
from humeai.api.assistant.evi_version import EviVersion
from humeai.api.assistant.service import Service
from humeai.api.assistant.user_turn import UserTurnHandler
from humeai.model.facade.features import FeatureToggleContext
from typer import Option

from vibe.utils.audio import Audio, AudioFormats
from vibe.utils.logger import Logger
from vibe.utils.microphone import Microphone
from vibe.utils.utils import Utils

logger = Logger.new(__name__)


@dataclasses.dataclass
class RequestInfo:
    user_id: str
    user_email: str
    request_id: str


@dataclasses.dataclass
class Main:
    DEFAULT_USER_ID: ClassVar[str] = "c8e7d7a5-e74a-48dd-aa90-a788b5aad36e"
    DEFAULT_USER_EMAIL: ClassVar[str] = "dev@hume.ai"
    DEFAULT_REQUEST_ID: ClassVar[str] = str(uuid.uuid4())
    EVI_VERSION: ClassVar[EviVersion] = EviVersion.V3

    microphone: Microphone
    request_info: RequestInfo
    service: Service
    user_turn_handler: UserTurnHandler

    @staticmethod
    def _service(*, secret_filepath: Path) -> Service:
        secret = Secret.from_filepath(secret_filepath)
        config = Config(secret=secret)
        service = Service.new(config=config)

        return service

    @staticmethod
    def _feature_toggle_context(*, service: Service, user_id: str, user_email: str) -> FeatureToggleContext:
        return service.feature_toggle_client.get_context(user_id=user_id, email=user_email)

    # pylint: disable=too-many-arguments
    @classmethod
    async def run(
        cls,
        *,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        secret_filepath: Annotated[Path, Option("--secret")],
        user_id: Annotated[str, Option()] = DEFAULT_USER_ID,
        user_email: Annotated[str, Option()] = DEFAULT_USER_EMAIL,
        request_id: Annotated[str, Option()] = DEFAULT_REQUEST_ID,
    ) -> None:
        async with Microphone.context(device=device) as microphone:
            # NOTE: set [feature_toggle_context] to None to disable noise cancellation
            request_info = RequestInfo(user_id=user_id, user_email=user_email, request_id=request_id)
            service = cls._service(secret_filepath=secret_filepath)
            user_turn_handler = UserTurnHandler.new(
                service=service,
                request_id=request_id,
                evi_version=cls.EVI_VERSION,
                feature_toggle_context=None,
            )
            main = cls(
                microphone=microphone, request_info=request_info, service=service, user_turn_handler=user_turn_handler
            )

            await main._run()

    @contextlib.contextmanager
    def _audio_writer(self, *, filepath_str: str) -> Iterator[list[Audio]]:
        audio_list = []

        try:
            yield audio_list
        finally:
            wav_byte_str = Audio.cat(audio_list).byte_str(audio_format=AudioFormats.WAV)

            Path(filepath_str).write_bytes(wav_byte_str)

    async def _feed(self) -> None:
        async for microphone_input in self.microphone:
            self.user_turn_handler.set_first_audio_chunk_received()

            audio = microphone_input.audio.resample(sample_rate=UserTurnHandler.SAMPLE_RATE).mono()
            byte_str = audio.byte_str(audio_format=AudioFormats.FLOAT)

            await self.user_turn_handler.push(byte_str)
            await Utils.yield_now()

    async def _process(self) -> None:
        async for response in self.user_turn_handler.iter_responses():
            logger.info(response=response)

    async def _run(self) -> None:
        logger.info(request_info=self.request_info)

        await Utils.wait(self._feed(), self._process())
