import dataclasses
from pathlib import Path
from typing import Annotated, ClassVar

import sounddevice
from typer import Option, Typer

from vibe.chat import Chat
from vibe.utils.audio import Audio, AudioFormat
from vibe.utils.datetime import Datetime
from vibe.utils.logger import Logger
from vibe.utils.microphone import Dtype, Microphone
from vibe.utils.utils import Utils

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class App:
    PRETTY_EXCEPTIONS_ENABLE: ClassVar[bool] = False

    @classmethod
    def cli(cls) -> None:
        typer = Typer(pretty_exceptions_enable=cls.PRETTY_EXCEPTIONS_ENABLE)

        Logger.init()

        Utils.add_typer_command(typer=typer, fn=cls.devices)
        Utils.add_typer_command(typer=typer, fn=cls.record)
        Utils.add_typer_command(typer=typer, fn=Chat.chat)

        typer()

    @staticmethod
    async def devices() -> None:
        devices_str = str(sounddevice.query_devices())

        print(devices_str)

    @classmethod
    async def record(
        cls,
        *,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        dtype: Annotated[Dtype, Option()] = Microphone.DEFAULT_DTYPE,
        out_dirpath: Annotated[Path, Option("--out-dir")] = Path("./"),
    ) -> None:
        timestamp_ms = Datetime.now().timestamp_ms()
        out_filepath = out_dirpath.joinpath(f"microphone-{timestamp_ms}.wav")
        audio_list = []

        try:
            async with Microphone.context(device=device, dtype=dtype) as microphone:
                async for microphone_input in microphone:
                    audio_list.append(microphone_input.audio)
        finally:
            wav_byte_str = Audio.cat(audio_list).byte_str(audio_format=AudioFormat.WAV)

            out_filepath.write_bytes(wav_byte_str)

            logger.info(out_filepath=str(out_filepath))
