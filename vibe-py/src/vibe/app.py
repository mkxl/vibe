import asyncio
import dataclasses
from pathlib import Path
from typing import Annotated, ClassVar, Optional

import sounddevice
from typer import Option, Typer

from vibe.utils.audio import Audio, AudioFormats
from vibe.utils.datetime import Datetime
from vibe.utils.logger import Logger
from vibe.utils.microphone import Microphone
from vibe.utils.utils import Utils
from vibe.vad import Vad

logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class App:
    PRETTY_EXCEPTIONS_ENABLE: ClassVar[bool] = False

    @classmethod
    def cli(cls) -> None:
        typer = Typer(pretty_exceptions_enable=cls.PRETTY_EXCEPTIONS_ENABLE)

        Logger.init()

        Utils.add_typer_command(typer=typer, fn=cls.devices)
        Utils.add_typer_command(typer=typer, fn=cls.microphone)
        Utils.add_typer_command(typer=typer, fn=cls.vad)

        typer()

    @staticmethod
    async def devices() -> None:
        devices_str = str(sounddevice.query_devices())

        print(devices_str)

    @classmethod
    async def microphone(
        cls,
        *,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
        timeout_secs: Annotated[float, Option("--timeout")] = 3,
        out_dirpath: Annotated[Path, Option("--out-dir")] = Path("./"),
        sample_rate: Annotated[Optional[int], Option()] = None,
    ) -> None:
        timestamp_ms = Datetime.now().timestamp_ms()
        out_filepath = out_dirpath.joinpath(f"microphone-{timestamp_ms}.wav")
        audio_list = []

        async with Microphone.context(device=device) as microphone:
            try:
                async with asyncio.timeout(timeout_secs):
                    async for microphone_input in microphone:
                        audio_list.append(microphone_input.audio)
            except TimeoutError:
                pass

            audio = Audio.cat(audio_list)

            if sample_rate is not None:
                audio = audio.resample(sample_rate=sample_rate)

            wav_byte_str = audio.byte_str(audio_format=AudioFormats.WAV)

            out_filepath.write_bytes(wav_byte_str)

    @classmethod
    async def vad(
        cls,
        *,
        device: Annotated[int, Option()] = Microphone.DEFAULT_DEVICE,
    ) -> None:
        with logger.bookend(message="vad"):
            vad = Vad.new()

            async with Microphone.context(device=device) as microphone:
                async for microphone_input in microphone:
                    for vad_result in vad.add(audio=microphone_input.audio):
                        logger.info(vad_result_interval=vad_result.interval)
