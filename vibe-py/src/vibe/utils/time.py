import asyncio
import contextlib
import dataclasses
import datetime
from datetime import UTC
from typing import AsyncIterator, ClassVar, Self


@dataclasses.dataclass(kw_only=True)
class Datetime:
    std_datetime: datetime.datetime

    @classmethod
    def now(cls) -> Self:
        return cls(std_datetime=datetime.datetime.now(UTC))

    def __str__(self) -> str:
        return str(self.std_datetime)

    def timestamp(self) -> "Duration":
        return Duration.new(seconds=self.std_datetime.timestamp())


@dataclasses.dataclass(kw_only=True)
class Duration:
    DEFAULT_SECONDS: ClassVar[float] = 0.0
    DEFAULT_MILLISECONDS: ClassVar[int] = 0
    MILLISECONDS_PER_SECOND: ClassVar[int] = 1000
    SECONDS_PER_HOUR: ClassVar[int] = 60

    timedelta: datetime.timedelta

    @classmethod
    def new(cls, *, seconds: float = DEFAULT_SECONDS, milliseconds: int = DEFAULT_MILLISECONDS) -> Self:
        timedelta = datetime.timedelta(seconds=seconds, milliseconds=milliseconds)
        duration = cls(timedelta=timedelta)

        return duration

    def __str__(self) -> str:
        return self.string()

    def seconds(self) -> float:
        return self.timedelta.total_seconds()

    def milliseconds(self) -> int:
        return int(self.seconds() * self.MILLISECONDS_PER_SECOND)

    def string(self) -> str:
        total_seconds = self.seconds()
        prefix, total_seconds = ("-", abs(total_seconds)) if total_seconds < 0 else ("", total_seconds)
        total_minutes, seconds = divmod(total_seconds, self.SECONDS_PER_HOUR)
        hours, minutes = divmod(total_minutes, self.SECONDS_PER_HOUR)
        hours = int(hours)
        minutes = int(minutes)
        string = f"{prefix}{hours}:{minutes:02d}:{seconds:06.3f}"

        return string

    async def iter_datetimes(self) -> AsyncIterator[Datetime]:
        while True:
            yield Datetime.now()

            await self.sleep()

    async def sleep(self) -> None:
        await asyncio.sleep(self.seconds())

    @contextlib.asynccontextmanager
    async def wait(self) -> AsyncIterator[None]:
        try:
            async with asyncio.timeout(self.seconds()):
                yield None
        except TimeoutError:
            pass
