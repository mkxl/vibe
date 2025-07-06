import dataclasses
import datetime
from datetime import UTC
from typing import ClassVar, Self


@dataclasses.dataclass(kw_only=True)
class Datetime:
    MILLISECONDS_PER_SECOND: ClassVar[int] = 1000

    std_datetime: datetime.datetime

    @classmethod
    def now(cls) -> Self:
        return cls(std_datetime=datetime.datetime.now(UTC))

    def timestamp_ms(self) -> int:
        return int(self.std_datetime.timestamp() * self.MILLISECONDS_PER_SECOND)
