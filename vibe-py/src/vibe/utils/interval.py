import dataclasses
from typing import Generic

from vibe.utils.typing import T


@dataclasses.dataclass
class Interval(Generic[T]):
    begin: T
    end: T
