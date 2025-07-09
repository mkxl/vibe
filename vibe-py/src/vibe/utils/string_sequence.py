import dataclasses
from typing import ClassVar, Iterable, Optional, Self


@dataclasses.dataclass(kw_only=True)
class StringSequence:
    INITIAL_LENGTH: ClassVar[int] = 0

    strings: list[str]
    length: int

    @classmethod
    def new(cls) -> Self:
        return cls(strings=[], length=cls.INITIAL_LENGTH)

    def is_empty(self) -> bool:
        return self.length == 0

    def is_nonempty(self) -> bool:
        return not self.is_empty()

    def string(self, strings: Optional[Iterable[str]] = None) -> str:
        # NOTE: construct a list rather than chain iterables bc of [https://stackoverflow.com/q/34822676]
        string_list = self.strings if strings is None else [*self.strings, *strings]
        joined_string = "".join(string_list)

        return joined_string

    def last(self) -> Optional[str]:
        return None if len(self.strings) == 0 else self.strings[-1]

    def append(self, string: str) -> None:
        self.strings.append(string)

        self.length += len(string)

    def take(self) -> str:
        string = self.string()
        self.length = 0

        self.strings.clear()

        return string
