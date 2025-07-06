import dataclasses


@dataclasses.dataclass
class Interval[T]:
    begin: T
    end: T
