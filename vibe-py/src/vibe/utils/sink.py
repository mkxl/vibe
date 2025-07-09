import contextlib
from typing import AsyncIterable, Protocol


class Sink[T](Protocol):
    async def asend(self, value: T) -> None:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError

    async def aconsume(self, value_iter: AsyncIterable[T]) -> None:
        async with contextlib.aclosing(self):
            async for value in value_iter:
                await self.asend(value)
