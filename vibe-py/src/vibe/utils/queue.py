import asyncio
import collections
import dataclasses
from asyncio import Task
from typing import Any, AsyncIterable, AsyncIterator, ClassVar, Coroutine, Generator, Self


# TODO: rewrite like [scratch/old-queue.py] because of [https://stackoverflow.com/a/63966273]
@dataclasses.dataclass(kw_only=True)
class Queue[T]:
    APPEND_ITEM_ERROR_MESSAGE: ClassVar[str] = "unable to append item because the queue is closed"
    INITIAL_CLOSED: ClassVar[bool] = False

    items: collections.deque[T]
    closed: bool

    @classmethod
    def new(cls) -> Self:
        return cls(items=collections.deque(), closed=cls.INITIAL_CLOSED)

    def __await__(self) -> Generator[Any, Any, T]:
        while True:
            if 0 < len(self.items):
                return self.items.popleft()

            if self.closed:
                raise StopAsyncIteration

            yield

    async def __anext__(self) -> T:
        return await self

    def __aiter__(self) -> Self:
        return self

    def append(self, item: T) -> None:
        if self.closed:
            raise ValueError(self.APPEND_ITEM_ERROR_MESSAGE)

        self.items.append(item)

    async def aclose(self) -> None:
        self.closed = True

    async def consume(self, items: AsyncIterable[T]) -> None:
        async for item in items:
            self.append(item)


@dataclasses.dataclass(kw_only=True)
class TaskQueue[T]:
    tasks: Queue[Task[T]]

    @classmethod
    def new(cls) -> Self:
        return cls(tasks=Queue.new())

    async def __aiter__(self) -> AsyncIterator[T]:
        async for task in self.tasks:
            yield await task

    def append(self, coro: Coroutine[Any, Any, T]) -> None:
        task = asyncio.create_task(coro)

        self.tasks.append(task)
