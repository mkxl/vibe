import asyncio
import collections
import dataclasses
from asyncio import AbstractEventLoop, Future
from typing import ClassVar, Self


# NOTE: prefer this implementation over the hume __await__ based one because of
# [https://stackoverflow.com/a/63966273]
@dataclasses.dataclass(kw_only=True)
class Queue[T]:
    APPEND_ITEM_ERROR_MESSAGE: ClassVar[str] = "unable to append item because the queue is closed"
    INITIAL_CLOSED: ClassVar[bool] = False

    head: Future[T]
    items: collections.deque[T]
    closed: bool

    @classmethod
    async def new(cls) -> Self:
        head = await cls._head()
        queue = cls(head=head, items=collections.deque(), closed=cls.INITIAL_CLOSED)

        return queue

    # NOTE: make this an async method to ensure it's being called from an async context to ensure that
    # [asyncio.get_running_loop()] can run
    @classmethod
    async def _head(cls) -> Future[T]:
        # NOTE:
        # - prefer [asyncio.get_running_loop()] over [asyncio.get_event_loop] per
        #   [https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop]
        # - prefer [loop.create_future()] over [asyncio.Future()] per
        #   [https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop]
        return asyncio.get_running_loop().create_future()

    async def __anext__(self) -> T:
        head = await self.head

        self.head = await self._head()

        if 0 < len(self.items):
            self.head.set_result(self.items.popleft())

        return head

    def __aiter__(self) -> Self:
        return self

    def event_loop(self) -> AbstractEventLoop:
        return self.head.get_loop()

    def append(self, item: T) -> None:
        if self.closed:
            raise ValueError(self.APPEND_ITEM_ERROR_MESSAGE)

        if self.head.done():
            self.items.append(item)
        else:
            self.head.set_result(item)

    async def aclose(self) -> None:
        self.closed = True
