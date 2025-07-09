import asyncio
import collections
import dataclasses
from asyncio import AbstractEventLoop, Task
from typing import Any, AsyncIterator, ClassVar, Coroutine, Generator, Iterable, Self

from vibe.utils.sink import Sink

# NOTE:
# - prefer this implementation over the hume __await__ based one because of
#   [https://stackoverflow.com/a/63966273]
# - NOTE-b68740: maintains invariant that if [self.head] is pending then [self.tail] is empty
# @dataclasses.dataclass(kw_only=True)
# class Queue[T](Sink[T]):
#     APPEND_ITEM_ERROR_MESSAGE: ClassVar[str] = "unable to send value because the queue is closed"
#     INITIAL_IS_CLOSED: ClassVar[bool] = False

#     head: Future[T]
#     tail: collections.deque[T]
#     is_closed: bool

#     @classmethod
#     async def new(cls) -> Self:
#         head = await Utils.future()
#         queue = cls(head=head, tail=collections.deque(), is_closed=cls.INITIAL_IS_CLOSED)

#         return queue

#     async def __aiter__(self) -> AsyncIterator[T]:
#         while True:
#             # NOTE: break if self.head is pending and self.is_closed
#             if not self.head.done() and self.is_closed:
#                 break

#             old_head = await self.head
#             self.head = await Utils.future()

#             # NOTE: bc [self.head] is now pending, give it a value if possible in order to to maintain [NOTE-b68740]
#             if 0 < len(self.tail):
#                 self.head.set_result(self.tail.popleft())

#             yield old_head

#     async def asend(self, value: T) -> None:
#         if self.is_closed:
#             raise ValueError(self.APPEND_ITEM_ERROR_MESSAGE)

#         # NOTE: bc of [NOTE-b68740], in the [else] branch, can assume that [self.tail] is empty and just set
#         # [self.head]'s result directly
#         if self.head.done():
#             self.tail.append(value)
#         else:
#             self.head.set_result(value)

#     async def aclose(self) -> None:
#         self.is_closed = True

#     def event_loop(self) -> AbstractEventLoop:
#         return self.head.get_loop()


@dataclasses.dataclass(kw_only=True)
class Queue[T](Sink[T]):
    APPEND_ITEM_ERROR_MESSAGE: ClassVar[str] = "unable to append item because the queue is closed"
    INITIAL_IS_CLOSED: ClassVar[bool] = False

    items: collections.deque[T]
    is_closed: bool
    event_loop_value: AbstractEventLoop

    @classmethod
    def new(cls) -> Self:
        event_loop_value = asyncio.get_running_loop()
        queue = cls(items=collections.deque(), is_closed=cls.INITIAL_IS_CLOSED, event_loop_value=event_loop_value)

        return queue

    def __await__(self) -> Generator[Any, Any, T]:
        while True:
            if 0 < len(self.items):
                return self.items.popleft()

            if self.is_closed:
                raise StopAsyncIteration

            yield

    async def __anext__(self) -> T:
        return await self

    def __aiter__(self) -> Self:
        return self

    def append(self, item: T) -> None:
        if self.is_closed:
            raise ValueError(self.APPEND_ITEM_ERROR_MESSAGE)

        self.items.append(item)

    def extend(self, item_iter: Iterable[T]) -> None:
        for item in item_iter:
            self.append(item)

    async def asend(self, value: T) -> None:
        self.append(value)

    async def aclose(self) -> None:
        self.is_closed = True

    def event_loop(self) -> AbstractEventLoop:
        return self.event_loop_value


@dataclasses.dataclass(kw_only=True)
class TaskQueue[T]:
    task_queue: Queue[Task[T]]

    @classmethod
    def new(cls) -> Self:
        inner_task_queue = Queue.new()
        task_queue = cls(task_queue=inner_task_queue)

        return task_queue

    async def __aiter__(self) -> AsyncIterator[T]:
        async for task in self.task_queue:
            yield await task

    def append(self, coro: Coroutine[Any, Any, T]) -> None:
        task = asyncio.create_task(coro)

        self.task_queue.append(task)
