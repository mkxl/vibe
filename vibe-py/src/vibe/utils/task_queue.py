import asyncio
import dataclasses
from asyncio import Task
from typing import Any, AsyncIterator, Coroutine, Self

from vibe.utils.queue import Queue


@dataclasses.dataclass(kw_only=True)
class TaskQueue[T]:
    tasks: Queue[Task[T]]

    @classmethod
    async def new(cls) -> Self:
        tasks = await Queue[Task[T]].new()
        task_queue = cls(tasks=tasks)

        return task_queue

    async def __aiter__(self) -> AsyncIterator[T]:
        async for task in self.tasks:
            yield await task

    def append(self, coro: Coroutine[Any, Any, T]) -> None:
        task = asyncio.create_task(coro)

        self.tasks.append(task)
