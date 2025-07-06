import asyncio
import functools
import inspect
from asyncio import FIRST_COMPLETED, Future
from typing import Any, Coroutine, Iterator, Union

from typer import Typer

from vibe.utils.interval import Interval
from vibe.utils.typing import AsyncFunction, Function, P, T


class Utils:
    @classmethod
    def add_typer_command(cls, *, typer: Typer, fn: Union[Function[P, T], AsyncFunction[P, T]]) -> None:
        if inspect.iscoroutinefunction(fn):
            fn = cls.to_sync_fn(fn)

        typer.command()(fn)

    @staticmethod
    def iter_intervals(*, total: int, chunk_size: int) -> Iterator[Interval[int]]:
        endpoint_range = range(0, total, chunk_size)
        endpoint_iter = iter(endpoint_range)
        begin = next(endpoint_iter, None)

        if begin is None:
            return

        for endpoint in endpoint_iter:
            yield Interval[int](begin=begin, end=endpoint)

            begin = endpoint

        if begin != total:
            yield Interval[int](begin=begin, end=total)

    @staticmethod
    async def pending() -> None:
        await Future()

    @staticmethod
    def to_sync_fn(async_fn: AsyncFunction[P, T]) -> Function[P, T]:
        @functools.wraps(async_fn)
        def fn(*args: P.args, **kwds: P.kwargs) -> T:
            coroutine = async_fn(*args, **kwds)
            value = asyncio.run(coroutine)

            return value

        return fn

    @staticmethod
    async def wait(*coros: Coroutine[Any, Any, T]) -> None:
        # NOTE: can't use a generator here but map() works
        tasks = map(asyncio.create_task, coros)

        await asyncio.wait(tasks, return_when=FIRST_COMPLETED)

    # NOTE: yield points: [https://tokio.rs/blog/2020-04-preemption]
    @staticmethod
    async def yield_now() -> None:
        await asyncio.sleep(0)
