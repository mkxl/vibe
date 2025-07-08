import asyncio
import functools
import inspect
from asyncio import FIRST_COMPLETED, Future
from typing import Any, Coroutine, Iterator, Union

import orjson
from typer import Typer

from vibe.utils.constants import ENCODING
from vibe.utils.interval import Interval
from vibe.utils.typing import AsyncFunction, Function


class Shape:
    def __init__(self, *shape: str) -> None:
        self.shape = shape


class Utils:
    @classmethod
    def add_typer_command[T, **P](cls, *, typer: Typer, fn: Union[Function[P, T], AsyncFunction[P, T]]) -> None:
        if inspect.iscoroutinefunction(fn):
            fn = cls.to_sync_fn(fn)

        typer.command()(fn)

    @classmethod
    def invalid_result(cls, invalid_result: Any) -> ValueError:
        return cls.value_error(invalid_result=invalid_result)

    @staticmethod
    def iter_intervals(*, begin: int, total: int, chunk_size: int, exact: bool) -> Iterator[Interval[int]]:
        endpoint_range = range(begin, total, chunk_size)
        endpoint_iter = iter(endpoint_range)
        interval_begin = next(endpoint_iter, None)

        if interval_begin is None:
            return

        for endpoint in endpoint_iter:
            yield Interval[int](begin=interval_begin, end=endpoint)

            interval_begin = endpoint

        if not exact and interval_begin != total:
            yield Interval[int](begin=interval_begin, end=total)

    @staticmethod
    async def pending() -> None:
        await Future()

    @staticmethod
    def to_sync_fn[T, **P](async_fn: AsyncFunction[P, T]) -> Function[P, T]:
        @functools.wraps(async_fn)
        def fn(*args: P.args, **kwargs: P.kwargs) -> T:
            coroutine = async_fn(*args, **kwargs)
            value = asyncio.run(coroutine)

            return value

        return fn

    @staticmethod
    def value_error(**kwargs: Any) -> ValueError:
        error_str = orjson.dumps(kwargs).decode(ENCODING)
        value_error = ValueError(error_str)

        return value_error

    @staticmethod
    async def wait[T](*coros: Coroutine[Any, Any, T]) -> None:
        # NOTE: can't use a generator here but map() works
        tasks = map(asyncio.create_task, coros)

        await asyncio.wait(tasks, return_when=FIRST_COMPLETED)

    # NOTE: yield points: [https://tokio.rs/blog/2020-04-preemption]
    @staticmethod
    async def yield_now() -> None:
        await asyncio.sleep(0)
