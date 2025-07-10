import asyncio
import functools
import inspect
from asyncio import FIRST_COMPLETED, Future
from typing import Any, ClassVar, Coroutine, Iterator, Optional, Union

import orjson
from httpx import URL
from typer import Typer

from vibe.utils.interval import Interval
from vibe.utils.time import Duration
from vibe.utils.typing import AsyncFunction, Function, JsonObject


class Shape:
    def __init__(self, *shape: str) -> None:
        self.shape = shape


class Utils:
    ENCODING: ClassVar[str] = "utf-8"

    @classmethod
    def add_typer_command(cls, *, typer: Typer, fn: Union[Function[Any, Any], AsyncFunction[Any, Any]]) -> None:
        if inspect.iscoroutinefunction(fn):
            fn = cls.to_sync_fn(fn)

        typer.command()(fn)

    # NOTE: make this an async method to ensure it's being called from an async context to ensure that
    # [asyncio.get_running_loop()] can run
    @staticmethod
    async def future[T]() -> Future[T]:
        # NOTE:
        # - prefer [asyncio.get_running_loop()] over [asyncio.get_event_loop] per
        #   [https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop]
        # - prefer [loop.create_future()] over [asyncio.Future()] per
        #   [https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop]
        return asyncio.get_running_loop().create_future()

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
    def is_none_or_empty(*, text: Optional[str]) -> bool:
        return text is None or text == ""

    @staticmethod
    def to_sync_fn[T, **P](async_fn: AsyncFunction[P, T]) -> Function[P, T]:
        @functools.wraps(async_fn)
        def fn(*args: P.args, **kwargs: P.kwargs) -> T:
            coroutine = async_fn(*args, **kwargs)
            value = asyncio.run(coroutine)

            return value

        return fn

    @staticmethod
    def url(*, url: str, query_params: JsonObject) -> str:
        url_obj = URL(url, params=query_params)
        url = str(url_obj)

        return url

    @classmethod
    def value_error(cls, **kwargs: Any) -> ValueError:
        error_str = orjson.dumps(kwargs).decode(cls.ENCODING)
        value_error = ValueError(error_str)

        return value_error

    @staticmethod
    async def wait(*coros: Coroutine[Any, Any, Any]) -> None:
        # NOTE: can't use a generator here but map() works
        tasks = map(asyncio.create_task, coros)

        await asyncio.wait(tasks, return_when=FIRST_COMPLETED)

    # NOTE: yield points: [https://tokio.rs/blog/2020-04-preemption]
    @staticmethod
    async def yield_now() -> None:
        await Duration.new(seconds=0).sleep()
