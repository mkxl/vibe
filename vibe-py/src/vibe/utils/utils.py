import asyncio
import contextlib
import functools
import inspect
from asyncio import FIRST_COMPLETED, Future, Task
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, ClassVar, Coroutine, Iterator, Optional, TypeGuard, Union

import orjson
import yaml
from httpx import URL
from pydantic import BaseModel
from typer import Typer

from vibe.utils.interval import Interval
from vibe.utils.time import Duration
from vibe.utils.typing import AsyncFunction, Function, JsonObject


class Shape:
    def __init__(self, *shape: str) -> None:
        self.shape = shape


class Utils:
    DOCUMENTS_DIRNAME: ClassVar[str] = "docs"
    ENCODING: ClassVar[str] = "utf-8"
    PYDANTIC_BASE_MODEL_DUMP_MODE: ClassVar[str] = "json"

    @classmethod
    def add_typer_command(cls, *, typer: Typer, fn: Union[Function[Any, Any], AsyncFunction[Any, Any]]) -> None:
        if inspect.iscoroutinefunction(fn):
            fn = cls.to_sync_fn(fn)

        typer.command()(fn)

    @classmethod
    @contextlib.contextmanager
    def context_map[T, S: AbstractContextManager](
        cls, *, value: Optional[S], fn: Optional[Callable[[S], T]]
    ) -> Iterator[Optional[Union[S, T]]]:
        if value is None:
            yield None
        elif fn is None:
            yield value
        else:
            yield fn(value)

    @staticmethod
    def create_task[**P, T](fn: AsyncFunction[P, T], *args: P.args, **kwargs: P.kwargs) -> Task[T]:
        coro = fn(*args, **kwargs)
        task = asyncio.create_task(coro)

        return task

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
    def is_not_none_and_is_nonempty(text: Optional[str]) -> TypeGuard[str]:
        return text is not None and text != ""

    @classmethod
    def _json_dumps_default(cls, value: Any) -> Union[str, JsonObject]:
        if isinstance(value, BaseModel):
            return value.model_dump(mode=cls.PYDANTIC_BASE_MODEL_DUMP_MODE)

        return str(value)

    # NOTE-17964d: use orjson because it's faster [https://github.com/ijl/orjson?tab=readme-ov-file#serialize]
    @classmethod
    def json_dumps(cls, json_obj: Optional[JsonObject] = None, **kwargs: Any) -> str:
        json_obj = kwargs if json_obj is None else (json_obj | kwargs)
        json_str = orjson.dumps(json_obj, default=cls._json_dumps_default).decode(cls.ENCODING)

        return json_str

    # NOTE-17964d
    @classmethod
    def json_loads(cls, json_str: str) -> Any:
        return orjson.loads(json_str)

    @staticmethod
    def model_validate_yaml[S: BaseModel](yaml_str: str, *, base_model_type: type[S]) -> S:
        json_obj = yaml.safe_load(yaml_str)
        base_model = base_model_type.model_validate(json_obj)

        return base_model

    @classmethod
    def read_document(cls, filename: str) -> str:
        return Path(__file__).parent.parent.joinpath(cls.DOCUMENTS_DIRNAME, filename).read_text(encoding=cls.ENCODING)

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
        error_str = cls.json_dumps(kwargs)
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
