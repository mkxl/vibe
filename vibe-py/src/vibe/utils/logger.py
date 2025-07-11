import contextlib
import dataclasses
import datetime
import functools
import inspect
import logging
import traceback
from contextvars import ContextVar
from enum import StrEnum
from logging import Formatter as StdFormatter
from logging import Logger as StdLogger
from logging import LogRecord, StreamHandler
from typing import Any, Callable, ClassVar, Iterator, Optional, Self, TextIO

from vibe.utils.typing import AnyFunction, JsonObject
from vibe.utils.utils import Utils


class Level(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @functools.cached_property
    def level(self) -> int:
        return logging.getLevelName(self.name)  # ty: ignore[invalid-return-type]


class JsonFormatter(StdFormatter):
    def __init__(self, *, include_spans: bool, populate_message: bool):
        super().__init__()

        self._include_spans = include_spans
        self._populate_message = populate_message

    def _fields(self, *, log_record: LogRecord) -> JsonObject:
        log_record_fields = getattr(log_record, "fields", {})
        fields = {}

        fields.update(Logger.CONTEXT_VAR.get())

        if not self._include_spans:
            fields.pop(Logger.SPANS_FIELD_NAME, None)

        fields.update(log_record_fields)

        return fields

    def _message(self, *, record: LogRecord, fields: JsonObject) -> Optional[str]:
        if record.msg is not None:
            return record.getMessage()

        if self._populate_message:
            return " ".join(f"{key}={value!r}" for (key, value) in fields.items())

        return None

    def format(self, record: LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat()
        process = str(record.process)
        thread = str(record.thread)
        fields = self._fields(log_record=record)
        message = self._message(record=record, fields=fields)
        json_obj = {
            "name": record.name,
            "level": record.levelname,
            "process": process,
            "thread": thread,
            "timestamp": timestamp,
            "message": message,
            "fields": fields,
        }

        # NOTE: should be an optional tuple per [https://docs.python.org/3/library/logging.html#logging.LogRecord], but
        # it's sometimes a bool
        if isinstance(record.exc_info, tuple):
            fields["traceback"] = self.formatException(record.exc_info)

        return Utils.json_dumps(json_obj)


@dataclasses.dataclass
class Logger:
    CONTEXT_VAR: ClassVar[ContextVar[JsonObject]] = ContextVar("logger", default={})  # ty: ignore[invalid-assignment]
    DEFAULT_INCLUDE_SPANS: ClassVar[bool] = False
    DEFAULT_LEVEL: ClassVar[Level] = Level.INFO
    DEFAULT_POPULATE_MESSAGE: ClassVar[bool] = False
    LOGGING_FORCE: ClassVar[bool] = True
    MIN_LEVEL: ClassVar[Level] = Level.DEBUG
    SPANS_FIELD_NAME: ClassVar[str] = "spans"

    std_logger: StdLogger

    @classmethod
    def new(cls, name: str) -> Self:
        std_logger = logging.getLogger(name)
        logger = cls(std_logger=std_logger)

        return logger

    def log(self, level: Level, message: Optional[str] = None, **fields: Any) -> None:
        self.std_logger.log(level.level, message, extra={"fields": fields})

    def debug(self, **fields: Any) -> None:
        self.log(level=Level.DEBUG, **fields)

    def info(self, **fields: Any) -> None:
        self.log(level=Level.INFO, **fields)

    def warning(self, **fields: Any) -> None:
        self.log(level=Level.WARNING, **fields)

    def error(self, **fields: Any) -> None:
        self.log(level=Level.ERROR, **fields)

    @contextlib.contextmanager
    def bookend(self, *, where: str, level: Level = DEFAULT_LEVEL) -> Iterator[None]:
        self.log(level=level, where=where, when="begin")

        yield

        self.log(level=level, where=where, when="end")

    @classmethod
    def _merge_fields(cls, *, current_fields: JsonObject, new_fields: JsonObject) -> JsonObject:
        merged_fields = current_fields | new_fields
        current_fields_spans = current_fields.get(cls.SPANS_FIELD_NAME)
        new_fields_spans = new_fields.get(cls.SPANS_FIELD_NAME)
        merged_fields[cls.SPANS_FIELD_NAME] = merged_fields_spans = []

        if current_fields_spans is not None:
            merged_fields_spans.extend(current_fields_spans)

        if new_fields_spans is not None:
            merged_fields_spans.extend(new_fields_spans)

        return merged_fields

    @classmethod
    @contextlib.contextmanager
    def context(cls, fields_dict: Optional[JsonObject] = None, **fields_kwargs: Any) -> Iterator[None]:
        new_fields = fields_kwargs if fields_dict is None else (fields_dict | fields_kwargs)
        merged_fields = cls._merge_fields(current_fields=cls.CONTEXT_VAR.get(), new_fields=new_fields)
        token = cls.CONTEXT_VAR.set(merged_fields)

        try:
            yield None
        finally:
            cls.CONTEXT_VAR.reset(token)

    @staticmethod
    def traceback(exception: Exception) -> str:
        traceback_lines = traceback.format_exception(exception)
        traceback_str = "".join(traceback_lines)

        return traceback_str

    def instrument[**P, T](self, *, level: Level = DEFAULT_LEVEL) -> Callable[[AnyFunction[P, T]], AnyFunction[P, T]]:
        def decorator(fn: AnyFunction[P, T]) -> AnyFunction[P, T]:
            if inspect.isgeneratorfunction(fn):

                def new_fn(*args: P.args, **kwargs: P.kwargs) -> T:  # ty: ignore[invalid-return-type]
                    with self.bookend(where=fn.__qualname__, level=level):
                        yield from fn(*args, **kwargs)

            elif inspect.iscoroutinefunction(fn):

                async def new_fn(*args: P.args, **kwargs: P.kwargs) -> T:
                    with self.bookend(where=fn.__qualname__, level=level):
                        return await fn(*args, **kwargs)

            elif inspect.isasyncgenfunction(fn):

                async def new_fn(*args: P.args, **kwargs: P.kwargs) -> T:  # ty: ignore[invalid-return-type]
                    with self.bookend(where=fn.__qualname__, level=level):
                        async for value in fn(*args, **kwargs):
                            yield value

            else:

                def new_fn(*args: P.args, **kwargs: P.kwargs) -> T:
                    with self.bookend(where=fn.__qualname__, level=level):
                        return fn(*args, **kwargs)

            return functools.wraps(fn)(new_fn)

        return decorator

    @classmethod
    def _add_stream_handler(
        cls, *, root_logger: StdLogger, json_formatter: JsonFormatter, stream: Optional[Any], level: Level
    ) -> None:
        stream_handler = StreamHandler(stream=stream)

        stream_handler.setFormatter(json_formatter)
        stream_handler.setLevel(level.level)
        root_logger.addHandler(stream_handler)

    @classmethod
    def init(
        cls,
        *,
        debug_file: Optional[TextIO] = None,
        include_spans: bool = DEFAULT_INCLUDE_SPANS,
        populate_message: bool = DEFAULT_POPULATE_MESSAGE,
    ) -> None:
        root_logger = logging.getLogger()
        json_formatter = JsonFormatter(include_spans=include_spans, populate_message=populate_message)

        root_logger.setLevel(cls.MIN_LEVEL.level)

        cls._add_stream_handler(
            root_logger=root_logger, json_formatter=json_formatter, stream=None, level=cls.DEFAULT_LEVEL
        )

        if debug_file is not None:
            cls._add_stream_handler(
                root_logger=root_logger, json_formatter=json_formatter, stream=debug_file, level=cls.MIN_LEVEL
            )
