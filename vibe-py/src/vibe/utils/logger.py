import contextlib
import dataclasses
import datetime
import functools
import logging
import traceback
from contextvars import ContextVar
from enum import StrEnum
from logging import Formatter as StdFormatter
from logging import Logger as StdLogger
from logging import LogRecord, StreamHandler
from typing import Any, ClassVar, Iterator, Optional, Self, Union

import orjson

from vibe.utils.constants import ENCODING
from vibe.utils.typing import JsonObject


class Level(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @functools.cached_property
    def level(self) -> int:
        return logging.getLevelName(self.name)  # ty: ignore[invalid-return-type]


@dataclasses.dataclass
class Logger:
    CONTEXT_VAR: ClassVar[ContextVar[JsonObject]] = ContextVar("logger", default={})  # ty: ignore[invalid-assignment]
    DEFAULT_INCLUDE_SPANS: ClassVar[bool] = False
    DEFAULT_LEVEL: ClassVar[Level] = Level.INFO
    FORCE: ClassVar[bool] = True
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
    def bookend(self, message: str, *, level: Level = DEFAULT_LEVEL) -> Iterator[None]:
        self.log(level=level, message=message, bookend="begin")

        yield

        self.log(level=level, message=message, bookend="end")

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
    def context(cls, fields_dict: Optional[JsonObject] = None, **fields_kwds: Any) -> Iterator[None]:
        new_fields = fields_kwds if fields_dict is None else (fields_dict | fields_kwds)
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

    @classmethod
    def init(
        cls, *, stream: Optional[Any] = None, level: Level = DEFAULT_LEVEL, include_spans: bool = DEFAULT_INCLUDE_SPANS
    ) -> None:
        stream_handler = StreamHandler(stream=stream)
        json_formatter = JsonFormatter(include_spans=include_spans)

        stream_handler.setFormatter(json_formatter)
        logging.basicConfig(handlers=[stream_handler], level=level.level, force=cls.FORCE)


class JsonFormatter(StdFormatter):
    def __init__(self, *, include_spans: bool):
        super().__init__()

        self._include_spans = include_spans

    def _fields(self, *, log_record: LogRecord) -> JsonObject:
        log_record_fields = getattr(log_record, "fields", {})
        fields = {}

        fields.update(Logger.CONTEXT_VAR.get())

        if not self._include_spans:
            fields.pop(Logger.SPANS_FIELD_NAME, None)

        fields.update(log_record_fields)

        return fields

    @staticmethod
    def _message(*, fields: JsonObject) -> str:
        return " ".join(f"{key}={value!r}" for (key, value) in fields.items())

    @staticmethod
    def _default(o: Any) -> Union[str, JsonObject]:
        # TODO: decide what i'm doing here
        # if isinstance(o, BaseModel):
        #     return json.loads(o.model_dump_json())

        return str(o)

    def format(self, record: LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat()
        process = str(record.process)
        thread = str(record.thread)
        fields = self._fields(log_record=record)
        message = self._message(fields=fields) if record.msg is None else record.msg
        json_object = {
            "name": record.name,
            "level": record.levelname,
            "process": process,
            "thread": thread,
            "timestamp": timestamp,
            "message": message,
            "fields": fields,
        }

        if record.exc_info is not None:
            fields["traceback"] = self.formatException(record.exc_info)

        # NOTE: use orjson because it's faster [https://github.com/ijl/orjson?tab=readme-ov-file#serialize]
        return orjson.dumps(json_object, default=self._default).decode(ENCODING)
