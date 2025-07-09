import contextlib
import dataclasses
from http import HTTPMethod
from typing import AsyncIterator, ClassVar, Optional, Self

from httpx import AsyncClient
from pydantic import BaseModel, ValidationError

from vibe.utils.logger import Logger

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class Http:
    client: AsyncClient

    @classmethod
    @contextlib.asynccontextmanager
    async def context(cls) -> AsyncIterator[Self]:
        async with AsyncClient() as client:
            yield cls(client=client)


@dataclasses.dataclass(kw_only=True)
class Request:
    HEADER_NAME_CONTENT_TYPE: ClassVar[str] = "content-type"
    HEADER_NAME_AUTHORIZATION: ClassVar[str] = "authorization"
    CONTENT_TYPE_APPLICATION_JSON: ClassVar[str] = "application/json"
    SSE_LINE_PREFIX: ClassVar[str] = "data:"
    SSE_TERMINAL_LINE: ClassVar[str] = "[DONE]"
    SSE_VALIDATION_ERROR_MESSAGE: ClassVar[str] = "unable to deserialize sse line"

    http: Http
    method: HTTPMethod
    url: str
    content: Optional[str]
    headers: dict[str, str]

    @classmethod
    def new(cls, *, http: Http, method: HTTPMethod, url: str) -> Self:
        return cls(http=http, method=method, url=url, content=None, headers={})

    def content_type_application_json(self) -> Self:
        self.headers[self.HEADER_NAME_CONTENT_TYPE] = self.CONTENT_TYPE_APPLICATION_JSON

        return self

    def bearer_auth(self, *, token: str) -> Self:
        self.headers[self.HEADER_NAME_AUTHORIZATION] = f"Bearer {token}"

        return self

    def set_content(self, content: str) -> Self:
        self.content = content

        return self

    async def iter_lines(self) -> AsyncIterator[str]:
        async with self.http.client.stream(
            method=self.method, url=self.url, content=self.content, headers=self.headers
        ) as response:
            async for line in response.aiter_lines():
                yield line

    async def iter_sse[T: BaseModel](self, *, base_model_type: type[T]) -> AsyncIterator[T]:
        async for raw_line in self.iter_lines():
            line = raw_line.removeprefix(self.SSE_LINE_PREFIX).strip()

            if line == "":
                continue

            if line == self.SSE_TERMINAL_LINE:
                break

            try:
                yield base_model_type.model_validate_json(line)
            except ValidationError as validation_error:
                logger.warning(
                    message=self.SSE_VALIDATION_ERROR_MESSAGE, validation_error=validation_error, raw_line=raw_line
                )
