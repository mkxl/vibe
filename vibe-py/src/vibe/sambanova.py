import dataclasses
from http import HTTPMethod
from typing import AsyncIterator, ClassVar, Optional

import pydantic
from pydantic import BaseModel

from vibe.conversation import Conversation
from vibe.language_model import LanguageModel
from vibe.models import ChatMessage, Tool
from vibe.utils.http import Http, Request
from vibe.utils.logger import Level, Logger

logger: Logger = Logger.new(__name__)


class SambanovaRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: Optional[list[Tool]]

    # NOTE-f94e7d: figure out a way to have this be a class var and still be serialized
    @pydantic.computed_field
    def stream(self) -> bool:
        return True


class SambanovaResponseChunkChoice(BaseModel):
    delta: ChatMessage


class SambanovaResponseChunk(BaseModel):
    choices: list[SambanovaResponseChunkChoice]


@dataclasses.dataclass(kw_only=True)
class Sambanova(LanguageModel):
    EXCLUDE_NONE_ON_MODEL_DUMP: ClassVar[bool] = True
    URL: ClassVar[str] = "https://api.sambanova.ai/v1/chat/completions"

    http: Http
    api_key: str
    model: str
    tools: Optional[list[Tool]]

    def _content(self, *, conversation: Conversation) -> str:
        sambanova_request = SambanovaRequest(model=self.model, messages=conversation.chat_messages(), tools=self.tools)
        content = sambanova_request.model_dump_json(exclude_none=self.EXCLUDE_NONE_ON_MODEL_DUMP)

        logger.debug(sambanova_request=sambanova_request)

        return content

    # pylint: disable=invalid-overridden-method
    @logger.instrument(level=Level.DEBUG)
    async def _iter_response_chat_messages(self, *, conversation: Conversation) -> AsyncIterator[ChatMessage]:
        content = self._content(conversation=conversation)
        request = (
            Request.new(http=self.http, method=HTTPMethod.POST, url=self.URL)
            .bearer_auth(token=self.api_key)
            .content_type_application_json()
            .set_content(content)
        )

        async for chunk in request.iter_sse(base_model_type=SambanovaResponseChunk):
            yield chunk.choices[0].delta
