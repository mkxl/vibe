import dataclasses
from http import HTTPMethod
from typing import AsyncIterator, ClassVar, Optional

import pydantic
from pydantic import BaseModel, ConfigDict

from vibe.conversation import ChatMessage, Conversation
from vibe.language_model import LanguageModel
from vibe.utils.http import Http, Request
from vibe.utils.logger import Logger
from vibe.utils.pydantic import FORBID_EXTRA
from vibe.utils.typing import JsonObject

logger: Logger = Logger.new(__name__)


class SambanovaFunction(BaseModel):
    name: str
    description: str
    parameters: JsonObject


class SambanovaTool(BaseModel):
    type: str
    function: SambanovaFunction


class SambanovaRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = FORBID_EXTRA

    model: str
    messages: list[ChatMessage]
    tools: Optional[list[SambanovaTool]]

    # NOTE-f94e7d: figure out a way to have this be a class var and still be rendered
    @pydantic.computed_field
    def stream(self) -> bool:
        return True


class SambanovaResponseChunkChoice(BaseModel):
    delta: ChatMessage


class SambanovaResponseChunk(BaseModel):
    choices: list[SambanovaResponseChunkChoice]


# TODO: figure out model choice, llama won't respond unless it's given a user message relating to the supplied tools
@dataclasses.dataclass(kw_only=True)
class Sambanova(LanguageModel):
    URL: ClassVar[str] = "https://api.sambanova.ai/v1/chat/completions"

    http: Http
    api_key: str
    model: str
    tools: Optional[list[SambanovaTool]]

    def _content(self, *, conversation: Conversation) -> str:
        sambanova_request = SambanovaRequest(model=self.model, messages=conversation.chat_messages(), tools=self.tools)
        content = sambanova_request.model_dump_json()

        logger.info(sambanova_request=sambanova_request)

        return content

    # pylint: disable=invalid-overridden-method
    async def _respond(self, *, conversation: Conversation) -> AsyncIterator[str]:
        content = self._content(conversation=conversation)
        request = (
            Request.new(http=self.http, method=HTTPMethod.POST, url=self.URL)
            .bearer_auth(token=self.api_key)
            .content_type_application_json()
            .set_content(content)
        )

        async for chunk in request.iter_sse(base_model_type=SambanovaResponseChunk):
            chat_message = chunk.choices[0].delta

            logger.info(delta_chat_message=chat_message)

            if chat_message.content is not None:
                yield chat_message.content
