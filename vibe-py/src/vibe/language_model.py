import asyncio
import dataclasses
from asyncio import Task
from typing import AsyncIterable, AsyncIterator, Protocol, Self

from vibe.conversation import Conversation
from vibe.utils.logger import Logger
from vibe.utils.queue import Queue

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelResponse:
    queue: Queue[str]
    task: Task[None]

    @classmethod
    async def new(cls, *, text_iter: AsyncIterable[str]) -> Self:
        queue = Queue.new()
        coro = cls._coro(queue=queue, text_iter=text_iter)
        task = asyncio.create_task(coro)
        language_model_response = cls(queue=queue, task=task)

        return language_model_response

    @staticmethod
    async def _coro(*, queue: Queue[str], text_iter: AsyncIterable[str]) -> None:
        await queue.consume(text_iter)
        await queue.aclose()


class LanguageModel(Protocol):
    def _respond(self, *, conversation: Conversation) -> AsyncIterator[str]:
        raise NotImplementedError

    async def respond(self, *, conversation: Conversation) -> LanguageModelResponse:
        text_iter = self._respond(conversation=conversation)
        language_model_response = await LanguageModelResponse.new(text_iter=text_iter)

        return language_model_response
