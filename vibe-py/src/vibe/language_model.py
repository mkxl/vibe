import asyncio
import contextlib
import dataclasses
from asyncio import Task
from typing import AsyncIterator, Protocol

from vibe.conversation import ChatMessage, Conversation, ToolCall
from vibe.utils.logger import Logger
from vibe.utils.queue import Queue

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelResponse:
    text_queue: Queue[str]
    tool_call_queue: Queue[ToolCall]
    populate_queues_task: Task[None]


class LanguageModel(Protocol):
    def _iter_chat_messages(self, *, conversation: Conversation) -> AsyncIterator[ChatMessage]:
        raise NotImplementedError

    @logger.instrument()
    async def _populate_queues(
        self, *, conversation: Conversation, text_queue: Queue[str], tool_call_queue: Queue[ToolCall]
    ) -> None:
        async with contextlib.aclosing(text_queue), contextlib.aclosing(tool_call_queue):
            async for chat_message in self._iter_chat_messages(conversation=conversation):
                logger.info(language_model_response_chat_message=chat_message)

                if chat_message.content is not None:
                    text_queue.append(chat_message.content)

                if chat_message.tool_calls is not None:
                    tool_call_queue.extend(chat_message.tool_calls)

    def respond(self, *, conversation: Conversation) -> LanguageModelResponse:
        text_queue = Queue.new()
        tool_call_queue = Queue.new()
        populate_queues_coro = self._populate_queues(
            conversation=conversation,
            text_queue=text_queue,
            tool_call_queue=tool_call_queue,
        )
        populate_queues_task = asyncio.create_task(populate_queues_coro)
        language_model_response = LanguageModelResponse(
            text_queue=text_queue, tool_call_queue=tool_call_queue, populate_queues_task=populate_queues_task
        )

        return language_model_response
