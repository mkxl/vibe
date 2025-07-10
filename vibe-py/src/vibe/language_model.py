import dataclasses
from asyncio import Task
from typing import AsyncIterator, Protocol

from vibe.conversation import ChatMessage, Conversation
from vibe.utils.logger import Logger
from vibe.utils.queue import Queue
from vibe.utils.utils import Utils

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelResponse:
    chat_message_queue: Queue[ChatMessage]
    task: Task[None]


class LanguageModel(Protocol):
    def _iter_response_chat_messages(self, *, conversation: Conversation) -> AsyncIterator[ChatMessage]:
        raise NotImplementedError

    def respond(self, *, conversation: Conversation) -> LanguageModelResponse:
        chat_message_queue = Queue.new()
        chat_message_iter = self._iter_response_chat_messages(conversation=conversation)
        task = Utils.create_task(chat_message_queue.aconsume, chat_message_iter)
        language_model_response = LanguageModelResponse(
            chat_message_queue=chat_message_queue,
            task=task,
        )

        return language_model_response
