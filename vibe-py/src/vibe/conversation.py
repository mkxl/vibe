import dataclasses
from enum import StrEnum
from typing import Optional, Self

from pydantic import BaseModel


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    function: Function
    id: str


# NOTE-e71aa5: use [= <default>] for pydantic fields bc of [ty]
class ChatMessage(BaseModel):
    role: Role
    content: Optional[str]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None


@dataclasses.dataclass(kw_only=True)
class Turn:
    role: Role
    chat_messages: list[ChatMessage]
    is_tool_call: bool

    @classmethod
    def new(cls, *, role: Role, chat_message: ChatMessage) -> Self:
        return cls(role=role, chat_messages=[chat_message], is_tool_call=chat_message.is_tool_call())

    def append_chat_message(self, chat_message: ChatMessage) -> None:
        self.chat_messages.append(chat_message)

    def compatible_with(self, *, chat_message: ChatMessage) -> bool:
        return self.role == chat_message.role and self.is_tool_call == chat_message.is_tool_call()

    def as_chat_message(self) -> ChatMessage:
        if self.is_tool_call:
            content = None
            tool_calls = [tool_call for chat_message in self.chat_messages for tool_call in chat_message.tool_calls]
        else:
            content = "".join(
                chat_message.content for chat_message in self.chat_messages if chat_message.content is not None
            )
            tool_calls = None

        return ChatMessage(role=self.role, content=content, tool_calls=tool_calls)


@dataclasses.dataclass(kw_only=True)
class Conversation:
    turns: list[Turn]

    @classmethod
    def new(cls, *, system_prompt: str) -> Self:
        conversation = cls(turns=[])

        if system_prompt != "":
            conversation.append(role=Role.SYSTEM, text=system_prompt)

        return conversation

    def _append_new_turn(self, *, chat_message: ChatMessage) -> None:
        turn = Turn.new(role=chat_message.role, chat_message=chat_message)

        self.turns.append(turn)

    def append_chat_message(self, chat_message: ChatMessage) -> None:
        if len(self.turns) == 0:
            self._append_new_turn(chat_message=chat_message)

        last_turn = self.turns[-1]

        if last_turn.compatible_with(chat_message=chat_message):
            last_turn.append_chat_message(chat_message)
        else:
            self._append_new_turn(chat_message=chat_message)

    def append(self, *, role: Role, text: str) -> None:
        chat_message = ChatMessage(role=role, content=text)

        self.append_chat_message(chat_message)

    def chat_messages(self) -> list[ChatMessage]:
        return [turn.as_chat_message() for turn in self.turns]
