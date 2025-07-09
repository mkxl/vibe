import dataclasses
from enum import StrEnum
from typing import Optional, Self

from pydantic import BaseModel

from vibe.utils.string_sequence import StringSequence


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


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
    tool_calls: Optional[list[ToolCall]] = None


@dataclasses.dataclass(kw_only=True)
class Turn:
    role: Role
    texts: StringSequence

    @classmethod
    def new(cls, *, role: Role, text: str) -> Self:
        turn = cls(role=role, texts=StringSequence.new())

        turn.texts.append(text)

        return turn


@dataclasses.dataclass(kw_only=True)
class Conversation:
    turns: list[Turn]

    @classmethod
    def new(cls, *, system_prompt: str) -> Self:
        conversation = cls(turns=[])

        if system_prompt != "":
            conversation.append(role=Role.SYSTEM, text=system_prompt)

        return conversation

    def append(self, *, role: Role, text: str) -> None:
        match self.turns:
            case [*_, last_turn] if last_turn.role == role:
                last_turn.texts.append(text)
            case _:
                last_turn = Turn.new(role=role, text=text)

                self.turns.append(last_turn)

    def chat_messages(self) -> list[ChatMessage]:
        return [ChatMessage(role=turn.role, content=turn.texts.string()) for turn in self.turns]
