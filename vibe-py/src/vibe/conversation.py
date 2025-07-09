import dataclasses
from enum import StrEnum
from typing import ClassVar, Optional, Self

from pydantic import BaseModel, ConfigDict

from vibe.utils.pydantic import FORBID_EXTRA
from vibe.utils.string_sequence import StringSequence


class Role(StrEnum):
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
    model_config: ClassVar[ConfigDict] = FORBID_EXTRA

    role: Role
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]] = None


@dataclasses.dataclass(kw_only=True)
class Turn:
    role: Role
    texts: StringSequence

    @classmethod
    def new(cls, *, role: Role) -> Self:
        return cls(role=role, texts=StringSequence.new())


@dataclasses.dataclass(kw_only=True)
class Conversation:
    turns: list[Turn]

    @classmethod
    def new(cls) -> Self:
        return cls(turns=[])

    def ensure_last(self, *, role: Role) -> Turn:
        if 0 < len(self.turns):
            return self.turns[-1]

        last_turn = Turn.new(role=role)

        self.turns.append(last_turn)

        return last_turn

    def append(self, *, role: Role, text: str) -> None:
        self.ensure_last(role=role).texts.append(text)

    def chat_messages(self) -> list[ChatMessage]:
        return [ChatMessage(role=turn.role, content=turn.texts.string()) for turn in self.turns]
