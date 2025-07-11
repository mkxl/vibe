from enum import StrEnum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from vibe.utils.typing import JsonObject
from vibe.utils.utils import Utils

type Implementation = Annotated[Union["ConstantImplementation", "ProcessImplementation"], Field(discriminator="type")]


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    function: FunctionCall
    id: str


# NOTE-e71aa5: use [= <default>] for pydantic fields bc of [ty]
class ChatMessage(BaseModel):
    role: Role
    content: Optional[str]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None


class ImplementationType(StrEnum):
    CONSTANT = "constant"
    PROCESS = "process"


class ConstantImplementation(BaseModel):
    type: Literal[ImplementationType.CONSTANT]
    value: str


# NOTE-e71aa5
class ProcessImplementation(BaseModel):
    type: Literal[ImplementationType.PROCESS]
    wait: bool
    command: str
    args: list[str] = []
    input_field_name: Optional[str] = None

    def input_byte_str(self, *, tool_call: ToolCall) -> Optional[bytes]:
        if self.input_field_name is None:
            return None

        input_str = Utils.json_loads(tool_call.function.arguments)[self.input_field_name]
        input_byte_str = None if input_str is None else input_str.encode(Utils.ENCODING)

        return input_byte_str


# TODO: just flat out excluding [implementation] so it's not included in requests to language model providers is probs
# not the right approach, but it's the one we're taking for now
# NOTE-e71aa5
class FunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: JsonObject = {}
    implementation: Annotated[Optional[Implementation], Field(exclude=True)] = None


class Tool(BaseModel):
    type: str
    function: FunctionDeclaration
