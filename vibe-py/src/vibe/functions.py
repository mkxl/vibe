from enum import StrEnum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from vibe.utils.typing import JsonObject

type Implementation = Annotated[Union["ConstantImplementation", "ProcessImplementation"], Field(discriminator="type")]


class ImplementationType(StrEnum):
    CONSTANT = "constant"
    PROCESS = "process"


class ConstantImplementation(BaseModel):
    type: Literal[ImplementationType.CONSTANT]
    value: str


# NOTE-e71aa5
class ProcessImplementation(BaseModel):
    type: Literal[ImplementationType.PROCESS]
    command: str
    args: list[str] = []


# TODO: just flat out excluding [implementation] so it's not included in requests to language model providers is probs
# not the right approach, but it's the one we're taking for now
# NOTE-e71aa5
class Function(BaseModel):
    name: str
    description: str
    parameters: JsonObject = {}
    implementation: Annotated[Optional[Implementation], Field(exclude=True)] = None


class Tool(BaseModel):
    type: str
    function: Function
