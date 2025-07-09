from typing import Optional

from pydantic import BaseModel

from vibe.sambanova import SambanovaTool


# NOTE-e71aa5
class Config(BaseModel):
    system_prompt: str = ""
    sambanova_api_key: str
    sambanova_model: str
    sambanova_tools: Optional[list[SambanovaTool]] = None
