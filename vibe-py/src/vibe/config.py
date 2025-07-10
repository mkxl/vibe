from typing import Optional

from pydantic import BaseModel

from vibe.sambanova import SambanovaTool


# NOTE-e71aa5
class Config(BaseModel):
    eleven_labs_voice_id: str
    eleven_labs_api_key: str
    sambanova_api_key: str
    sambanova_model: str
    sambanova_tools: Optional[list[SambanovaTool]] = None
    system_prompt: str = ""
