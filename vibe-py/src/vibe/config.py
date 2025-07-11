from typing import Optional

from pydantic import BaseModel

from vibe.models import Tool


# NOTE-e71aa5
class Config(BaseModel):
    eleven_labs_voice_id: str
    eleven_labs_api_key: str
    sambanova_api_key: str
    sambanova_model: str
    system_prompt: str = ""
    tools: Optional[list[Tool]] = None
