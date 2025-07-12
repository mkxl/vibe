from typing import AsyncIterator, Protocol

from vibe.utils.audio import Audio


class Tts(Protocol):
    def iter_audio(self) -> AsyncIterator[Audio]:
        raise NotImplementedError

    async def asend(self, text: str) -> None:
        raise NotImplementedError
