import asyncio
import dataclasses
from asyncio.subprocess import PIPE
from asyncio.subprocess import Process as StdProcess
from typing import AsyncIterator, ClassVar, Optional, Self

from vibe.utils.logger import Logger
from vibe.utils.sink import Sink

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(kw_only=True)
class Process(Sink[bytes]):
    READ_SIZE: ClassVar[int] = 2**24

    std_process: StdProcess

    @classmethod
    async def new(cls, command: str, *args: str) -> Self:
        std_process = await asyncio.create_subprocess_exec(
            command, *args, limit=cls.READ_SIZE, stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
        process = cls(std_process=std_process)

        return process

    async def __aiter__(self) -> AsyncIterator[bytes]:
        # NOTE: use while loop with [read()] rather than [__aiter__()] as [__aiter__()] reads line by line:
        # [https://github.com/python/cpython/blob/3/Lib/asyncio/streams.py]
        while True:
            byte_str = await self.std_process.stdout.read(self.READ_SIZE)

            if len(byte_str) == 0:
                break

            yield byte_str

    async def run(self, *, input_byte_str: Optional[bytes]) -> str:
        stdout_byte_str, _stderr_byte_str = await self.std_process.communicate(input=input_byte_str)
        stdout = stdout_byte_str.decode()

        return stdout

    async def asend(self, value: bytes) -> None:
        # NOTE: [self.std_process.stdin.write()] and [self.std_process.stdin.drain()] should be called together per
        # [https://docs.python.org/3/library/asyncio-stream.html#asyncio.StreamWriter.drain]
        self.std_process.stdin.write(value)

        await self.std_process.stdin.drain()

    async def aclose_stdin(self) -> None:
        # NOTE: [self.std_process.stdin.close()] and [self.std_process.stdin.wait_closed()] should be called together
        # per [https://docs.python.org/3/library/asyncio-stream.html#asyncio.StreamWriter.close]
        self.std_process.stdin.close()

        await self.std_process.stdin.wait_closed()

    async def aclose(self) -> None:
        await self.aclose_stdin()
        await self.std_process.wait()
