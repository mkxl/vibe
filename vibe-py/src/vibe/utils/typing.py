from typing import Any, Callable, Coroutine

type JsonObject = dict[str, Any]
type Function[**P, T] = Callable[P, T]
type AsyncFunction[**P, T] = Callable[P, Coroutine[Any, Any, T]]
