from typing import Any, Callable, Coroutine, Union

type JsonObject = dict[str, Any]
type Function[**P1, T] = Callable[P1, T]
type AsyncFunction[**P1, T] = Callable[P1, Coroutine[Any, Any, T]]
type AnyFunction[**P, T] = Union[Function[P, T], AsyncFunction[P, T]]
