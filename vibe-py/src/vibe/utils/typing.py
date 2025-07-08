from typing import Any, Callable, Coroutine, Union

type JsonObject = dict[str, Any]
type Function[**P, T] = Callable[P, T]
type AsyncFunction[**P, T] = Callable[P, Coroutine[Any, Any, T]]
type AnyFunction[**P, T] = Union[Function[P, T], AsyncFunction[P, T]]
