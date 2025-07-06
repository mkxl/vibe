from typing import Any, Callable, Coroutine, ParamSpec, TypeAlias, TypeVar

# NOTE:
# - no type annotations for [T] because of [https://docs.astral.sh/ty/reference/rules/#invalid-legacy-type-variable]
T = TypeVar("T")
P: ParamSpec = ParamSpec("P")
JsonObject: TypeAlias = dict[str, Any]
Function: TypeAlias = Callable[P, T]
AsyncFunction: TypeAlias = Callable[P, Coroutine[Any, Any, T]]
