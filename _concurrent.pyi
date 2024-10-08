# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

from typing import Any, Optional

class ConcurrentDict:
    def __init__(self, initial_capacity: Optional[int] = 8) -> None: ...
    def __contains__(self, key: Any) -> bool: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __getitem__(self, key: Any) -> None: ...

class AtomicInt64:
    def __init__(self, value: int = 0) -> None: ...
    def set(self, value: int) -> None: ...
    def get(self) -> int: ...
    def incr(self) -> int: ...
    def decr(self) -> int: ...
    def __format__(self, format_spec: str) -> str: ...
    def __add__(self, other: int) -> int: ...
    def __sub__(self, other: int) -> int: ...
    def __mul__(self, other: int) -> int: ...
    def __floordiv__(self, other: int) -> int: ...
    def __neg__(self) -> int: ...
    def __pos__(self) -> int: ...
    def __abs__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __or__(self, other: int) -> int: ...  # type: ignore
    def __xor__(self, other: int) -> int: ...
    def __and__(self, other: int) -> int: ...
    def __invert__(self) -> int: ...
    def __iadd__(self, other: int) -> "AtomicInt64": ...
    def __isub__(self, other: int) -> "AtomicInt64": ...
    def __imul__(self, other: int) -> "AtomicInt64": ...
    def __ifloordiv__(self, other: int) -> "AtomicInt64": ...
    def __ior__(self, other: int) -> "AtomicInt64": ...
    def __ixor__(self, other: int) -> "AtomicInt64": ...
    def __iand__(self, other: int) -> "AtomicInt64": ...
    def __int__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...

class AtomicReference:
    def __init__(self, value: Optional[Any]) -> None: ...
    def set(self, value: Any) -> None: ...
    def get(self) -> Any: ...
    def exchange(self, value: Any) -> Any: ...
    def compare_exchange(self, expected: Any, value: Any) -> bool: ...
