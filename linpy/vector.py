from __future__ import annotations

import math
from typing import Iterator, Union
import numpy as np
from .util import rsqrt, name_to_idx, has_unique_characters, highest_idx, _VALID_COMPONENTS

# Type alias for scalar values accepted in arithmetic
Scalar = Union[int, float]


class Vector:
    __slots__ = 'num', 'values'

    def __init__(self, *args: Scalar | list[Scalar] | tuple[Scalar, ...] | np.ndarray | Vector) -> None:
        if type(self) not in (Vector2, Vector3, Vector4):
            raise ValueError("Cannot instantiate Vector directly; use Vector2, Vector3, or Vector4")

        self.num: int = int(type(self).__name__[-1])
        self.values: list[float] = []
        
        if len(args) == 1 and isinstance(args[0], (int, float)):
            self.values = [float(args[0])] * self.num
            return

        for value in args:
            if isinstance(value, (int, float)):
                self.values.append(float(value))
            elif isinstance(value, (list, tuple, Vector2, Vector3, Vector4)):
                for v in value:
                    self.values.append(float(v))
            elif isinstance(value, np.ndarray):
                for v in value.tolist():
                    self.values.append(float(v))
            else:
                raise NotImplementedError(f"Unsupported argument type: {type(value)}")
            
        if len(self.values) != self.num:
            raise AttributeError(
                f"{type(self).__name__} requires {self.num} components, got {len(self.values)}"
            )
        
    def __repr__(self) -> str:
        args = ', '.join(repr(v) for v in self.values)
        return f"{type(self).__name__}({args})"

    def __str__(self) -> str:
        return '[' + ', '.join(str(v) for v in self.values) + ']'
    
    def __len__(self) -> int:
        return self.num

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.values == other.values  # type: ignore[union-attr]

    def __ne__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.values != other.values  # type: ignore[union-attr]

    def __neg__(self) -> Vector:
        return type(self)([-v for v in self.values])

    def __add__(self, other: Vector | Scalar) -> Vector:
        if type(self) is type(other):
            return type(self)([a + b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a + other for a in self.values])
        raise TypeError(f"Cannot add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self, other: Scalar) -> Vector:
        if isinstance(other, (float, int)):
            return self + other
        return NotImplemented  # type: ignore[return-value]

    def __sub__(self, other: Vector | Scalar) -> Vector:
        if type(self) is type(other):
            return type(self)([a - b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a - other for a in self.values])
        raise TypeError(f"Cannot subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other: Scalar) -> Vector:
        if isinstance(other, (float, int)):
            return type(self)([other - a for a in self.values])
        return NotImplemented  # type: ignore[return-value]

    def __mul__(self, other: Vector | Scalar) -> Vector:
        if type(self) is type(other):
            return type(self)([a * b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a * other for a in self.values])
        raise TypeError(f"Cannot multiply {type(self).__name__} by {type(other).__name__}")

    def __rmul__(self, other: Scalar) -> Vector:
        if isinstance(other, (float, int)):
            return self * other
        return NotImplemented  # type: ignore[return-value]

    def __truediv__(self, other: Vector | Scalar) -> Vector:
        if type(self) is type(other):
            return type(self)([a / b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a / other for a in self.values])
        raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}")

    def __rtruediv__(self, other: Scalar) -> Vector:
        if isinstance(other, (float, int)):
            return type(self)([other / a for a in self.values])
        return NotImplemented  # type: ignore[return-value]

    def __getattr__(self, name: str) -> float | Vector:
        if all(c in _VALID_COMPONENTS for c in name):
            indices = [name_to_idx(n) for n in name]
            if any(i >= self.num for i in indices):
                raise IndexError(
                    f"Component '{name}' out of range for {type(self).__name__}"
                )
            if len(name) == 1:
                return self.values[indices[0]]
            vals = [self.values[i] for i in indices]
            if len(name) == 2:
                return Vector2(vals)
            elif len(name) == 3:
                return Vector3(vals)
            elif len(name) == 4:
                return Vector4(vals)
            raise IndexError(f"Swizzle too long: '{name}'")
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, items: int | slice) -> float | Vector:
        sliced = self.values[items]
        if isinstance(sliced, (int, float)):
            return sliced
        if len(sliced) == 2:
            return Vector2(sliced)
        elif len(sliced) == 3:
            return Vector3(sliced)
        elif len(sliced) == 4:
            return Vector4(sliced)
        raise ValueError(f"Slice produced {len(sliced)} elements; expected 2, 3, or 4")

    def __setattr__(self, name: str, value: Scalar | Vector | list[float] | tuple[float, ...]) -> None:
        done = False
        if name in __class__.__slots__:
            super().__setattr__(name, value)
            done = True
        elif (all(c in _VALID_COMPONENTS for c in name)
              and highest_idx(name) < self.num
              and has_unique_characters(name)):
            if isinstance(value, (int, float)):
                for n in name:
                    self.values[name_to_idx(n)] = float(value)
                done = True
            elif isinstance(value, (list, tuple, Vector2, Vector3, Vector4)) and 1 < len(name) <= self.num and len(name) == len(value):
                for i, n in enumerate(name):
                    self.values[name_to_idx(n)] = float(value[i])
                done = True
        if not done:
            raise IndexError(f"Cannot set '{name}' on {type(self).__name__}")

    def __setitem__(self, key: int | slice, newvalue: Scalar) -> None:
        if isinstance(newvalue, (int, float)):
            self.values[key] = float(newvalue)
        else:
            self.values[key] = newvalue
    
    def dot(self, other: Vector) -> float:
        if type(self) is not type(other):
            raise TypeError(f"dot() requires same type, got {type(other).__name__}")
        return sum(a * b for a, b in zip(self.values, other.values))
    
    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> None:
        scale = rsqrt(self.dot(self))
        self.values = [a * scale for a in self.values]
    
    def normalized(self) -> Vector:
        return rsqrt(self.dot(self)) * self
    
    def inverse(self) -> Vector:
        return -self
    
    def sin(self) -> Vector:
        return type(self)([math.sin(v) for v in self.values])
    
    def cos(self) -> Vector:
        return type(self)([math.cos(v) for v in self.values])
    
    def degree(self) -> Vector:
        return type(self)([math.degrees(v) for v in self.values])
    
    def radians(self) -> Vector:
        return type(self)([math.radians(v) for v in self.values])
    
    def to_list(self) -> list[float]:
        return self.values[:]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values)


class Vector2(Vector):
    __slots__ = ()

    @staticmethod
    def zero() -> Vector2:
        return Vector2(0.0, 0.0)
    
    @staticmethod
    def one() -> Vector2:
        return Vector2(1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector2:
        return Vector2(1.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector2:
        return Vector2(0.0, 1.0)


class Vector3(Vector):
    __slots__ = ()

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0.0, 0.0, 0.0)
    
    @staticmethod
    def one() -> Vector3:
        return Vector3(1.0, 1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector3:
        return Vector3(1.0, 0.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector3:
        return Vector3(0.0, 1.0, 0.0)
    
    @staticmethod
    def z_one() -> Vector3:
        return Vector3(0.0, 0.0, 1.0)
    
    def cross(self, other: Vector3) -> Vector3:
        return (self * other.yzx - self.yzx * other).yzx


class Vector4(Vector):
    __slots__ = ()
    
    @staticmethod
    def zero() -> Vector4:
        return Vector4(0.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def one() -> Vector4:
        return Vector4(1.0, 1.0, 1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector4:
        return Vector4(1.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector4:
        return Vector4(0.0, 1.0, 0.0, 0.0)
    
    @staticmethod
    def z_one() -> Vector4:
        return Vector4(0.0, 0.0, 1.0, 0.0)
    
    @staticmethod
    def w_one() -> Vector4:
        return Vector4(0.0, 0.0, 0.0, 1.0)