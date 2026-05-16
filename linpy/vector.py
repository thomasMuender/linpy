from __future__ import annotations

import math
from typing import Iterator, Union, Self
import numpy as np
from .util import clamp, rsqrt, name_to_idx, has_unique_characters, highest_idx, _VALID_COMPONENTS

# Type alias for scalar values accepted in arithmetic
Scalar = Union[int, float]

class Vector:
    """Abstract base class for fixed-size numeric vectors.

    Do not instantiate directly; use :class:`Vector2`, :class:`Vector3`,
    or :class:`Vector4`.
    """
    __slots__ = 'num', 'values'

    def __init__(self, *args: Scalar | list[Scalar] | tuple[Scalar, ...] | np.ndarray | Vector) -> None:
        """Initialize a vector from scalar, list, tuple, ndarray, or other vector args.

        :param args: Components to construct the vector from.
        :raises ValueError: If instantiated directly instead of via a subclass.
        :raises AttributeError: If the wrong number of components is provided.
        """
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
        """Return a string representation that can recreate the vector.

        :return: Repr string in the form ``VectorN(x, y, ...)``.
        :rtype: str
        """
        args = ', '.join(repr(v) for v in self.values)
        return f"{type(self).__name__}({args})"

    def __str__(self) -> str:
        """Return a human-readable bracketed string of component values.

        :return: String in the form ``[x, y, ...]``.
        :rtype: str
        """
        return '[' + ', '.join(str(v) for v in self.values) + ']'
    
    def __len__(self) -> int:
        """Return the number of components.

        :return: Component count.
        :rtype: int
        """
        return self.num

    def __iter__(self) -> Iterator[float]:
        """Iterate over the component values.

        :return: Iterator of float values.
        :rtype: Iterator[float]
        """
        return self.values.__iter__()

    def __eq__(self, other: object) -> bool:
        """Test equality with another vector of the same type.

        :param other: The vector to compare.
        :return: True if all components are equal.
        :rtype: bool
        """
        if type(self) is type(other):
            return self.values == other.values  # type: ignore[union-attr]
        
        return False

    def __ne__(self, other: object) -> bool:
        """Test inequality with another vector of the same type.

        :param other: The vector to compare.
        :return: True if any component differs.
        :rtype: bool
        """
        if type(self) is type(other):
            return self.values != other.values  # type: ignore[union-attr]
        
        return False

    def __neg__(self) -> Self:
        """Negate all components.

        :return: A new vector with all components negated.
        """
        return type(self)([-v for v in self.values])

    def __add__(self, other: Self | Scalar) -> Self:
        """Add a vector or scalar to this vector component-wise.

        :param other: A vector of the same type or a scalar.
        :return: The resulting vector.
        :raises TypeError: If *other* is an incompatible type.
        """
        if type(self) is type(other):
            return type(self)([a + b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a + other for a in self.values])
        
        raise TypeError(f"Cannot add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self, other: Scalar) -> Self:
        """Support scalar + vector addition.

        :param other: A scalar value.
        :return: The resulting vector.
        """
        if isinstance(other, (float, int)):
            return self + other
        
        raise TypeError(f"Cannot add {type(other).__name__} to {type(self).__name__}")

    def __sub__(self, other: Self | Scalar) -> Self:
        """Subtract a vector or scalar from this vector component-wise.

        :param other: A vector of the same type or a scalar.
        :return: The resulting vector.
        :raises TypeError: If *other* is an incompatible type.
        """
        if type(self) is type(other):
            return type(self)([a - b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a - other for a in self.values])
        
        raise TypeError(f"Cannot subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other: Scalar) -> Self:
        """Support scalar - vector subtraction.

        :param other: A scalar value.
        :return: The resulting vector.
        """
        if isinstance(other, (float, int)):
            return type(self)([other - a for a in self.values])
        
        raise TypeError(f"Cannot subtract {type(other).__name__} from {type(self).__name__}")

    def __mul__(self, other: Self | Scalar) -> Self:
        """Multiply this vector by another vector (component-wise) or a scalar.

        :param other: A vector of the same type or a scalar.
        :return: The resulting vector.
        :raises TypeError: If *other* is an incompatible type.
        """
        if type(self) is type(other):
            return type(self)([a * b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a * other for a in self.values])
        
        raise TypeError(f"Cannot multiply {type(self).__name__} by {type(other).__name__}")

    def __rmul__(self, other: Scalar) -> Self:
        """Support scalar * vector multiplication.

        :param other: A scalar value.
        :return: The resulting vector.
        """
        if isinstance(other, (float, int)):
            return self * other
        
        raise TypeError(f"Cannot multiply {type(self).__name__} by {type(other).__name__}")

    def __truediv__(self, other: Self | Scalar) -> Self:
        """Divide this vector by another vector (component-wise) or a scalar.

        :param other: A vector of the same type or a scalar.
        :return: The resulting vector.
        :raises TypeError: If *other* is an incompatible type.
        """
        if type(self) is type(other):
            return type(self)([a / b for a, b in zip(self.values, other.values)])  # type: ignore[union-attr]
        elif isinstance(other, (float, int)):
            return type(self)([a / other for a in self.values])
        
        raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}")

    def __rtruediv__(self, other: Scalar) -> Self:
        """Support scalar / vector division.

        :param other: A scalar value.
        :return: The resulting vector.
        """
        if isinstance(other, (float, int)):
            return type(self)([other / a for a in self.values])
        
        raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}")

    def __getattr__(self, name: str) -> float | Vector2 | Vector3 | Vector4:
        """Access components by name via swizzle notation (e.g. ``v.xy``, ``v.zyx``).

        :param name: One or more component characters (x, y, z, w).
        :return: A float for single components, or a new vector for swizzles.
        :raises IndexError: If any component is out of range or swizzle is too long.
        :raises AttributeError: If *name* contains invalid characters.
        """
        if all(c in _VALID_COMPONENTS for c in name):
            indices = [name_to_idx(n) for n in name]
            if any(i >= self.num for i in indices):
                raise IndexError(f"Component '{name}' out of range for {type(self).__name__}")
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

    def __getitem__(self, items: int | slice) -> float | Vector2 | Vector3 | Vector4:
        """Index or slice the vector components.

        :param items: An integer index or slice.
        :return: A float for single index, or a new vector for slices.
        :raises ValueError: If slice produces an unsupported number of elements.
        """
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

    def __setattr__(self, name: str, value: Scalar | Vector2 | Vector3 | Vector4 | list[float] | tuple[float, ...]) -> None:
        """Set component values by name using swizzle notation.

        :param name: One or more component characters to set.
        :param value: A scalar, vector, list, or tuple of values.
        :raises IndexError: If assignment is invalid for the given name.
        """
        if name in __class__.__slots__:
            super().__setattr__(name, value)
            return
        elif (all(c in _VALID_COMPONENTS for c in name)
              and highest_idx(name) < self.num
              and has_unique_characters(name)):
            if isinstance(value, (int, float)):
                for n in name:
                    self.values[name_to_idx(n)] = float(value)
                return
            elif isinstance(value, (list, tuple, Vector2, Vector3, Vector4)) and 1 < len(name) <= self.num and len(name) == len(value):
                for i, n in enumerate(name):
                    self.values[name_to_idx(n)] = float(value[i]) # type: ignore
                return

        raise IndexError(f"Cannot set '{name}' on {type(self).__name__}")

    def __setitem__(self, key: int | slice, newvalue: Scalar) -> None:
        """Set component value(s) by index or slice.

        :param key: An integer index or slice.
        :param newvalue: The new scalar value to assign.
        """
        if isinstance(newvalue, (int, float)):
            self.values[key] = float(newvalue) # type: ignore
        else:
            self.values[key] = newvalue
    
    def dot(self, other: Self) -> float:
        """Compute the dot product with another vector.

        :param other: A vector of the same type.
        :return: The dot product.
        :rtype: float
        :raises TypeError: If *other* is not the same vector type.
        """
        if type(self) is type(other):
            return sum(a * b for a, b in zip(self.values, other.values))
        
        raise TypeError(f"dot() requires same type, got {type(other).__name__}")
    
    def magnitude(self) -> float:
        """Compute the Euclidean length of the vector.

        :return: The magnitude.
        :rtype: float
        """
        return math.sqrt(self.dot(self))

    def normalize(self) -> None:
        """Normalize this vector in place to unit length."""
        d = self.dot(self)
        if d < 1e-30:
            return
        scale = rsqrt(d)
        self.values = [a * scale for a in self.values]
    
    def normalized(self) -> Self:
        """Return a unit-length copy of this vector.

        :return: The normalized vector, or a zero vector if magnitude is near zero.
        """
        d = self.dot(self)
        if d < 1e-30:
            return type(self)(0.0)
        return rsqrt(d) * self
    
    def inverse(self) -> Self:
        """Return the additive inverse (negation) of this vector.

        :return: The negated vector.
        """
        return -self
    
    def sin(self) -> Self:
        """Apply sine to each component (in radians).

        :return: A new vector with sine of each component.
        """
        return type(self)([math.sin(v) for v in self.values])
    
    def cos(self) -> Self:
        """Apply cosine to each component (in radians).

        :return: A new vector with cosine of each component.
        """
        return type(self)([math.cos(v) for v in self.values])
    
    def degree(self) -> Self:
        """Convert each component from radians to degrees.

        :return: A new vector with values in degrees.
        """
        return type(self)([math.degrees(v) for v in self.values])
    
    def radians(self) -> Self:
        """Convert each component from degrees to radians.

        :return: A new vector with values in radians.
        """
        return type(self)([math.radians(v) for v in self.values])
    
    def lerp(self, other: Self, t: float) -> Self:
        """Linearly interpolate between this vector and another.

        :param other: The target vector.
        :param t: Interpolation factor (0 = self, 1 = other).
        :return: The interpolated vector.
        :raises TypeError: If *other* is not the same vector type.
        """
        if type(self) is type(other):
            return self + (other - self) * t
        
        raise TypeError(f"lerp() requires same type, got {type(other).__name__}")

    def distance(self, other: Self) -> float:
        """Compute the Euclidean distance to another vector.

        :param other: The other vector.
        :return: The distance.
        :rtype: float
        """
        return (self - other).magnitude()

    def distance_squared(self, other: Self) -> float:
        """Compute the squared Euclidean distance to another vector.

        :param other: The other vector.
        :return: The squared distance.
        :rtype: float
        """
        diff = self - other
        return diff.dot(diff)

    def angle_between(self, other: Self) -> float:
        """Compute the angle in degrees between this vector and another.

        :param other: The other vector.
        :return: The angle in degrees.
        :rtype: float
        :raises TypeError: If *other* is not the same vector type.
        """
        if type(self) is type(other):     
            d = self.dot(other)
            m = self.magnitude() * other.magnitude()
            if m < 1e-15:
                return 0.0
            return math.degrees(math.acos(clamp(d / m, -1.0, 1.0)))
        
        raise TypeError(f"angle_between() requires same type, got {type(other).__name__}")

    def project(self, onto: Self) -> Self:
        """Project this vector onto another vector.

        :param onto: The vector to project onto.
        :return: The projected vector.
        :raises TypeError: If *onto* is not the same vector type.
        """
        if type(self) is type(onto):
            d = onto.dot(onto)
            if d < 1e-15:
                return type(self)(0.0)
            return onto * (self.dot(onto) / d)
    
        raise TypeError(f"project() requires same type, got {type(onto).__name__}")

    def reflect(self, normal: Self) -> Self:
        """Reflect this vector about a normal.

        :param normal: The surface normal to reflect about.
        :return: The reflected vector.
        :raises TypeError: If *normal* is not the same vector type.
        """
        if type(self) is type(normal):
            return self - 2.0 * self.dot(normal) * normal
        
        raise TypeError(f"reflect() requires same type, got {type(normal).__name__}")

    def clamp_magnitude(self, max_len: float) -> Self:
        """Clamp the vector's magnitude to a maximum length.

        :param max_len: The maximum allowed magnitude.
        :return: The clamped vector.
        """
        sq = self.dot(self)
        if sq > max_len * max_len:
            return self.normalized() * max_len
        return type(self)(self.values[:])

    def abs(self) -> Self:
        """Return a vector with the absolute value of each component.

        :return: The component-wise absolute vector.
        """
        return type(self)([abs(v) for v in self.values])

    def min(self, other: Self) -> Self:
        """Return the component-wise minimum of this vector and another.

        :param other: The other vector.
        :return: A vector with the minimum of each pair of components.
        :raises TypeError: If *other* is not the same vector type.
        """
        if type(self) is type(other):
            return type(self)([min(a, b) for a, b in zip(self.values, other.values)])
        
        raise TypeError(f"min() requires same type, got {type(other).__name__}")

    def max(self, other: Self) -> Self:
        """Return the component-wise maximum of this vector and another.

        :param other: The other vector.
        :return: A vector with the maximum of each pair of components.
        :raises TypeError: If *other* is not the same vector type.
        """
        if type(self) is type(other):
            return type(self)([max(a, b) for a, b in zip(self.values, other.values)])
        
        raise TypeError(f"max() requires same type, got {type(other).__name__}")

    def floor(self) -> Self:
        """Return a vector with each component floored.

        :return: The floored vector.
        """
        return type(self)([math.floor(v) for v in self.values])

    def ceil(self):
        """Return a vector with each component ceiled.

        :return: The ceiled vector.
        """
        return type(self)([math.ceil(v) for v in self.values])

    def to_list(self) -> list[float]:
        """Convert the vector to a list of floats.

        :return: A list copy of the component values.
        :rtype: list[float]
        """
        return self.values[:]

    def to_numpy(self) -> np.ndarray:
        """Convert the vector to a NumPy array.

        :return: A 1-D NumPy array of the component values.
        :rtype: numpy.ndarray
        """
        return np.array(self.values)


class Vector2(Vector):
    """A 2-component floating-point vector."""
    __slots__ = ()

    @staticmethod
    def zero() -> Vector2:
        """Return the zero vector (0, 0).

        :return: A zero Vector2.
        :rtype: Vector2
        """
        return Vector2(0.0, 0.0)
    
    @staticmethod
    def one() -> Vector2:
        """Return the one vector (1, 1).

        :return: A unit-value Vector2.
        :rtype: Vector2
        """
        return Vector2(1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector2:
        """Return the unit X vector (1, 0).

        :return: A Vector2 pointing along the X axis.
        :rtype: Vector2
        """
        return Vector2(1.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector2:
        """Return the unit Y vector (0, 1).

        :return: A Vector2 pointing along the Y axis.
        :rtype: Vector2
        """
        return Vector2(0.0, 1.0)


class Vector3(Vector):
    """A 3-component floating-point vector."""
    __slots__ = ()

    @staticmethod
    def zero() -> Vector3:
        """Return the zero vector (0, 0, 0).

        :return: A zero Vector3.
        :rtype: Vector3
        """
        return Vector3(0.0, 0.0, 0.0)
    
    @staticmethod
    def one() -> Vector3:
        """Return the one vector (1, 1, 1).

        :return: A unit-value Vector3.
        :rtype: Vector3
        """
        return Vector3(1.0, 1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector3:
        """Return the unit X vector (1, 0, 0).

        :return: A Vector3 pointing along the X axis.
        :rtype: Vector3
        """
        return Vector3(1.0, 0.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector3:
        """Return the unit Y vector (0, 1, 0).

        :return: A Vector3 pointing along the Y axis.
        :rtype: Vector3
        """
        return Vector3(0.0, 1.0, 0.0)
    
    @staticmethod
    def z_one() -> Vector3:
        """Return the unit Z vector (0, 0, 1).

        :return: A Vector3 pointing along the Z axis.
        :rtype: Vector3
        """
        return Vector3(0.0, 0.0, 1.0)
    
    def cross(self, other: Vector3) -> Vector3:
        """Compute the cross product with another Vector3.

        :param other: The other vector.
        :return: The cross product vector.
        :rtype: Vector3
        """
        return (self * other.yzx - self.yzx * other).yzx # type: ignore


class Vector4(Vector):
    """A 4-component floating-point vector."""
    __slots__ = ()
    
    @staticmethod
    def zero() -> Vector4:
        """Return the zero vector (0, 0, 0, 0).

        :return: A zero Vector4.
        :rtype: Vector4
        """
        return Vector4(0.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def one() -> Vector4:
        """Return the one vector (1, 1, 1, 1).

        :return: A unit-value Vector4.
        :rtype: Vector4
        """
        return Vector4(1.0, 1.0, 1.0, 1.0)
    
    @staticmethod
    def x_one() -> Vector4:
        """Return the unit X vector (1, 0, 0, 0).

        :return: A Vector4 pointing along the X axis.
        :rtype: Vector4
        """
        return Vector4(1.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def y_one() -> Vector4:
        """Return the unit Y vector (0, 1, 0, 0).

        :return: A Vector4 pointing along the Y axis.
        :rtype: Vector4
        """
        return Vector4(0.0, 1.0, 0.0, 0.0)
    
    @staticmethod
    def z_one() -> Vector4:
        """Return the unit Z vector (0, 0, 1, 0).

        :return: A Vector4 pointing along the Z axis.
        :rtype: Vector4
        """
        return Vector4(0.0, 0.0, 1.0, 0.0)
    
    @staticmethod
    def w_one() -> Vector4:
        """Return the unit W vector (0, 0, 0, 1).

        :return: A Vector4 pointing along the W axis.
        :rtype: Vector4
        """
        return Vector4(0.0, 0.0, 0.0, 1.0)