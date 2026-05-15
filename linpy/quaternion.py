from __future__ import annotations

import math
from typing import Iterator
import numpy as np
from .util import clamp, rcp, rsqrt, sincos, name_to_idx
from .vector import Vector2, Vector3, Vector4, Scalar

# Pre-computed sign sequences for each Euler rotation order
_EULER_SIGN_MAP: dict[str, Vector4] = {
    "xyz": Vector4(-1., 1., -1., 1.),
    "xzy": Vector4(1., 1., -1., -1.),
    "yxz": Vector4(-1., 1., 1., -1.),
    "yzx": Vector4(-1., -1., 1., 1.),
    "zxy": Vector4(1., -1., -1., 1.),
    "zyx": Vector4(1., -1., 1., -1.),
}

class Quaternion:
    """A quaternion representing a 3-D rotation, stored as (x, y, z, w)."""

    __slots__ = 'values'

    def __init__(self, *args: Scalar | list[Scalar] | tuple[Scalar, ...] | np.ndarray | Vector2 | Vector3 | Vector4) -> None:
        """Initialize a quaternion from scalar, list, tuple, ndarray, or vector args.

        :param args: Components to construct the quaternion from (must total 4).
        :raises AttributeError: If the number of components is not 4.
        """
        if len(args) == 1 and isinstance(args[0], (int, float)):
            self.values: Vector4 = Vector4([float(args[0])] * 4)
            return

        values: list[float] = []
        for value in args:
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, (list, tuple, Vector2, Vector3, Vector4)):
                for v in value:
                    values.append(float(v))
            elif isinstance(value, np.ndarray):
                for v in value.tolist():
                    values.append(float(v))
            else:
                raise NotImplementedError(f"Unsupported argument type: {type(value)}")

        if len(values) != 4:
            raise AttributeError(f"Quaternion requires 4 components, got {len(values)}")

        self.values = Vector4(values)

    def __repr__(self) -> str:
        """Return a string representation that can recreate the quaternion.

        :return: Repr string in the form ``Quaternion(x, y, z, w)``.
        :rtype: str
        """
        return f"Quaternion({self.x!r}, {self.y!r}, {self.z!r}, {self.w!r})"

    def __str__(self) -> str:
        """Return a human-readable string of the quaternion values.

        :return: String representation of the internal Vector4.
        :rtype: str
        """
        return str(self.values)

    def __len__(self) -> int:
        """Return the number of components (always 4).

        :return: 4
        :rtype: int
        """
        return 4

    def __iter__(self) -> Iterator[float]:
        """Iterate over the quaternion components (x, y, z, w).

        :return: Iterator of float values.
        :rtype: Iterator[float]
        """
        return iter(self.values.values)

    def __eq__(self, other: object) -> bool:
        """Test equality with another quaternion.

        :param other: The quaternion to compare.
        :return: True if all components are equal.
        :rtype: bool
        """
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.values == other.values

    def __ne__(self, other: object) -> bool:
        """Test inequality with another quaternion.

        :param other: The quaternion to compare.
        :return: True if any component differs.
        :rtype: bool
        """
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.values != other.values

    def __neg__(self) -> Quaternion:
        """Negate all components of the quaternion.

        :return: The negated quaternion.
        :rtype: Quaternion
        """
        return Quaternion(-self.values)

    def __mul__(self, other: Quaternion | Vector3 | Scalar) -> Quaternion | Vector3:
        """Multiply this quaternion by another quaternion, a Vector3, or a scalar.

        - Quaternion * Quaternion: Hamilton product (rotation composition).
        - Quaternion * Vector3: Rotate the vector by this quaternion.
        - Quaternion * scalar: Scale all components.

        :param other: The right-hand operand.
        :return: The product result.
        :raises TypeError: If *other* is an unsupported type.
        """
        if isinstance(other, Quaternion):
            return Quaternion(
                self.values.wwww * other.values
                + (self.values.xyzx * other.values.wwwx + self.values.yzxy * other.values.zxyy)
                * Vector4(1.0, 1.0, 1.0, -1.0)
                - self.values.zxyz * other.values.yzxz
            )
        elif isinstance(other, Vector3):
            t = 2 * self.xyz.cross(other)
            return other + self.w * t + self.xyz.cross(t)
        elif isinstance(other, (float, int)):
            return Quaternion(self.values * other)
        raise TypeError(f"Cannot multiply Quaternion by {type(other).__name__}")

    def __rmul__(self, other: Scalar) -> Quaternion:
        """Support scalar * quaternion multiplication.

        :param other: A scalar value.
        :return: The scaled quaternion.
        :rtype: Quaternion
        """
        if isinstance(other, (float, int)):
            return Quaternion(self.values * other)
        return NotImplemented  # type: ignore[return-value]

    def __getattr__(self, name: str) -> float | Vector2 | Vector3 | Vector4:
        """Access quaternion components via swizzle notation.

        :param name: Component name(s) (x, y, z, w or combinations).
        :return: A float or vector of the requested components.
        """
        return self.values.__getattr__(name)

    def __getitem__(self, items: int | slice) -> float | Vector2 | Vector3 | Vector4:
        """Index or slice quaternion components.

        :param items: An integer index or slice.
        :return: The component value(s).
        """
        return self.values.__getitem__(items)

    def __setattr__(self, name: str, value: object) -> None:
        """Set quaternion component values by name.

        :param name: The attribute name to set.
        :param value: The value to assign.
        """
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            self.values.__setattr__(name, value)

    def __setitem__(self, key: int | slice, newvalue: Scalar) -> None:
        """Set quaternion component value(s) by index.

        :param key: An integer index or slice.
        :param newvalue: The new scalar value to assign.
        """
        self.values.__setitem__(key, newvalue)

    @staticmethod
    def identity() -> Quaternion:
        """Return the identity quaternion (no rotation).

        :return: The identity quaternion (0, 0, 0, 1).
        :rtype: Quaternion
        """
        return Quaternion(0., 0., 0., 1.)

    @staticmethod
    def from_rotation_x(deg: float) -> Quaternion:
        """Create a quaternion representing a rotation around the X axis.

        :param deg: Rotation angle in degrees.
        :return: The rotation quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(s, 0., 0., c)

    @staticmethod
    def from_rotation_y(deg: float) -> Quaternion:
        """Create a quaternion representing a rotation around the Y axis.

        :param deg: Rotation angle in degrees.
        :return: The rotation quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(0., s, 0., c)

    @staticmethod
    def from_rotation_z(deg: float) -> Quaternion:
        """Create a quaternion representing a rotation around the Z axis.

        :param deg: Rotation angle in degrees.
        :return: The rotation quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(0., 0., s, c)

    @staticmethod
    def from_angle_axis(axis: Vector3, deg: float) -> Quaternion:
        """Create a quaternion from an axis and angle.

        :param axis: The rotation axis (will be normalized).
        :param deg: Rotation angle in degrees.
        :return: The rotation quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(axis.normalized() * s, c)

    @staticmethod
    def from_matrix3x3(matrix: np.ndarray) -> Quaternion:
        """
        Creates a quaternion from a 3x3 rotation matrix
        Uses the trace method for numerical stability
        """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            raise ValueError("Input must be a 3x3 numpy array")

        m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

        trace = m00 + m11 + m22

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            val = 1.0 + m00 - m11 - m22
            s = math.sqrt(max(0.0, val)) * 2
            if s < 1e-15:
                return Quaternion.identity()
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            val = 1.0 + m11 - m00 - m22
            s = math.sqrt(max(0.0, val)) * 2
            if s < 1e-15:
                return Quaternion.identity()
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            val = 1.0 + m22 - m00 - m11
            s = math.sqrt(max(0.0, val)) * 2
            if s < 1e-15:
                return Quaternion.identity()
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quaternion(x, y, z, w)

    @staticmethod
    def from_euler(degX: float, degY: float, degZ: float, order: str = "ZXY") -> Quaternion:
        """Create a quaternion from Euler angles.

        :param degX: Rotation around X in degrees.
        :param degY: Rotation around Y in degrees.
        :param degZ: Rotation around Z in degrees.
        :param order: Rotation order as a permutation of 'XYZ' (default 'ZXY').
        :return: The rotation quaternion.
        :rtype: Quaternion
        :raises ValueError: If *order* is not a valid permutation of XYZ.
        """
        key = order.lower()
        if key not in _EULER_SIGN_MAP:
            raise ValueError(f"Invalid Euler order: '{order}'. Must be a permutation of XYZ.")
        seq = _EULER_SIGN_MAP[key]

        h = 0.5 * Vector3(degX, degY, degZ).radians()
        s = h.sin()
        c = h.cos()
        return Quaternion(
            Vector4(s.xyz, c.x) * c.yxxy * c.zzyz
            + s.yxxy * s.zzyz * Vector4(c.xyz, s.x) * seq
        )

    def dot(self, other: Quaternion | Vector4) -> float:
        """Compute the dot product with another quaternion or Vector4.

        :param other: A Quaternion or Vector4.
        :return: The dot product.
        :rtype: float
        :raises TypeError: If *other* is not a Quaternion or Vector4.
        """
        if isinstance(other, Quaternion):
            return self.values.dot(other.values)
        elif isinstance(other, Vector4):
            return self.values.dot(other)
        raise TypeError(f"dot() expects Quaternion or Vec4, got {type(other).__name__}")

    def normalize(self) -> None:
        """Normalize this quaternion in place to unit length."""
        d = self.values.dot(self.values)
        if d < 1e-30:
            return
        self.values = rsqrt(d) * self.values

    def normalized(self) -> Quaternion:
        """Return a unit-length copy of this quaternion.

        :return: The normalized quaternion, or identity if magnitude is near zero.
        :rtype: Quaternion
        """
        d = self.values.dot(self.values)
        if d < 1e-30:
            return Quaternion.identity()
        return Quaternion(rsqrt(d) * self.values)

    def inverse(self) -> Quaternion:
        """Return the multiplicative inverse (conjugate divided by norm squared).

        :return: The inverse quaternion, or identity if magnitude is near zero.
        :rtype: Quaternion
        """
        d = self.values.dot(self.values)
        if d < 1e-30:
            return Quaternion.identity()
        return Quaternion(rcp(d) * self.values * Vector4(-1., -1., -1., 1.))

    def rotate_x(self, deg: float) -> Quaternion:
        """Compose this quaternion with a rotation around the X axis.

        :param deg: Rotation angle in degrees.
        :return: The composed quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return self * Quaternion(s, 0., 0., c)

    def rotate_y(self, deg: float) -> Quaternion:
        """Compose this quaternion with a rotation around the Y axis.

        :param deg: Rotation angle in degrees.
        :return: The composed quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., s, 0., c)

    def rotate_z(self, deg: float) -> Quaternion:
        """Compose this quaternion with a rotation around the Z axis.

        :param deg: Rotation angle in degrees.
        :return: The composed quaternion.
        :rtype: Quaternion
        """
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., 0., s, c)

    def to_matrix3x3(self) -> np.ndarray:
        """Convert this quaternion to a 3x3 rotation matrix.

        :return: A 3x3 NumPy rotation matrix.
        :rtype: numpy.ndarray
        """
        q = self.normalized()
        x, y, z, w = q.values

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array([
            [1 - 2 * (yy + zz),   2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),       1 - 2 * (xx + zz),  2 * (yz - wx)],
            [2 * (xz - wy),       2 * (yz + wx),      1 - 2 * (xx + yy)],
        ])

    def to_angle_axis(self) -> tuple[Vector3, float]:
        """Convert this quaternion to an axis-angle representation.

        :return: A tuple of (axis, angle_in_degrees).
        :rtype: tuple[Vector3, float]
        """
        q = self.normalized()
        w = max(-1.0, min(1.0, q.w))
        t = math.sqrt(1.0 - w * w)
        deg = math.degrees(math.acos(w) * 2.0)
        if t < 1e-8:
            return Vector3(1.0, 0.0, 0.0), deg
        axis = Vector3(q.x / t, q.y / t, q.z / t)
        return axis, deg

    def to_euler(self, order: str = "ZXY") -> Vector3:
        """Convert this quaternion to Euler angles.

        :param order: Rotation order as a permutation of 'XYZ' (default 'ZXY').
        :return: Euler angles in degrees as a Vector3.
        :rtype: Vector3
        :raises ValueError: If *order* is not a valid permutation of XYZ.
        """
        q = self.normalized()
        key = order.lower()
        if key not in _EULER_SIGN_MAP:
            raise ValueError(f"Invalid Euler order: '{order}'. Must be a permutation of XYZ.")

        i = name_to_idx(key[0])
        j = name_to_idx(key[1])
        k = name_to_idx(key[2])

        sign = int((i - j) * (j - k) * (k - i) / 2)
        t = q[k] * sign

        a = q.w - q[j]
        b = q[i] + t
        c = q[j] + q.w
        d = t - q[i]

        eps = 1e-7
        n2 = a ** 2 + b ** 2 + c ** 2 + d ** 2
        rj = math.acos(max(-1.0, min(1.0, 2 * (a ** 2 + b ** 2) / n2 - 1)))

        safe1 = abs(rj) >= eps
        safe2 = abs(rj - math.pi) >= eps
        safe = safe1 and safe2

        half_sum = math.atan2(b, a)
        half_diff = math.atan2(-d, c)

        if safe:
            ri = half_sum + half_diff
            rk = half_sum - half_diff
        else:
            rk = 0
            if not safe1:
                ri = 2 * half_sum
            if not safe2:
                ri = 2 * half_diff

        rk *= sign
        rj -= math.pi / 2

        euler = [0., 0., 0.]
        euler[i] = ri
        euler[j] = rj
        euler[k] = rk

        return Vector3(euler).degree()

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Spherically interpolate between this quaternion and another.

        :param other: The target quaternion.
        :param t: Interpolation factor (0 = self, 1 = other).
        :return: The interpolated quaternion.
        :rtype: Quaternion
        """
        d = clamp(self.dot(other), -1.0, 1.0)
        # If dot is negative, negate one to take the short path
        if d < 0.0:
            other = -other
            d = -d
        # Fall back to linear interpolation for nearly-identical orientations
        if d > 0.9995:
            result = Quaternion(self.values + t * (other.values - self.values))
            result.normalize()
            return result
        theta = math.acos(d)
        sin_theta = math.sin(theta)
        a = math.sin((1.0 - t) * theta) / sin_theta
        b = math.sin(t * theta) / sin_theta
        return Quaternion(a * self.values + b * other.values)

    @staticmethod
    def look_rotation(forward: Vector3, up: Vector3 | None = None) -> Quaternion:
        """Create a quaternion that orients the Z axis along *forward*.

        :param forward: The desired forward direction.
        :param up: The up reference vector (default world Y-up).
        :return: The look-rotation quaternion.
        :rtype: Quaternion
        """
        if up is None:
            up = Vector3(0.0, 1.0, 0.0)
        f = forward.normalized()
        r = up.cross(f)
        if r.dot(r) < 1e-15:
            # forward is parallel to up — pick an arbitrary perpendicular axis
            alt = Vector3(1.0, 0.0, 0.0) if abs(f.y) < 0.99 else Vector3(0.0, 0.0, 1.0)
            r = alt.cross(f).normalized()
        else:
            r = r.normalized()
        u = f.cross(r)
        m = np.array([
            [r.x, u.x, f.x],
            [r.y, u.y, f.y],
            [r.z, u.z, f.z],
        ])
        return Quaternion.from_matrix3x3(m)

    def angle_between(self, other: Quaternion) -> float:
        """Compute the angular difference in degrees between two quaternions.

        :param other: The other quaternion.
        :return: The angle in degrees.
        :rtype: float
        """
        d = clamp(self.dot(other), -1.0, 1.0)
        return math.degrees(2.0 * math.acos(abs(d)))

    def to_matrix4x4(self) -> np.ndarray:
        """Convert this quaternion to a 4x4 homogeneous rotation matrix.

        :return: A 4x4 NumPy rotation matrix with identity translation.
        :rtype: numpy.ndarray
        """
        m3 = self.to_matrix3x3()
        m4 = np.eye(4)
        m4[:3, :3] = m3
        return m4