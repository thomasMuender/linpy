from __future__ import annotations

import math
from typing import Iterator, Union
import numpy as np
from .util import rcp, rsqrt, sincos, has_unique_characters, name_to_idx
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
    __slots__ = 'values'

    def __init__(self, *args: Scalar | list[Scalar] | tuple[Scalar, ...] | np.ndarray | Vector2 | Vector3 | Vector4) -> None:
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
        return f"Quaternion({self.x!r}, {self.y!r}, {self.z!r}, {self.w!r})"

    def __str__(self) -> str:
        return str(self.values)

    def __len__(self) -> int:
        return 4

    def __iter__(self) -> Iterator[float]:
        return iter(self.values.values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.values == other.values

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.values != other.values

    def __neg__(self) -> Quaternion:
        return Quaternion(-self.values)

    def __mul__(self, other: Quaternion | Vector3 | Scalar) -> Quaternion | Vector3:
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
        if isinstance(other, (float, int)):
            return Quaternion(self.values * other)
        return NotImplemented  # type: ignore[return-value]

    def __getattr__(self, name: str) -> float | Vector2 | Vector3 | Vector4:
        return self.values.__getattr__(name)

    def __getitem__(self, items: int | slice) -> float | Vector2 | Vector3 | Vector4:
        return self.values.__getitem__(items)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            self.values.__setattr__(name, value)

    def __setitem__(self, key: int | slice, newvalue: Scalar) -> None:
        self.values.__setitem__(key, newvalue)

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(0., 0., 0., 1.)

    @staticmethod
    def from_rotation_x(deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return Quaternion(s, 0., 0., c)

    @staticmethod
    def from_rotation_y(deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return Quaternion(0., s, 0., c)

    @staticmethod
    def from_rotation_z(deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return Quaternion(0., 0., s, c)

    @staticmethod
    def from_angle_axis(axis: Vector3, deg: float) -> Quaternion:
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
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quaternion(x, y, z, w)

    @staticmethod
    def from_euler(degX: float, degY: float, degZ: float, order: str = "ZXY") -> Quaternion:
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
        if isinstance(other, Quaternion):
            return self.values.dot(other.values)
        elif isinstance(other, Vector4):
            return self.values.dot(other)
        raise TypeError(f"dot() expects Quaternion or Vec4, got {type(other).__name__}")

    def normalize(self) -> None:
        self.values = rsqrt(self.values.dot(self.values)) * self.values

    def normalized(self) -> Quaternion:
        return Quaternion(rsqrt(self.values.dot(self.values)) * self.values)

    def inverse(self) -> Quaternion:
        return Quaternion(rcp(self.values.dot(self.values)) * self.values * Vector4(-1., -1., -1., 1.))

    def rotate_x(self, deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return self * Quaternion(s, 0., 0., c)

    def rotate_y(self, deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., s, 0., c)

    def rotate_z(self, deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., 0., s, c)

    def to_matrix3x3(self) -> np.ndarray:
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
        q = self.normalized()
        w = max(-1.0, min(1.0, q.w))
        t = math.sqrt(1.0 - w * w)
        deg = math.degrees(math.acos(w) * 2.0)
        if t < 1e-8:
            return Vector3(1.0, 0.0, 0.0), deg
        axis = Vector3(q.x / t, q.y / t, q.z / t)
        return axis, deg

    def to_euler(self, order: str = "ZXY") -> Vector3:
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