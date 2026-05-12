from __future__ import annotations

import math
import numpy as np
from util import rcp, rsqrt, sincos, has_unique_characters, name_to_idx
from vector import Vec2, Vec3, Vec4

class Quaternion:
    __slots__ = 'values'

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (int, float)):
            self.values = Vec4([float(args[0])] * 4)
            return

        values = []
        for value in args:
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, (list, tuple, Vec2, Vec3, Vec4)):
                for v in value:
                    values.append(float(v))
            elif isinstance(value, np.ndarray):
                l = value.tolist()
                for v in l:
                    values.append(float(v))
            else:
                raise NotImplementedError
            
        if len(values) != 4:
            raise AttributeError
        
        self.values = Vec4(values)

    def __str__(self) -> str:
        return str(self.values)
    
    def __len__(self):
        return 4

    def __iter__(self):
        return iter(self.values.values)

    def __mul__(self, other) -> Quaternion | Vec3:
        if isinstance(other, Quaternion):
            return Quaternion(self.values.wwww * other.values + (self.values.xyzx * other.values.wwwx + self.values.yzxy * other.values.zxyy) * Vec4(1.0, 1.0, 1.0, -1.0) - self.values.zxyz * other.values.yzxz)
        elif isinstance(other, Vec3):
            t = 2 * self.xyz.cross(other)
            return other + self.w * t + self.xyz.cross(t)
        elif isinstance(other, (float, int)):
            return Quaternion(self.values * other)
        raise TypeError

    def __getattr__(self, name):
        return self.values.__getattr__(name)

    def __getitem__(self, items):
        return self.values.__getitem__(items)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            self.values.__setattr__(name, value)

    def __setitem__(self, key, newvalue):
        self.values.__setitem__(key, newvalue)

    @property
    def identity(self) -> Quaternion:
        return Quaternion(0., 0., 0., 1.)
    
    @staticmethod
    def fromRotationX(deg: float) -> Quaternion:
        """
        Creates a rotation around the x-axis from degrees
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(s, 0., 0., c)
    
    @staticmethod
    def fromRotationY(deg: float) -> Quaternion:
        """
        Creates a rotation around the y-axis from degrees
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(0., s, 0., c)
    
    @staticmethod
    def fromRotationZ(deg: float) -> Quaternion:
        """
        Creates a rotation around the z-axis from degrees
        """
        s, c = sincos(0.5 * deg)
        return Quaternion(0., 0., s, c)
    
    @staticmethod
    def fromAngleAxis(axis: Vec3, deg: float) -> Quaternion:
        s, c = sincos(0.5 * deg)
        return Quaternion(axis.normalize() * s, c)
    
    @staticmethod
    def fromMatrix3x3(matrix) -> Quaternion:
        """
        Creates a quaternion from a 3x3 rotation matrix
        Uses the trace method for numerical stability
        """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            raise ValueError("Input must be a 3x3 numpy array")
        
        # Extract matrix elements
        m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
        
        # Calculate trace
        trace = m00 + m11 + m22
        
        if trace > 0:
            # Case 1: trace > 0
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            # Case 2: m00 is the largest diagonal element
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2  # s = 4 * qx
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            # Case 3: m11 is the largest diagonal element
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2  # s = 4 * qy
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            # Case 4: m22 is the largest diagonal element
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2  # s = 4 * qz
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        
        return Quaternion(x, y, z, w)
    
    @staticmethod
    def fromEuler(degX: float, degY: float, degZ: float, order="ZXY") -> Quaternion:
        order = order.lower()
        assert len(order) == 3 and all(c in ('x', 'y', 'z') for c in order) and has_unique_characters(order)

        seq = None
        if order == "xyz":
            seq = Vec4(-1., 1., -1., 1.)
        elif order == "xzy":
            seq = Vec4(1., 1., -1., -1.)
        elif order == "yxz":
            seq = Vec4(-1., 1., 1., -1.)
        elif order == "yzx":
            seq = Vec4(-1., -1., 1., 1.)
        elif order == "zxy":
            seq = Vec4(1., -1., -1., 1.)
        elif order == "zyx":
            seq = Vec4(1., -1., 1., -1.)

        assert seq != None

        h = 0.5 * Vec3(degX, degY, degZ).radians()
        s = h.sin()
        c = h.cos()
        return Quaternion(Vec4(s.xyz, c.x) * c.yxxy * c.zzyz + s.yxxy * s.zzyz * Vec4(c.xyz, s.x) * seq)

    def dot(self, other):
        if isinstance(other, Quaternion):
            return self.values.dot(other.values)
        elif isinstance(other, Vec4):
            return self.values.dot(other)
        raise ValueError
    
    def normalize(self) -> None:
        self.values = rsqrt(self.values.dot(self.values)) * self.values
    
    def normalized(self) -> Quaternion:
        return Quaternion(rsqrt(self.values.dot(self.values)) * self.values)
    
    def inverse(self) -> Quaternion:
        return Quaternion(rcp(self.values.dot(self.values)) * self.values * Vec4(-1., -1., -1., 1.))
    
    def rotateX(self, deg: float):
        s, c = sincos(0.5 * deg)
        return self * Quaternion(s, 0., 0., c)

    def rotateY(self, deg: float):
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., s, 0., c)
    
    def rotateZ(self, deg: float):
        s, c = sincos(0.5 * deg)
        return self * Quaternion(0., 0., s, c)

    def toMatrix3x3(self):
        self.normalize()
        x, y, z, w = self.values

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        rot_matrix = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
        ])

        return rot_matrix
    
    def toAngleAxis(self) -> tuple[Vec3, float]:
        self.normalize()
        t = math.sqrt(1.0 - self.w * self.w)
        axis = Vec3(self.x / t, self.y / t, self.z / t)
        deg = math.degrees(math.acos(self.w) * 2.0)
        return axis, deg
    
    def toEuler(self, order = "ZXY") -> Vec3:
        self.normalize()
        order = order.lower()
        assert len(order) == 3 and all(c in ('x', 'y', 'z') for c in order) and has_unique_characters(order)

        i = name_to_idx(order[0])
        j = name_to_idx(order[1])
        k = name_to_idx(order[2])

        sign = int((i-j)*(j-k)*(k-i)/2)
        t = self[k] * sign

        a = self.w - self[j]
        b = self[i] + t
        c = self[j] + self.w
        d = t - self[i]

        eps = 1e-7
        n2 = a**2 + b**2 + c**2 + d**2
        rj = math.acos(2*(a**2 + b**2) / n2 - 1)

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

        return Vec3(euler).degree()