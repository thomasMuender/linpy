from libc cimport math
from typing import Iterator
from .vector3 cimport Vector3
from .util cimport rsqrt, to_degree, to_radians

cdef double EPSILON = 1e-9

cdef class Quaternion:
    def __cinit__(self, double x, double y, double z, double w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __init__(self, double x, double y, double z, double w):
        ...

    def __len__(self) -> int:
        return 4

    def __iter__(self) -> Iterator:
        return iter([self.x, self.y, self.z, self.w])

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}, {self.w}]"

    def __repr__(self) -> str:
        return "{\"x\": " + str(self.x) + ", \"y\": " + str(self.y) + ", \"z\": " + str(self.z) + ", \"w\": " + str(self.w) + "}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False

        cdef double dx = math.fabs(self.x - other.x)
        cdef double dy = math.fabs(self.y - other.y)
        cdef double dz = math.fabs(self.z - other.z)
        cdef double dw = math.fabs(self.w - other.w)

        return dx < EPSILON and dy < EPSILON and dz < EPSILON and dw < EPSILON

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __neg__(self) -> Quaternion:
        return Quaternion(-self.x, -self.y, -self.z, -self.w)

    def __mul__(self, other: Quaternion | Vector3) -> Quaternion | Vector3:
        if isinstance(other, Quaternion):
            return self.c_mul_quat(other)

        if isinstance(other, Vector3):
            return self.c_mul_vector(other)

        raise TypeError

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def from_euler(degX: float, degY: float, degZ: float, order: str = "ZXY") -> Quaternion:
        cdef double rx = to_radians(degX) * 0.5
        cdef double ry = to_radians(degY) * 0.5
        cdef double rz = to_radians(degZ) * 0.5

        cdef double sx = math.sin(rx)
        cdef double cx = math.cos(rx)
        cdef double sy = math.sin(ry)
        cdef double cy = math.cos(ry)
        cdef double sz = math.sin(rz)
        cdef double cz = math.cos(rz)

        cdef double qx, qy, qz, qw

        if order == "ZXY":
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz + sx * sy * cz
            qw = cx * cy * cz - sx * sy * sz
        elif order == "XYZ":
            qx = sx * cy * cz + cx * sy * sz
            qy = cx * sy * cz - sx * cy * sz
            qz = cx * cy * sz + sx * sy * cz
            qw = cx * cy * cz - sx * sy * sz
        elif order == "YXZ":
            qx = sx * cy * cz + cx * sy * sz
            qy = cx * sy * cz - sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            qw = cx * cy * cz + sx * sy * sz
        elif order == "ZYX":
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            qw = cx * cy * cz + sx * sy * sz
        elif order == "YZX":
            qx = sx * cy * cz + cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            qw = cx * cy * cz - sx * sy * sz
        elif order == "XZY":
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz - sx * cy * sz
            qz = cx * cy * sz + sx * sy * cz
            qw = cx * cy * cz + sx * sy * sz
        else:
            raise ValueError(f"Unknown rotation order: {order}")

        return Quaternion(qx, qy, qz, qw)

    cpdef double dot(self, Quaternion other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    cpdef void normalize(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return

        cdef double inv = rsqrt(mag_sq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    cpdef Quaternion normalized(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return Quaternion(self.x, self.y, self.z, self.w)

        cdef double inv = rsqrt(mag_sq)
        return Quaternion(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    cpdef Quaternion inverse(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return Quaternion(0.0, 0.0, 0.0, 0.0)

        cdef double inv = 1.0 / mag_sq
        return Quaternion(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)

    cpdef Quaternion slerp(self, Quaternion other, double t):
        cdef double d = self.dot(other)
        cdef double ox = other.x
        cdef double oy = other.y
        cdef double oz = other.z
        cdef double ow = other.w
        cdef double theta = math.acos(d)
        cdef double sin_theta = math.sin(theta)

        if d < 0.0:
            d = -d
            ox = -ox
            oy = -oy
            oz = -oz
            ow = -ow

        cdef double s0, s1
        if d > 1.0 - EPSILON:
            s0 = 1.0 - t
            s1 = t
        else:
            s0 = math.sin((1.0 - t) * theta) / sin_theta
            s1 = math.sin(t * theta) / sin_theta

        return Quaternion(
            s0 * self.x + s1 * ox,
            s0 * self.y + s1 * oy,
            s0 * self.z + s1 * oz,
            s0 * self.w + s1 * ow,
        )

    cpdef list to_list(self):
        return [self.x, self.y, self.z, self.w]

    def to_euler(self, order: str = "ZXY") -> Vector3:
        cdef double sqx = self.x * self.x
        cdef double sqy = self.y * self.y
        cdef double sqz = self.z * self.z
        cdef double sqw = self.w * self.w
        cdef double sinx = 2.0 * (self.w * self.x - self.y * self.z)
        cdef double siny = 2.0 * (self.w * self.y - self.x * self.z)

        cdef double ex, ey, ez

        if order == "ZXY":
            if math.fabs(sinx) >= 1.0:
                ex = math.copysign(90.0, sinx)
                ey = to_degree(math.atan2(2.0 * (self.x * self.z + self.w * self.y), sqw - sqx + sqz - sqy))
                ez = 0.0
            else:
                ex = to_degree(math.asin(sinx))
                ey = to_degree(math.atan2(2.0 * (self.x * self.z + self.w * self.y), 1.0 - 2.0 * (sqx + sqy)))
                ez = to_degree(math.atan2(2.0 * (self.x * self.y + self.w * self.z), 1.0 - 2.0 * (sqx + sqz)))
        else:
            if math.fabs(siny) >= 1.0:
                ex = to_degree(math.atan2(2.0 * (self.x * self.y + self.w * self.z), sqw + sqx - sqy - sqz))
                ey = math.copysign(90.0, siny)
                ez = 0.0
            else:
                ex = to_degree(math.atan2(2.0 * (self.y * self.z + self.w * self.x), 1.0 - 2.0 * (sqx + sqy)))
                ey = to_degree(math.asin(siny))
                ez = to_degree(math.atan2(2.0 * (self.x * self.y + self.w * self.z), 1.0 - 2.0 * (sqy + sqz)))

        return Vector3(ex, ey, ez)

    cdef Vector3 c_mul_vector(self, Vector3 other):
        cdef double ix = self.w * other.x + self.y * other.z - self.z * other.y
        cdef double iy = self.w * other.y + self.z * other.x - self.x * other.z
        cdef double iz = self.w * other.z + self.x * other.y - self.y * other.x
        cdef double iw = -self.x * other.x - self.y * other.y - self.z * other.z

        return Vector3(
            ix * self.w + iw * (-self.x) + iy * (-self.z) - iz * (-self.y),
            iy * self.w + iw * (-self.y) + iz * (-self.x) - ix * (-self.z),
            iz * self.w + iw * (-self.z) + ix * (-self.y) - iy * (-self.x),
        )

    cdef Quaternion c_mul_quat(self, Quaternion other):
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
            self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

