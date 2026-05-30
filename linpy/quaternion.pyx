from libc cimport math
from typing import Iterator
from .vector3 cimport Vector3
from .util cimport rsqrt, to_degree, to_radians

cdef double EPSILON = 1e-9

cdef Quaternion c_from_euler(double degX, double degY, double degZ, str order):
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
        return c_from_euler(degX, degY, degZ, order)

    @staticmethod
    def from_iterable(iterable) -> Quaternion:
        try:
            iter(iterable)
        except:
            raise TypeError("Provided argument to from_iterable is not iterable")

        values = []

        try:
            for i in iterable:
                values.append(float(i))
        except:
            raise TypeError("Provided argument to from_iterable is not of scalar type (castable to float)")

        if len(values) != 4:
            raise TypeError("Provided argument does not have four components")

        return Quaternion(values[0], values[1], values[2], values[3])

    cpdef double dot(self, Quaternion other):
        return self.c_dot(other)

    cpdef void normalize(self):
        self.c_normalize()

    cpdef Quaternion normalized(self):
        return self.c_normalized()

    cpdef Quaternion inverse(self):
        return self.c_inverse()

    cpdef Quaternion slerp(self, Quaternion other, double t):
        return self.c_slerp(other, t)

    cpdef list to_list(self):
        return [self.x, self.y, self.z, self.w]

    def to_euler(self, order: str = "ZXY") -> Vector3:
        return self.c_to_euler(order)

    cdef Vector3 c_mul_vector(self, Vector3 other):
        cdef double tx = 2.0 * (self.y * other.z - self.z * other.y)
        cdef double ty = 2.0 * (self.z * other.x - self.x * other.z)
        cdef double tz = 2.0 * (self.x * other.y - self.y * other.x)

        return Vector3(
            other.x + self.w * tx + self.y * tz - self.z * ty,
            other.y + self.w * ty + self.z * tx - self.x * tz,
            other.z + self.w * tz + self.x * ty - self.y * tx,
        )

    cdef Quaternion c_mul_quat(self, Quaternion other):
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
            self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    cdef double c_dot(self, Quaternion other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    cdef void c_normalize(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return

        cdef double inv = rsqrt(mag_sq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    cdef Quaternion c_normalized(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return Quaternion(self.x, self.y, self.z, self.w)

        cdef double inv = rsqrt(mag_sq)
        return Quaternion(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    cdef Quaternion c_inverse(self):
        cdef double mag_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

        if mag_sq < EPSILON:
            return Quaternion(0.0, 0.0, 0.0, 0.0)

        cdef double inv = 1.0 / mag_sq
        return Quaternion(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)

    cdef Quaternion c_slerp(self, Quaternion other, double t):
        cdef double d = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
        cdef double ox = other.x
        cdef double oy = other.y
        cdef double oz = other.z
        cdef double ow = other.w

        if d < 0.0:
            d = -d
            ox = -ox
            oy = -oy
            oz = -oz
            ow = -ow

        cdef double theta, sin_theta
        cdef double s0, s1
        if d > 1.0 - EPSILON:
            s0 = 1.0 - t
            s1 = t
        else:
            theta = math.acos(d)
            sin_theta = math.sin(theta)
            s0 = math.sin((1.0 - t) * theta) / sin_theta
            s1 = math.sin(t * theta) / sin_theta

        return Quaternion(
            s0 * self.x + s1 * ox,
            s0 * self.y + s1 * oy,
            s0 * self.z + s1 * oz,
            s0 * self.w + s1 * ow,
        )

    cdef Vector3 c_to_euler(self, str order):
        cdef double xx = self.x * self.x
        cdef double yy = self.y * self.y
        cdef double zz = self.z * self.z
        cdef double xy = self.x * self.y
        cdef double xz = self.x * self.z
        cdef double xw = self.x * self.w
        cdef double yz = self.y * self.z
        cdef double yw = self.y * self.w
        cdef double zw = self.z * self.w

        cdef double ex, ey, ez, test

        if order == "ZXY":
            # X = asin(m32), m32 = 2(yz+xw)
            test = 2.0 * (yz + xw)
            if test > 1.0 - EPSILON:
                ex = 90.0
                ey = 0.0
                ez = to_degree(math.atan2(2.0 * (xy + zw), 1.0 - 2.0 * (yy + zz)))
            elif test < -1.0 + EPSILON:
                ex = -90.0
                ey = 0.0
                ez = to_degree(math.atan2(2.0 * (xy + zw), 1.0 - 2.0 * (yy + zz)))
            else:
                ex = to_degree(math.asin(test))
                ey = to_degree(math.atan2(2.0 * (yw - xz), 1.0 - 2.0 * (xx + yy)))
                ez = to_degree(math.atan2(2.0 * (zw - xy), 1.0 - 2.0 * (xx + zz)))
        elif order == "XYZ":
            # Y = asin(m13), m13 = 2(xz+yw)
            test = 2.0 * (xz + yw)
            if test > 1.0 - EPSILON:
                ey = 90.0
                ex = to_degree(math.atan2(2.0 * (yz + xw), 1.0 - 2.0 * (xx + zz)))
                ez = 0.0
            elif test < -1.0 + EPSILON:
                ey = -90.0
                ex = to_degree(math.atan2(2.0 * (yz + xw), 1.0 - 2.0 * (xx + zz)))
                ez = 0.0
            else:
                ey = to_degree(math.asin(test))
                ex = to_degree(math.atan2(2.0 * (xw - yz), 1.0 - 2.0 * (xx + yy)))
                ez = to_degree(math.atan2(2.0 * (zw - xy), 1.0 - 2.0 * (yy + zz)))
        elif order == "YXZ":
            # X = asin(-m23), -m23 = 2(xw-yz)
            test = 2.0 * (xw - yz)
            if test > 1.0 - EPSILON:
                ex = 90.0
                ey = to_degree(math.atan2(2.0 * (yw - xz), 1.0 - 2.0 * (yy + zz)))
                ez = 0.0
            elif test < -1.0 + EPSILON:
                ex = -90.0
                ey = to_degree(math.atan2(2.0 * (yw - xz), 1.0 - 2.0 * (yy + zz)))
                ez = 0.0
            else:
                ex = to_degree(math.asin(test))
                ey = to_degree(math.atan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy)))
                ez = to_degree(math.atan2(2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz)))
        elif order == "ZYX":
            # Y = asin(-m31), -m31 = 2(yw-xz)
            test = 2.0 * (yw - xz)
            if test > 1.0 - EPSILON:
                ey = 90.0
                ex = 0.0
                ez = to_degree(math.atan2(2.0 * (zw - xy), 1.0 - 2.0 * (xx + zz)))
            elif test < -1.0 + EPSILON:
                ey = -90.0
                ex = 0.0
                ez = to_degree(math.atan2(2.0 * (zw - xy), 1.0 - 2.0 * (xx + zz)))
            else:
                ey = to_degree(math.asin(test))
                ex = to_degree(math.atan2(2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)))
                ez = to_degree(math.atan2(2.0 * (xy + zw), 1.0 - 2.0 * (yy + zz)))
        elif order == "YZX":
            # Z = asin(m21), m21 = 2(xy+zw)
            test = 2.0 * (xy + zw)
            if test > 1.0 - EPSILON:
                ez = 90.0
                ex = 0.0
                ey = to_degree(math.atan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy)))
            elif test < -1.0 + EPSILON:
                ez = -90.0
                ex = 0.0
                ey = to_degree(math.atan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy)))
            else:
                ez = to_degree(math.asin(test))
                ex = to_degree(math.atan2(2.0 * (xw - yz), 1.0 - 2.0 * (xx + zz)))
                ey = to_degree(math.atan2(2.0 * (yw - xz), 1.0 - 2.0 * (yy + zz)))
        elif order == "XZY":
            # Z = asin(-m12), -m12 = 2(zw-xy)
            test = 2.0 * (zw - xy)
            if test > 1.0 - EPSILON:
                ez = 90.0
                ex = to_degree(math.atan2(2.0 * (xw - yz), 1.0 - 2.0 * (xx + yy)))
                ey = 0.0
            elif test < -1.0 + EPSILON:
                ez = -90.0
                ex = to_degree(math.atan2(2.0 * (xw - yz), 1.0 - 2.0 * (xx + yy)))
                ey = 0.0
            else:
                ez = to_degree(math.asin(test))
                ex = to_degree(math.atan2(2.0 * (yz + xw), 1.0 - 2.0 * (xx + zz)))
                ey = to_degree(math.atan2(2.0 * (xz + yw), 1.0 - 2.0 * (yy + zz)))
        else:
            ex = 0.0
            ey = 0.0
            ez = 0.0

        return Vector3(ex, ey, ez)