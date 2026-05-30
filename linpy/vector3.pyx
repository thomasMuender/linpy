# vector3.pyx — Cython implementation of Vector3
from libc cimport math
from .util cimport rsqrt

cdef double EPSILON = 1e-9

cdef class Vector3:
    def __cinit__(self, double x, double y, double z):
        self.x = x
        self.y = y
        self.z = z

    def __init__(self, double x, double y, double z):
        ...

    def __len__(self):
        return 3
    
    def __iter__(self):
        return iter([self.x, self.y, self.z])
    
    def __str__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"

    def __repr__(self):
        return "{\"x\": " + str(self.x) + ", \"y\": " + str(self.y) + ", \"z\": " + str(self.z) + "}"
    
    def __eq__(self, other):
        if not isinstance(other, Vector3):
            return False
        
        cdef double dx = math.fabs(self.x - other.x)
        cdef double dy = math.fabs(self.y - other.y)
        cdef double dz = math.fabs(self.z - other.z)

        return dx < EPSILON and dy < EPSILON and dz < EPSILON

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)
    
    def __add__(self, other) -> Vector3:
        if isinstance(other, Vector3):
            return self.c_add_vector(other)
        
        if isinstance(other, (float, int)):
            return self.c_add_double(other)
        
        raise TypeError

    def __sub__(self, other) -> Vector3:
        if isinstance(other, Vector3):
            return self.c_sub_vector(other)
        
        if isinstance(other, (float, int)):
            return self.c_sub_double(other)
        
        raise TypeError

    def __mul__(self, other) -> Vector3:
        if isinstance(other, Vector3):
            return self.c_mul_vector(other)
        
        if isinstance(other, (float, int)):
            return self.c_mul_double(other)
        
        raise TypeError

    def __truediv__(self, other) -> Vector3:
        if isinstance(other, Vector3):
            return self.c_div_vector(other)
        
        if isinstance(other, (float, int)):
            return self.c_div_double(other)
        
        raise TypeError
    
    def __rmul__(self, other) -> Vector3:
        if isinstance(other, (float, int)):
            return self.c_mul_double(other)
        
        raise TypeError


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

    @staticmethod
    def from_iterable(iterable) -> Vector3:
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

        if len(values) != 3:
            raise TypeError("Provided argument does not have three components")

        return Vector3(values[0], values[1], values[2])

    @property
    def xxx(self) -> Vector3:
        return Vector3(self.x, self.x, self.x)

    @property
    def xxy(self) -> Vector3:
        return Vector3(self.x, self.x, self.y)

    @property
    def xxz(self) -> Vector3:
        return Vector3(self.x, self.x, self.z)

    @property
    def xyx(self) -> Vector3:
        return Vector3(self.x, self.y, self.x)

    @property
    def xyy(self) -> Vector3:
        return Vector3(self.x, self.y, self.y)

    @property
    def xyz(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @property
    def xzx(self) -> Vector3:
        return Vector3(self.x, self.z, self.x)

    @property
    def xzy(self) -> Vector3:
        return Vector3(self.x, self.z, self.y)

    @property
    def xzz(self) -> Vector3:
        return Vector3(self.x, self.z, self.z)

    @property
    def yxx(self) -> Vector3:
        return Vector3(self.y, self.x, self.x)

    @property
    def yxy(self) -> Vector3:
        return Vector3(self.y, self.x, self.y)

    @property
    def yxz(self) -> Vector3:
        return Vector3(self.y, self.x, self.z)

    @property
    def yyx(self) -> Vector3:
        return Vector3(self.y, self.y, self.x)

    @property
    def yyy(self) -> Vector3:
        return Vector3(self.y, self.y, self.y)

    @property
    def yyz(self) -> Vector3:
        return Vector3(self.y, self.y, self.z)

    @property
    def yzx(self) -> Vector3:
        return Vector3(self.y, self.z, self.x)

    @property
    def yzy(self) -> Vector3:
        return Vector3(self.y, self.z, self.y)

    @property
    def yzz(self) -> Vector3:
        return Vector3(self.y, self.z, self.z)

    @property
    def zxx(self) -> Vector3:
        return Vector3(self.z, self.x, self.x)

    @property
    def zxy(self) -> Vector3:
        return Vector3(self.z, self.x, self.y)

    @property
    def zxz(self) -> Vector3:
        return Vector3(self.z, self.x, self.z)

    @property
    def zyx(self) -> Vector3:
        return Vector3(self.z, self.y, self.x)

    @property
    def zyy(self) -> Vector3:
        return Vector3(self.z, self.y, self.y)

    @property
    def zyz(self) -> Vector3:
        return Vector3(self.z, self.y, self.z)

    @property
    def zzx(self) -> Vector3:
        return Vector3(self.z, self.z, self.x)

    @property
    def zzy(self) -> Vector3:
        return Vector3(self.z, self.z, self.y)

    @property
    def zzz(self) -> Vector3:
        return Vector3(self.z, self.z, self.z)

    cpdef double dot(self, Vector3 other):
        return self.c_dot(other)

    cpdef double magnitude(self):
        return self.c_magnitude()
    
    cpdef void normalize(self):
        self.c_normalize()

    cpdef Vector3 normalized(self):
        return self.c_normalized()

    cpdef Vector3 cross(self, Vector3 other):
        return self.c_cross(other)

    cpdef Vector3 inverse(self):
        return Vector3(-self.x, -self.y, -self.z)

    cpdef Vector3 lerp(self, Vector3 other, double t):
        return self.c_lerp(other, t)

    cpdef double distance(self, Vector3 other):
        return self.c_distance(other)

    cpdef list to_list(self):
        return [self.x, self.y, self.z]


    cdef Vector3 c_add_vector(self, Vector3 other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef Vector3 c_add_double(self, double other):
        return Vector3(self.x + other, self.y + other, self.z + other)

    cdef Vector3 c_sub_vector(self, Vector3 other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    cdef Vector3 c_sub_double(self, double other):
        return Vector3(self.x - other, self.y - other, self.z - other)

    cdef Vector3 c_mul_vector(self, Vector3 other):
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)

    cdef Vector3 c_mul_double(self, double other):
        return Vector3(self.x * other, self.y * other, self.z * other)

    cdef Vector3 c_div_vector(self, Vector3 other):
        return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)

    cdef Vector3 c_div_double(self, double other):
        cdef double inv = 1.0 / other
        return Vector3(self.x * inv, self.y * inv, self.z * inv)


    cdef double c_dot(self, Vector3 other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    cdef double c_magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    cdef void c_normalize(self):
        cdef double d = self.x * self.x + self.y * self.y + self.z * self.z

        if d < EPSILON:
            return

        cdef double scale = rsqrt(d)
        self.x *= scale
        self.y *= scale
        self.z *= scale

    cdef Vector3 c_normalized(self):
        cdef double d = self.x * self.x + self.y * self.y + self.z * self.z
        if d < EPSILON:
            return Vector3(0.0, 0.0, 0.0)

        cdef double scale = rsqrt(d)
        return Vector3(self.x * scale, self.y * scale, self.z * scale)

    cdef Vector3 c_cross(self, Vector3 other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    cdef Vector3 c_lerp(self, Vector3 other, double t):
        return Vector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )

    cdef double c_distance(self, Vector3 other):
        cdef double dx = self.x - other.x
        cdef double dy = self.y - other.y
        cdef double dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)