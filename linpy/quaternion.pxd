from .vector3 cimport Vector3

cdef double EPSILON

cdef class Quaternion:
    cdef readonly double x, y, z, w

    cpdef double dot(self, Quaternion other)
    cpdef void normalize(self)
    cpdef Quaternion normalized(self)
    cpdef Quaternion inverse(self)
    cpdef Quaternion slerp(self, Quaternion other, double t)
    cpdef list to_list(self)

    cdef Vector3 c_mul_vector(self, Vector3 other)
    cdef Quaternion c_mul_quat(self, Quaternion other)