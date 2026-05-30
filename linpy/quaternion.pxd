from .vector3 cimport Vector3

cdef double EPSILON

cdef Quaternion c_from_euler(double degX, double degY, double degZ, str order)

cdef class Quaternion:
    cdef readonly double x, y, z, w

    cpdef double dot(self, Quaternion other)
    cpdef void normalize(self)
    cpdef Quaternion normalized(self)
    cpdef Quaternion inverse(self)
    cpdef Quaternion slerp(self, Quaternion other, double t)
    cpdef list to_list(self)
    cpdef Quaternion rotate_x(self, double degrees)
    cpdef Quaternion rotate_y(self, double degrees)
    cpdef Quaternion rotate_z(self, double degrees)

    cdef Vector3 c_mul_vector(self, Vector3 other)
    cdef Quaternion c_mul_quat(self, Quaternion other)

    cdef double c_dot(self, Quaternion other)
    cdef void c_normalize(self)
    cdef Quaternion c_normalized(self)
    cdef Quaternion c_inverse(self)
    cdef Quaternion c_slerp(self, Quaternion other, double t)
    cdef Vector3 c_to_euler(self, str order)
    