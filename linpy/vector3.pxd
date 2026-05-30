cdef double EPSILON

cdef class Vector3:
    cdef readonly double x, y, z

    cpdef double dot(self, Vector3 other)
    cpdef double magnitude(self)
    cpdef void normalize(self)
    cpdef Vector3 normalized(self)
    cpdef Vector3 cross(self, Vector3 other)
    cpdef Vector3 inverse(self)
    cpdef Vector3 lerp(self, Vector3 other, double t)
    cpdef double distance(self, Vector3 other)
    cpdef list to_list(self)

    cdef Vector3 c_add_vector(self, Vector3 other)
    cdef Vector3 c_add_double(self, double other)

    cdef Vector3 c_sub_vector(self, Vector3 other)
    cdef Vector3 c_sub_double(self, double other)

    cdef Vector3 c_mul_vector(self, Vector3 other)
    cdef Vector3 c_mul_double(self, double other)

    cdef Vector3 c_div_vector(self, Vector3 other)
    cdef Vector3 c_div_double(self, double other)

    cdef double c_dot(self, Vector3 other)
    cdef double c_magnitude(self)
    cdef void c_normalize(self)
    cdef Vector3 c_normalized(self)
    cdef Vector3 c_cross(self, Vector3 other)
    cdef Vector3 c_lerp(self, Vector3 other, double t)
    cdef double c_distance(self, Vector3 other)

