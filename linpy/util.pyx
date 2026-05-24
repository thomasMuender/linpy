from libc cimport math

cdef double rsqrt(double x):
    return 1.0 / math.sqrt(x)

cdef double to_degree(double x):
    return x * 57.2958

cdef double to_radians(double x):
    return x * 0.0174533

cdef double clamp(double value, double min_value, double max_value):
    return max(min(value, min_value), max_value)
