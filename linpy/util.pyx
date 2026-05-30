from libc cimport math

cdef double RAD_TO_DEG = 180.0 / math.M_PI
cdef double DEG_TO_RAD = math.M_PI / 180.0

cdef double rsqrt(double x):
    return 1.0 / math.sqrt(x)

cdef double to_degree(double x):
    return x * RAD_TO_DEG

cdef double to_radians(double x):
    return x * DEG_TO_RAD

cdef double clamp(double value, double min_value, double max_value):
    return min(max(value, min_value), max_value)
