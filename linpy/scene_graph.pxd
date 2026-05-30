from .transform cimport Transform
from .vector3 cimport Vector3
from .quaternion cimport Quaternion

cdef class SceneGraph:
    cdef dict _transforms
    cdef Transform _root

    cpdef void apply_transform(self, str transform_name, str parent_name, Vector3 local_position, Quaternion local_rotation)
    cpdef void remove(self, str transform_name)
    cpdef void print_graph(self)

    cdef void c_apply_transform(self, str transform_name, str parent_name, Vector3 local_position, Quaternion local_rotation)
    cdef void c_remove(self, str transform_name)