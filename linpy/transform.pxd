from .vector3 cimport Vector3
from .quaternion cimport Quaternion

cdef class Transform:
    cdef str _name
    cdef Vector3 _local_position
    cdef Quaternion _local_rotation
    cdef Vector3 _position
    cdef Quaternion _rotation
    cdef Transform _parent
    cdef list _children

    cpdef Vector3 world_to_local(self, Vector3 world_pos)
    cpdef Vector3 local_to_world(self, Vector3 local_pos)
    cpdef Transform inverse(self)
    cpdef void rotate(self, Quaternion rotation)
    cpdef void translate(self, Vector3 translation)
    cpdef void add_child(self, Transform transform)
    cpdef void set_local_pos_rot(self, Vector3 local_position, Quaternion local_rotation)
    cpdef void set_local_pos_rot_parent(self, Vector3 local_position, Quaternion local_rotation, Transform parent)

    cdef void _propagate(self)
    cdef bint _is_descendant(self, Transform transform)
    cdef void _add_child(self, Transform transform)
    cdef void _remove_child(self, Transform transform)
