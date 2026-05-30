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

    cdef void c_propagate(self)
    cdef bint c_is_descendant(self, Transform transform)
    cdef void c_append_child(self, Transform transform)
    cdef void c_remove_child(self, Transform transform)

    cdef Vector3 c_world_to_local(self, Vector3 world_pos)
    cdef Vector3 c_local_to_world(self, Vector3 local_pos)
    cdef Transform c_inverse(self)
    cdef void c_rotate(self, Quaternion rotation)
    cdef void c_translate(self, Vector3 translation)
    cdef void c_add_child(self, Transform transform)
    cdef void c_set_local_pos_rot(self, Vector3 local_position, Quaternion local_rotation)
    cdef void c_set_local_pos_rot_parent(self, Vector3 local_position, Quaternion local_rotation, Transform parent)

