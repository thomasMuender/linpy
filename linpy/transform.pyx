# transform.pyx — Cython implementation of Transform
from .vector3 cimport Vector3
from .quaternion cimport Quaternion

cdef class Transform:
    """A transform node with local and world position/rotation in a hierarchy."""

    def __cinit__(self, Vector3 pos, Quaternion rot, str name = ""):
        self._name = name
        self._local_position = pos
        self._local_rotation = rot
        self._position = pos
        self._rotation = rot
        self._parent = None
        self._children = []

    def __init__(self, Vector3 pos, Quaternion rot, str name = ""):
        ...

    @property
    def name(self) -> str:
        """The name of this transform."""
        return self._name

    @property
    def local_position(self) -> Vector3:
        """The local position relative to the parent."""
        return self._local_position

    @local_position.setter
    def local_position(self, Vector3 value):
        """Set the local position and propagate world transforms."""
        if self._local_position != value:
            self._local_position = value
            self.c_propagate()

    @property
    def local_rotation(self) -> Quaternion:
        """The local rotation relative to the parent."""
        return self._local_rotation

    @local_rotation.setter
    def local_rotation(self, Quaternion value):
        """Set the local rotation and propagate world transforms."""
        if self._local_rotation != value:
            self._local_rotation = value
            self.c_propagate()

    @property
    def position(self) -> Vector3:
        """The world-space position."""
        return self._position

    @position.setter
    def position(self, Vector3 value):
        """Set the world-space position, updating local position accordingly."""
        if self._position != value:
            if self._parent is None:
                self._local_position = value
            else:
                self._local_position = self._parent._rotation.c_inverse().c_mul_vector(
                    value.c_sub_vector(self._parent._position)
                )
            self.c_propagate()

    @property
    def rotation(self) -> Quaternion:
        """The world-space rotation."""
        return self._rotation

    @rotation.setter
    def rotation(self, Quaternion value):
        """Set the world-space rotation, updating local rotation accordingly."""
        if self._rotation != value:
            if self._parent is None:
                self._local_rotation = value
            else:
                self._local_rotation = self._parent._rotation.c_inverse().c_mul_quat(value)
            self.c_propagate()

    @property
    def parent(self) -> Transform:
        """The parent transform, or None if this is a root."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """Set the parent transform, reparenting this node in the hierarchy."""
        if value is self._parent:
            return

        if value is self:
            raise ValueError("A transform cannot be its own parent")

        if isinstance(value, Transform) and self.c_is_descendant(<Transform>value):
            raise ValueError("Cannot set a descendant as parent")

        # Detach from the current parent
        if self._parent is not None:
            self._parent.c_remove_child(self)

        # Attach to the new parent, or promote to world root
        if isinstance(value, Transform):
            self._parent = <Transform>value
            self._parent.c_append_child(self)
        else:
            self._parent = None

        # Recompute world transform and propagate to children
        self.c_propagate()

    @property
    def x_dir(self) -> Vector3:
        """The local X axis direction in world space."""
        return self._rotation.c_mul_vector(Vector3(1.0, 0.0, 0.0))

    @property
    def y_dir(self) -> Vector3:
        """The local Y axis direction in world space."""
        return self._rotation.c_mul_vector(Vector3(0.0, 1.0, 0.0))

    @property
    def z_dir(self) -> Vector3:
        """The local Z axis direction in world space."""
        return self._rotation.c_mul_vector(Vector3(0.0, 0.0, 1.0))

    def __repr__(self) -> str:
        return f"Transform({self._name!r}, {self._position!r}, {self._rotation!r})"

    def __str__(self) -> str:
        return self._name + ": [Pos: " + str(self._position) + ", Rot: " + str(self._rotation.c_to_euler("ZXY")) + "]"

    def __mul__(self, other):
        """Compose this transform with another object.

        Transform * Transform -> composed transform
        Transform * Vector3  -> point transformed into world space
        """
        cdef Transform t
        cdef Vector3 v

        if isinstance(other, Transform):
            t = <Transform>other
            return Transform(
                self._rotation.c_mul_vector(t._position).c_add_vector(self._position),
                self._rotation.c_mul_quat(t._rotation),
                self._name + "*" + t._name,
            )
        elif isinstance(other, Vector3):
            v = <Vector3>other
            return self._rotation.c_mul_vector(v).c_add_vector(self._position)

        raise TypeError(f"Cannot multiply Transform by {type(other).__name__}")

    def __len__(self) -> int:
        """Return the number of child transforms."""
        return len(self._children)

    def __iter__(self):
        """Iterate over child transforms."""
        return iter(self._children)

    def __getitem__(self, items):
        """Get child transform(s) by index or slice."""
        return self._children[items]

    def __setitem__(self, int key, Transform newvalue):
        """Set a child transform at the given index."""
        self._children[key] = newvalue
        (<Transform>self._children[key]).parent = self

    cpdef Transform inverse(self):
        return self.c_inverse()

    cpdef void set_local_pos_rot(self, Vector3 local_position, Quaternion local_rotation):
        self.c_set_local_pos_rot(local_position, local_rotation)

    cpdef void set_local_pos_rot_parent(self, Vector3 local_position, Quaternion local_rotation, Transform parent):
        self.c_set_local_pos_rot_parent(local_position, local_rotation, parent)

    cpdef Vector3 world_to_local(self, Vector3 world_pos):
        """Transform a world-space position into this transform's local space."""
        return self.c_world_to_local(world_pos)

    cpdef Vector3 local_to_world(self, Vector3 local_pos):
        """Transform a local-space position into world space."""
        return self.c_local_to_world(local_pos)

    cpdef void rotate(self, Quaternion rotation):
        """Apply an additional rotation to this transform."""
        self.c_rotate(rotation)

    cpdef void translate(self, Vector3 translation):
        """Translate this transform in its local space."""
        self.c_translate(translation)

    cpdef void add_child(self, Transform transform):
        """Add a child transform to this node."""
        self.c_add_child(transform)

        

    cdef void c_append_child(self, Transform transform):
        self._children.append(transform)

    cdef void c_remove_child(self, Transform transform):
        if transform in self._children:
            self._children.remove(transform)

    cdef bint c_is_descendant(self, Transform transform):
        cdef Transform child
        for child in self._children:
            if child is transform or child.c_is_descendant(transform):
                return True
        return False

    cdef void c_propagate(self):
        """Recalculate this transform's world position/rotation from its locals,
        then recursively propagate to all children."""
        cdef list stack = [self]
        cdef Transform transform
        cdef Transform child

        while stack:
            transform = <Transform>stack.pop()

            if transform._parent is None:
                transform._position = transform._local_position
                transform._rotation = transform._local_rotation
            else:
                transform._position = transform._parent._rotation.c_mul_vector(
                    transform._local_position
                ).c_add_vector(transform._parent._position)
                transform._rotation = transform._parent._rotation.c_mul_quat(
                    transform._local_rotation
                )

            for child in transform._children:
                stack.append(child)

    cdef Transform c_inverse(self):
        """Compute the inverse of this transform."""
        cdef Quaternion inv_rot = self._rotation.c_inverse()
        cdef Vector3 inv_pos = inv_rot.c_mul_vector(Vector3(-self._position.x, -self._position.y, -self._position.z))
        return Transform(inv_pos, inv_rot, self._name + "_inv")

    cdef void c_set_local_pos_rot(self, Vector3 local_position, Quaternion local_rotation):
        """Set local position and rotation in one call, propagating once."""
        if self._local_position != local_position or self._local_rotation != local_rotation:
            self._local_position = local_position
            self._local_rotation = local_rotation
            self.c_propagate()

    cdef void c_set_local_pos_rot_parent(self, Vector3 local_position, Quaternion local_rotation, Transform parent):
        """Set local position, rotation and parent in one call."""
        if self._local_position != local_position or self._local_rotation != local_rotation or self._parent is not parent:
            self._local_position = local_position
            self._local_rotation = local_rotation
            self.parent = parent

    cdef Vector3 c_world_to_local(self, Vector3 world_pos):
        """Transform a world-space position into this transform's local space."""
        return self._rotation.c_inverse().c_mul_vector(world_pos.c_sub_vector(self._position))

    cdef Vector3 c_local_to_world(self, Vector3 local_pos):
        """Transform a local-space position into world space."""
        return self._rotation.c_mul_vector(local_pos).c_add_vector(self._position)

    cdef void c_rotate(self, Quaternion rotation):
        """Apply an additional rotation to this transform."""
        cdef Quaternion new_rot = self._rotation.c_mul_quat(rotation)
        if self._parent is None:
            self._local_rotation = new_rot
        else:
            self._local_rotation = self._parent._rotation.c_inverse().c_mul_quat(new_rot)
        self.c_propagate()

    cdef void c_translate(self, Vector3 translation):
        """Translate this transform in its local space."""
        cdef Vector3 new_pos = self._rotation.c_mul_vector(translation).c_add_vector(self._position)
        if self._parent is None:
            self._local_position = new_pos
        else:
            self._local_position = self._parent._rotation.c_inverse().c_mul_vector(
                new_pos.c_sub_vector(self._parent._position)
            )
        self.c_propagate()

    cdef void c_add_child(self, Transform transform):
        """Add a child transform to this node."""
        transform.parent = self


