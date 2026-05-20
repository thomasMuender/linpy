from __future__ import annotations

from typing import Iterator, overload
import numpy as np
from .vector import Vector3, Vector4
from .quaternion import Quaternion

class Transform:
    """A transform node with local and world position/rotation in a hierarchy."""

    __slots__ = '__name', '__local_position', '__local_rotation', '__position', '__rotation', '__parent', '__children'

    def __init__(self, pos: Vector3, rot: Quaternion, name: str = "") -> None:
        """Initialize a transform with a local position and rotation.

        :param pos: The local position.
        :param rot: The local rotation.
        :param name: An optional name for this transform.
        :raises TypeError: If *pos* is not a Vector3 or *rot* is not a Quaternion.
        """
        if not isinstance(pos, Vector3):
            raise TypeError(f"pos must be Vector3, got {type(pos).__name__}")
        if not isinstance(rot, Quaternion):
            raise TypeError(f"rot must be Quaternion, got {type(rot).__name__}")

        self.__name: str = name
        self.__local_position: Vector3 = pos
        self.__local_rotation: Quaternion = rot
        self.__position: Vector3 = pos
        self.__rotation: Quaternion = rot
        self.__parent: Transform | None = None
        self.__children: list[Transform] = []

    @property
    def name(self) -> str:
        """The name of this transform.

        :return: The transform name.
        :rtype: str
        """
        return self.__name

    @property
    def local_position(self) -> Vector3:
        """The local position relative to the parent.

        :return: The local position vector.
        :rtype: Vector3
        """
        return self.__local_position
    
    @local_position.setter
    def local_position(self, value: Vector3) -> None:
        """Set the local position and propagate world transforms.

        :param value: The new local position.
        :raises TypeError: If *value* is not a Vector3.
        """
        if not isinstance(value, Vector3):
            raise TypeError(f"local_position must be Vector3, got {type(value).__name__}")
        
        if self.__local_position != value:
            self.__local_position = value
            self.__propagate()

    @property
    def local_rotation(self) -> Quaternion:
        """The local rotation relative to the parent.

        :return: The local rotation quaternion.
        :rtype: Quaternion
        """
        return self.__local_rotation
    
    @local_rotation.setter
    def local_rotation(self, value: Quaternion) -> None:
        """Set the local rotation and propagate world transforms.

        :param value: The new local rotation.
        :raises TypeError: If *value* is not a Quaternion.
        """
        if not isinstance(value, Quaternion):
            raise TypeError(f"local_rotation must be Quaternion, got {type(value).__name__}")
        
        if self.__local_rotation != value:
            self.__local_rotation = value
            self.__propagate()

    @property
    def position(self) -> Vector3:
        """The world-space position.

        :return: The world position vector.
        :rtype: Vector3
        """
        return self.__position
    
    @position.setter
    def position(self, value: Vector3) -> None:
        """Set the world-space position, updating local position accordingly.

        :param value: The new world position.
        :raises TypeError: If *value* is not a Vector3.
        """
        if not isinstance(value, Vector3):
            raise TypeError(f"position must be Vector3, got {type(value).__name__}")
        
        if self.__position != value:
            self.__local_position = value if self.__parent is None else self.__parent.rotation.inverse() * (value - self.__parent.position)
            self.__propagate()   

    @property
    def rotation(self) -> Quaternion:
        """The world-space rotation.

        :return: The world rotation quaternion.
        :rtype: Quaternion
        """
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        """Set the world-space rotation, updating local rotation accordingly.

        :param value: The new world rotation.
        :raises TypeError: If *value* is not a Quaternion.
        """
        if isinstance(value, Quaternion):
            raise TypeError(f"rotation must be Quaternion, got {type(value).__name__}")
        
        if self.__rotation != value:
            self.__local_rotation = value if self.__parent is None else self.__parent.rotation.inverse() * value
            self.__propagate()

    @property
    def parent(self) -> Transform | None:
        """The parent transform, or None if this is a root.

        :return: The parent transform.
        :rtype: Transform | None
        """
        return self.__parent
    
    @parent.setter
    def parent(self, value: Transform | None) -> None:
        """Set the parent transform, reparenting this node in the hierarchy.

        :param value: The new parent, or None to detach.
        :raises ValueError: If setting self as parent or creating a cycle.
        """
        if value is self.__parent:
            return
        
        if value is self:
            raise ValueError("A transform cannot be its own parent")
        
        if isinstance(value, Transform) and self.__is_descendant(value):
            raise ValueError("Cannot set a descendant as parent")

        # Detach from the current parent
        if self.__parent is not None:
            self.__parent.__remove_child(self)

        # Attach to the new parent, or promote to world root
        if isinstance(value, Transform):
            self.__parent = value
            self.__parent.__add_child(self)
        else:
            self.__parent = None

        # Recompute world transform and propagate to children
        self.__propagate()

    def set_local_pos_rot(self, local_position: Vector3, local_rotation: Quaternion) -> None:

        if not isinstance(local_position, Vector3) or not isinstance(local_rotation, Quaternion):
            raise TypeError
        
        if self.__local_position != local_position or self.__local_rotation != local_rotation:
            self.__local_position = local_position
            self.__local_rotation = local_rotation
            self.__propagate()
    
    def set_local_pos_rot_parent(self, local_position: Vector3, local_rotation: Quaternion, parent: Transform) -> None:

        if not isinstance(local_position, Vector3) or not isinstance(local_rotation, Quaternion) or not isinstance(parent, Transform):
            raise TypeError
        
        if self.__local_position != local_position or self.__local_rotation != local_rotation or self.__parent is not parent:
            self.__local_position = local_position
            self.__local_rotation = local_rotation
            self.parent = parent

    @property
    def x_dir(self) -> Vector3:
        """The local X axis direction in world space.

        :return: The world-space X direction.
        :rtype: Vector3
        """
        return self.rotation * Vector3(1.0, 0.0, 0.0)

    @property
    def y_dir(self) -> Vector3:
        """The local Y axis direction in world space.

        :return: The world-space Y direction.
        :rtype: Vector3
        """
        return self.rotation * Vector3(0.0, 1.0, 0.0)
    
    @property
    def z_dir(self) -> Vector3:
        """The local Z axis direction in world space.

        :return: The world-space Z direction.
        :rtype: Vector3
        """
        return self.rotation * Vector3(0.0, 0.0, 1.0)

    def __repr__(self) -> str:
        """Return a detailed string representation of this transform.

        :return: Repr string showing name, position, and rotation.
        :rtype: str
        """
        return f"Transform({self.name!r}, {self.position!r}, {self.rotation!r})"

    def __str__(self) -> str:
        """Return a human-readable string with position and Euler rotation.

        :return: Formatted string.
        :rtype: str
        """
        return self.name + ": [Pos: " + str(self.position) + ", Rot: " + str(self.rotation.to_euler()) + "]"


    @overload
    def __mul__(self, other: Transform) -> Transform:
        ...

    @overload
    def __mul__(self, other: Vector3) -> Vector3:
        ...

    @overload
    def __mul__(self, other: Vector4) -> Vector4:
        ...

    def __mul__(self, other: Transform | Vector3 | Vector4) -> Transform | Vector3 | Vector4:
        """Compose this transform with another object.

        Transform * Transform -> composed transform (other expressed in this space)
        Transform * Vector3  -> point transformed into world space
        Transform * Vector4  -> homogeneous-coordinate transform (w-weighted position)
        """
        if isinstance(other, Transform):
            return Transform((self.rotation * other.position) + self.position, self.rotation * other.rotation, self.name + "*" + other.name)
        elif isinstance(other, Vector3):
            return (self.rotation * other) + self.position
        elif isinstance(other, Vector4):
            return Vector4((self.rotation * other.xyz) + (self.position * other.w), other.w)
        
        raise TypeError(f"Cannot multiply Transform by {type(other).__name__}")
    
    def __len__(self) -> int:
        """Return the number of child transforms.

        :return: Child count.
        :rtype: int
        """
        return self.__children.__len__()

    def __iter__(self) -> Iterator[Transform]:
        """Iterate over child transforms.

        :return: Iterator of child Transform objects.
        :rtype: Iterator[Transform]
        """
        return self.__children.__iter__()
    
    def __getitem__(self, items: int | slice) -> Transform | list[Transform]:
        """Get child transform(s) by index or slice.

        :param items: An integer index or slice.
        :return: A single child or list of children.
        """
        return self.__children.__getitem__(items)

    def __setitem__(self, key: int, newvalue: Transform) -> None:
        """Set a child transform at the given index.

        :param key: The index to set.
        :param newvalue: The Transform to place at the index.
        :raises TypeError: If *newvalue* is not a Transform.
        """
        if isinstance(newvalue, Transform):
            self.__children[key] = newvalue
            self.__children[key].parent = self

        raise TypeError(f"children must be Transform, got {type(newvalue).__name__}")

    def to_matrix4x4(self) -> np.ndarray:
        """Convert this transform to a 4x4 homogeneous transformation matrix.

        :return: A 4x4 NumPy matrix encoding rotation and translation.
        :rtype: numpy.ndarray
        """
        m = self.rotation.to_matrix4x4()
        m[0, 3] = self.position.x
        m[1, 3] = self.position.y
        m[2, 3] = self.position.z
        return m

    @staticmethod
    def from_matrix4x4(matrix: np.ndarray, name: str = "") -> Transform:
        """Create a Transform from a 4x4 homogeneous matrix.

        :param matrix: A 4x4 NumPy array.
        :param name: Optional name for the transform.
        :return: The constructed Transform.
        :rtype: Transform
        :raises ValueError: If *matrix* is not a 4x4 ndarray.
        """
        if isinstance(matrix, np.ndarray) and matrix.shape == (4, 4):
            pos = Vector3(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
            rot = Quaternion.from_matrix3x3(matrix[:3, :3])
            return Transform(pos, rot, name)
        
        raise ValueError("Input must be a 4x4 numpy array")

    def inverse(self) -> Transform:
        """Compute the inverse of this transform.

        :return: A new Transform that undoes this transform.
        :rtype: Transform
        """
        inv_rot = self.rotation.inverse()
        inv_pos = inv_rot * (-self.position)
        return Transform(inv_pos, inv_rot, self.name + "_inv")
    
    def rotate(self, rotation: Quaternion) -> None:
        """Apply an additional rotation to this transform.

        :param rotation: The rotation to apply.
        """
        self.rotation = self.rotation * rotation
    
    def translate(self, translation: Vector3) -> None:
        """Translate this transform in its local space.

        :param translation: The translation vector in local coordinates.
        """
        self.position = (self.rotation * translation) + self.position

    def look_at(self, target: Vector3, up: Vector3 | None = None) -> None:
        """Orient this transform to face the target position.

        :param target: The world-space position to look at.
        :param up: The up reference vector (default world Y-up).
        """
        direction = target - self.position
        if direction.dot(direction) < 1e-15:
            return
        self.rotation = Quaternion.look_rotation(direction, up)

    def add_child(self, transform: Transform) -> None:
        """Add a child transform to this node.

        :param transform: The transform to add as a child.
        :raises TypeError: If *transform* is not a Transform.
        """
        if not isinstance(transform, Transform):
            raise TypeError(f"child must be Transform, got {type(transform).__name__}")
        
        transform.parent = self

    def __add_child(self, transform: Transform) -> None:
        self.__children.append(transform)

    def __remove_child(self, transform: Transform) -> None:
        if transform in self.__children:
            self.__children.remove(transform)

    def __is_descendant(self, transform: Transform) -> bool:
        for child in self.__children:
            if child is transform or child.__is_descendant(transform):
                return True
        return False

    def __propagate(self) -> None:
        """Recalculate this transform's world position/rotation from its locals,
        then recursively propagate to all children."""

        stack: list[Transform] = list()
        stack.append(self)

        while stack:
            transform: Transform = stack.pop()

            if transform.__parent is None:
                transform.__position = transform.__local_position
                transform.__rotation = transform.__local_rotation
            else:
                transform.__position = (transform.__parent.rotation * transform.__local_position) + transform.__parent.position
                transform.__rotation = transform.__parent.rotation * transform.__local_rotation
            
            for child in transform:
                stack.append(child)

    def world_to_local(self, world_pos: Vector3) -> Vector3:
        """Transform a world-space position into this transform's local space.

        :param world_pos: The world-space position.
        :return: The position in local space.
        :rtype: Vector3
        """
        return self.rotation.inverse() * (world_pos - self.position)
    
    def local_to_world(self, local_pos: Vector3) -> Vector3:
        """Transform a local-space position into world space.

        :param local_pos: The local-space position.
        :return: The position in world space.
        :rtype: Vector3
        """
        return self.rotation * local_pos + self.position