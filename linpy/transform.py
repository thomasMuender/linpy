from __future__ import annotations

from typing import Iterator
from .vector import Vector3, Vector4
from .quaternion import Quaternion


class Transform:
    __slots__ = '__name', '__local_position', '__local_rotation', '__position', '__rotation', '__parent', '__children'

    def __init__(self, pos: Vector3, rot: Quaternion, name: str = "") -> None:
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
        return self.__name

    @property
    def local_position(self) -> Vector3:
        return self.__local_position
    
    @local_position.setter
    def local_position(self, value: Vector3) -> None:
        if not isinstance(value, Vector3):
            raise TypeError(f"local_position must be Vector3, got {type(value).__name__}")
        self.__local_position = value
        self.__position = self.__local_position if self.__parent is None else (self.__parent.rotation * self.__local_position) + self.__parent.position
        self.__update_children()

    @property
    def local_rotation(self) -> Quaternion:
        return self.__local_rotation
    
    @local_rotation.setter
    def local_rotation(self, value: Quaternion) -> None:
        if not isinstance(value, Quaternion):
            raise TypeError(f"local_rotation must be Quaternion, got {type(value).__name__}")
        self.__local_rotation = value
        self.__rotation = self.__local_rotation if self.__parent is None else self.__parent.rotation * self.__local_rotation
        self.__update_children()

    @property
    def position(self) -> Vector3:
        return self.__position
    
    @position.setter
    def position(self, value: Vector3) -> None:
        if not isinstance(value, Vector3):
            raise TypeError(f"position must be Vector3, got {type(value).__name__}")
        self.__position = value
        self.__local_position = self.__position if self.__parent is None else self.__parent.rotation.inverse() * (self.__position - self.__parent.position)
        self.__update_children()

    @property
    def rotation(self) -> Quaternion:
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        if not isinstance(value, Quaternion):
            raise TypeError(f"rotation must be Quaternion, got {type(value).__name__}")
        self.__rotation = value
        self.__local_rotation = self.__rotation if self.__parent is None else self.__parent.rotation.inverse() * self.__rotation
        self.__update_children()

    @property
    def parent(self) -> Transform | None:
        return self.__parent
    
    @parent.setter
    def parent(self, value: Transform | None) -> None:
        if value is self:
            raise ValueError("A transform cannot be its own parent")
        if isinstance(value, Transform) and value is not self.__parent and self.__is_descendant(value):
            raise ValueError("Cannot set a descendant as parent")

        if self.__parent is not None:
            self.__parent.__remove_child(self)

        if isinstance(value, Transform):
            self.__parent = value
            self.__parent.__add_child(self)
            self.__position = (self.__parent.rotation * self.__local_position) + self.__parent.position
            self.__rotation = self.__parent.rotation * self.__local_rotation
        else:
            self.__parent = None
            self.__position = self.__local_position
            self.__rotation = self.__local_rotation

        self.__update_children()
    
    def __repr__(self) -> str:
        return f"Transform({self.name!r}, {self.position!r}, {self.rotation!r})"

    def __str__(self) -> str:
        return self.name + ": [Pos: " + str(self.position) + ", Rot: " + str(self.rotation.toEuler()) + "]"

    def __mul__(self, other: Transform | Vector3 | Vector4) -> Transform | Vector3 | Vector4:
        if isinstance(other, Transform):
            return Transform((self.rotation * other.position) + self.position, self.rotation * other.rotation, self.name + "*" + other.name)
        elif isinstance(other, Vector3):
            return (self.rotation * other) + self.position
        elif isinstance(other, Vector4):
            return Vector4((self.rotation * other.xyz) + (self.position * other.w), other.w)
        raise TypeError(f"Cannot multiply Transform by {type(other).__name__}")
    
    def __len__(self) -> int:
        return self.__children.__len__()

    def __iter__(self) -> Iterator[Transform]:
        return self.__children.__iter__()
    
    def __getitem__(self, items: int | slice) -> Transform:
        return self.__children.__getitem__(items)

    def __setitem__(self, key: int, newvalue: Transform) -> None:
        if not isinstance(newvalue, Transform):
            raise TypeError(f"children must be Transform, got {type(newvalue).__name__}")
        self.__children[key] = newvalue
        self.__children[key].parent = self

    def inverse(self) -> Transform:
        inv_rot = self.rotation.inverse()
        inv_pos = inv_rot * self.position.inverse()
        return Transform(inv_pos, inv_rot, self.name + "_inv")
    
    def rotate(self, rotation: Quaternion) -> None:
        self.rotation = self.rotation * rotation
    
    def translate(self, translation: Vector3) -> None:
        self.position = (self.rotation * translation) + self.position

    def add_child(self, transform: Transform) -> None:
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

    def __update_children(self) -> None:
        for child in self.__children:
            child.parent = self

    def world_to_local(self, world_pos: Vector3) -> Vector3:
        return self.rotation.inverse() * (world_pos - self.position)
    
    def local_to_world(self, local_pos: Vector3) -> Vector3:
        return self.rotation * local_pos + self.position