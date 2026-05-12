from __future__ import annotations

from vector import Vec3, Vec4
from quaternion import Quaternion

class Transform:
    __slots__ = '__name', '__local_position', '__local_rotation', '__position', '__rotation', '__parent', '__children'

    def __init__(self, name: str, pos: Vec3, rot: Quaternion):
        assert isinstance(pos, Vec3)
        assert isinstance(rot, Quaternion)

        self.__name = name
        self.__local_position = pos
        self.__local_rotation = rot
        self.__position = pos
        self.__rotation = rot
        self.__parent = None
        self.__children = []

    @property
    def name(self):
        return self.__name

    @property
    def local_position(self):
        return self.__local_position
    
    @local_position.setter
    def local_position(self, value):
        assert isinstance(value, Vec3)
        self.__local_position = value
        self.__position = self.__local_position if self.__parent == None else (self.__parent.rotation * self.__local_position) + self.__parent.position
        self.__update_children()

    @property
    def local_rotation(self):
        return self.__local_rotation
    
    @local_rotation.setter
    def local_rotation(self, value):
        assert isinstance(value, Quaternion)
        self.__local_rotation = value
        self.__rotation = self.__local_rotation if self.__parent == None else self.__parent.rotation * self.__local_rotation
        self.__update_children()

    @property
    def position(self):
        return self.__position
    
    @position.setter
    def position(self, value):
        assert isinstance(value, Vec3)
        self.__position = value
        self.__local_position = self.__position if self.__parent == None else (self.__parent.rotation.inverse() * self.__position) + self.__parent.position.inverse()
        self.__update_children()

    @property
    def rotation(self):
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value):
        assert isinstance(value, Quaternion)
        self.__rotation = value
        self.__local_rotation = self.__rotation if self.__parent == None else self.__parent.rotation.inverse() * self.__rotation
        self.__update_children()

    @property
    def parent(self):
        return self.__parent
    
    @parent.setter
    def parent(self, value: Transform | None):
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
    
    def __str__(self) -> str:
        return self.name + ": [Pos: " + str(self.position) + ", Rot: " + str(self.rotation.toEuler()) + "]"

    def __mul__(self, other):
        if isinstance(other, Transform):
            return Transform((self.rotation * other.position) + self.position, self.rotation * other.rotation)
        elif isinstance(other, Vec3):
            return (self.rotation * other) + self.position
        elif isinstance(other, Vec4):
            return Vec4((self.rotation * other.xyz) + (self.position * other.w), other.w)
        raise TypeError
    
    def __len__(self):
        return len(self.__children)

    def __iter__(self):
        return iter(self.__children)
    
    def __getitem__(self, items):
        return self.__children[items]

    def __setitem__(self, key, newvalue):
        assert isinstance(newvalue, Transform)
        self.__children[key] = newvalue
        self.__children[key].parent = self

    def inverse(self):
        inv_rot = self.rotation.inverse()
        inv_pos = inv_rot * self.position.inverse()
        return Transform(inv_pos, inv_rot)
    
    def rotate(self, rotation: Quaternion):
        self.rotation = self.rotation * rotation
    
    def translate(self, translation: Vec3):
        self.position = (self.rotation * translation) + self.position

    def add_child(self, transform: Transform) -> None:
        assert isinstance(transform, Transform)
        transform.parent = self

    def __add_child(self, transform: Transform) -> None:
        assert isinstance(transform, Transform)
        self.__children.append(transform)

    def __remove_child(self, transform: Transform) -> None:
        if transform in self.__children:
            self.__children.remove(transform)

    def __update_children(self):
        for child in self.__children:
            child.parent = self

    def world_to_local(self, world_pos: Vec3):
        return self.rotation.inverse() * (world_pos - self.position)
    
    def local_to_world(self, local_pos: Vec3):
        return self.rotation * local_pos + self.position


    

t1 = Transform("t1", Vec3(1, 0, 0), Quaternion.fromEuler(0, 0, 0))
t2 = Transform("t2", Vec3(2, 0, 0), Quaternion.fromEuler(0, 0, 0))
t3 = Transform("t3", Vec3(3, 0, 0), Quaternion.fromEuler(0, 0, 0))

t2.parent = t1
t3.parent = t2

#t1.add_child(t2)
print(t2, t3)

print(t2.world_to_local(Vec3(4, 0, 0)))
print(t2.local_to_world(Vec3(4, 0, 0)))

