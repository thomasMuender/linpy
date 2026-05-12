from __future__ import annotations

import math
import numpy as np
from util import rsqrt, name_to_idx, has_unique_characters, highest_idx

class Vector:
    __slots__ = 'num', 'values'

    def __init__(self, *args):
        if type(self) not in (Vec2, Vec3, Vec4):
            raise ValueError

        self.num = int(type(self).__name__[-1])
        self.values = []
        
        if len(args) == 1 and isinstance(args[0], (int, float)):
            self.values = [float(args[0])] * self.num
            return

        for value in args:
            if isinstance(value, (int, float)):
                self.values.append(float(value))
            elif isinstance(value, (list, tuple, Vec2, Vec3, Vec4)):
                for v in value:
                    self.values.append(float(v))
            elif isinstance(value, np.ndarray):
                l = value.tolist()
                for v in l:
                    self.values.append(float(v))
            else:
                raise NotImplementedError
            
        if len(self.values) != self.num:
            raise AttributeError
        
    def __str__(self) -> str:
        return '[' + ', '.join([str(v) for v in self.values]) + ']'
    
    def __len__(self):
        return self.num

    def __iter__(self):
        return iter(self.values)
    
    def __add__(self, other):
        if type(self) == type(other):
            return type(self)([a + b for a, b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return type(self)([a + other for a in self])
        raise TypeError

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return self + other

    def __sub__(self, other):
        if type(self) == type(other):
            return type(self)([a - b for a, b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return type(self)([a - other for a in self])
        raise TypeError

    def __mul__(self, other):
        if type(self) == type(other):
            return type(self)([a * b for a, b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return type(self)([a * other for a in self])
        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self * other

    def __truediv__(self, other):
        if type(self) == type(other):
            return type(self)([a / b for a, b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return type(self)([a / other for a in self])
        raise TypeError

    def __getattr__(self, name):
        if all(c in ('x', 'y', 'z', 'w') for c in name):
            if len(name) == 1:
                return self.values[name_to_idx(name)]
            elif len(name) == 2:
                return Vec2([self.values[name_to_idx(n)] for n in name])
            elif len(name) == 3:
                return Vec3([self.values[name_to_idx(n)] for n in name])
            elif len(name) == 4:
                return Vec4([self.values[name_to_idx(n)] for n in name])
            else:
                raise IndexError
        raise ValueError

    def __getitem__(self, items):
        sliced = self.values[items]
        if isinstance(sliced, (int, float)):
            return sliced
        if len(sliced) == 2:
            return Vec2(sliced)
        elif len(sliced) == 3:
            return Vec3(sliced)
        elif len(sliced) == 4:
            return Vec4(sliced)
        raise ValueError

    def __setattr__(self, name, value):
        set = False
        if name in __class__.__slots__:
            super().__setattr__(name, value)
            set = True
        elif all(c in ('x', 'y', 'z', 'w') for c in name) and highest_idx(name) < self.num and has_unique_characters(name):
            if isinstance(value, (int, float)):
                for n in name:
                    self.values[name_to_idx(n)] = float(value)
                set = True
            elif isinstance(value, (list, tuple, Vec2, Vec3, Vec4)) and 1 < len(name) <= self.num and len(name) == len(value):
                for i, n in enumerate(name):
                    self.values[name_to_idx(n)] = float(value[i])
                set = True
        if not set:
            raise IndexError

    #TODO check types and stuff
    def __setitem__(self, key, newvalue):
        self.values[key] = newvalue
    
    def dot(self, other) -> float:
        assert type(self) == type(other)
        return sum([a * b for a, b in zip(self, other)])
    
    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> None:
        scale = rsqrt(self.dot(self))
        self.values = [a * scale for a in self]
    
    def normalized(self):
        return rsqrt(self.dot(self)) * self
    
    def inverse(self):
        return self * -1.
    
    def sin(self):
        return type(self)([math.sin(i) for i in self.values])
    
    def cos(self):
        return type(self)([math.cos(i) for i in self.values])
    
    def degree(self):
        return type(self)([math.degrees(d) for d in self.values])
    
    def radians(self):
        return type(self)([math.radians(d) for d in self.values])
    
    def toList(self):
        return self.values

    def toNumpy(self):
        return np.array(self.values)


class Vec2(Vector):
    __slots__ = ()

    @property
    def zero(self) -> Vec2:
        return Vec2(0.0, 0.0)
    
    @property
    def one(self) -> Vec2:
        return Vec2(1.0, 1.0)
    
    @property
    def x_one(self) -> Vec2:
        return Vec2(1.0, 0.0)
    
    @property
    def y_one(self) -> Vec2:
        return Vec2(0.0, 1.0)


class Vec3(Vector):
    __slots__ = ()
    
    @property
    def zero(self) -> Vec3:
        return Vec3(0.0, 0.0, 0.0)
    
    @property
    def one(self) -> Vec3:
        return Vec3(1.0, 1.0, 1.0)
    
    @property
    def x_one(self) -> Vec3:
        return Vec3(1.0, 0.0, 0.0)
    
    @property
    def y_one(self) -> Vec3:
        return Vec3(0.0, 1.0, 0.0)
    
    @property
    def z_one(self) -> Vec3:
        return Vec3(0.0, 0.0, 1.0)
    
    def cross(self, other):
        return (self * other.yzx - self.yzx * other).yzx


class Vec4(Vector):
    __slots__ = ()
    
    @property
    def zero(self) -> Vec4:
        return Vec4(0.0, 0.0, 0.0, 0.0)
    
    @property
    def one(self) -> Vec4:
        return Vec4(1.0, 1.0, 1.0, 1.0)
    
    @property
    def x_one(self) -> Vec4:
        return Vec4(1.0, 0.0, 0.0, 0.0)
    
    @property
    def y_one(self) -> Vec4:
        return Vec4(0.0, 1.0, 0.0, 0.0)
    
    @property
    def z_one(self) -> Vec4:
        return Vec4(0.0, 0.0, 1.0, 0.0)
    
    @property
    def w_one(self) -> Vec4:
        return Vec4(0.0, 0.0, 0.0, 1.0)