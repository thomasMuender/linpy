# linpy

A Python library for 3D vector math, quaternion rotations, and scene-graph transforms.

## Requirements

- Python 3.10+
- NumPy

## Installation

```bash
pip install -e .
```

---

## Vectors

Three concrete vector types are provided: `Vec2`, `Vec3`, and `Vec4`. All share the same `Vector` base class.

```python
from linpy import Vec2, Vec3, Vec4

a = Vec3(1, 2, 3)
b = Vec3(4, 5, 6)
```

### Construction

```python
Vec3(1, 2, 3)            # from individual components
Vec3(1.0)                # broadcast — all components set to 1.0
Vec3([1, 2, 3])          # from list or tuple
Vec3(Vec2(1, 2), 3)      # from smaller vector + extra components
```

### Arithmetic

All operators work component-wise. Scalars (`int`/`float`) broadcast to every component.

```python
a + b          # component-wise addition
a - b          # component-wise subtraction
a * b          # component-wise multiplication
a / b          # component-wise division
a * 2.0        # scalar multiplication (also 2.0 * a)
-a             # negation
a.inverse()    # alias for negation (-a)
```

### Swizzling

Components can be read and written using any permutation of `x y z w` (or `r g b a` / `u v`).

```python
v = Vec3(1, 2, 3)
v.x            # 1.0  (single component → float)
v.xy           # Vec2(1.0, 2.0)
v.zyx          # Vec3(3.0, 2.0, 1.0)
v.xyzx         # Vec4(1.0, 2.0, 3.0, 1.0)

v.xy = Vec2(10, 20)   # multi-component write
v.z = 99              # single-component write
```

Indexing and slicing also work:

```python
v[0]      # 1.0
v[1:3]    # Vec2(2.0, 3.0)
v[0] = 5  # in-place element assignment
```

### Vector Operations

```python
a.dot(b)          # dot product → float
a.magnitude()     # Euclidean length → float
a.normalize()     # normalise in-place
a.normalized()    # returns a new normalised vector
```

`Vec3` additionally supports the cross product:

```python
Vec3(1, 0, 0).cross(Vec3(0, 1, 0))  # Vec3(0.0, 0.0, 1.0)
```

### Trigonometric & Conversion Helpers

```python
v.sin()       # component-wise sine (radians)
v.cos()       # component-wise cosine (radians)
v.radians()   # degrees → radians per component
v.degree()    # radians → degrees per component
v.toList()    # returns a copy as list[float]
v.toNumpy()   # returns a numpy array
```

### Constants (properties on each type)

```python
Vec3.zero    # Vec3(0, 0, 0)
Vec3.one     # Vec3(1, 1, 1)
Vec3.x_one   # Vec3(1, 0, 0)
Vec3.y_one   # Vec3(0, 1, 0)
Vec3.z_one   # Vec3(0, 0, 1)
```

---

## Quaternions

`Quaternion` represents a rotation in 3D space stored as `(x, y, z, w)`.

```python
from linpy import Quaternion, Vec3
```

### Construction

```python
Quaternion(x, y, z, w)         # from raw components
Quaternion(Vec4(x, y, z, w))   # from Vec4
```

### Factory Methods

```python
Quaternion.fromRotationX(90)           # 90° rotation about X axis (degrees)
Quaternion.fromRotationY(45)           # rotation about Y axis
Quaternion.fromRotationZ(30)           # rotation about Z axis
Quaternion.fromAngleAxis(axis, deg)    # axis (Vec3) + angle in degrees
Quaternion.fromEuler(rx, ry, rz)       # Euler angles in degrees, default order ZXY
Quaternion.fromEuler(rx, ry, rz, "XYZ")  # explicit rotation order
Quaternion.fromMatrix3x3(matrix)       # from a 3×3 numpy rotation matrix
```

Supported Euler orders: `XYZ`, `XZY`, `YXZ`, `YZX`, `ZXY`, `ZYX`.

### Rotating Vectors

```python
q = Quaternion.fromRotationZ(90)
q * Vec3(1, 0, 0)   # Vec3(0.0, 1.0, 0.0)
```

### Composition & Algebra

```python
q1 * q2          # compose rotations
q.inverse()      # inverse rotation
q.normalize()    # normalise in-place
q.normalized()   # returns a new normalised quaternion
q.dot(other)     # dot product with another Quaternion or Vec4
```

### Incremental Rotation

```python
q = q.rotateX(30)   # apply additional 30° rotation about X
q = q.rotateY(45)
q = q.rotateZ(60)
```

### Conversion

```python
q.toEuler()              # Vec3 of Euler angles (degrees), default order ZXY
q.toEuler("XYZ")         # explicit order
q.toMatrix3x3()          # 3×3 numpy rotation matrix
q.toAngleAxis()          # (axis: Vec3, angle_degrees: float)
```

---

## Transforms

`Transform` combines a position (`Vec3`) and rotation (`Quaternion`) and supports parent–child hierarchies that automatically propagate world-space values.

```python
from linpy import Transform, Vec3, Quaternion

t = Transform(Vec3(1, 2, 3), Quaternion.fromEuler(0, 0, 0), "myNode")
# name is optional:
t = Transform(Vec3(1, 2, 3), Quaternion.fromEuler(0, 0, 0))
```

### Properties

| Property | Description |
|---|---|
| `position` | World-space position (`Vec3`) |
| `rotation` | World-space rotation (`Quaternion`) |
| `local_position` | Position relative to parent (`Vec3`) |
| `local_rotation` | Rotation relative to parent (`Quaternion`) |
| `parent` | Parent `Transform` or `None` |
| `name` | Optional string label |

Setting any of these properties automatically recomputes the dependent values and cascades the update to all children.

### Parent–Child Hierarchy

```python
parent = Transform(Vec3(10, 0, 0), Quaternion.fromEuler(0, 0, 0), "parent")
child  = Transform(Vec3( 5, 0, 0), Quaternion.fromEuler(0, 0, 0), "child")

child.parent = parent          # child.position → Vec3(15, 0, 0)
parent.add_child(other_child)  # alternative API

child.parent = None            # unparent; world reverts to local values
```

Children are iterable and support indexing:

```python
len(parent)     # number of direct children
parent[0]       # first child
list(parent)    # all children
for c in parent:
    print(c.name)
```

### Translation & Rotation

```python
t.translate(Vec3(1, 0, 0))            # move by offset in local frame
t.rotate(Quaternion.fromRotationZ(90)) # accumulate additional rotation
```

### Coordinate Conversion

```python
t.local_to_world(Vec3(1, 0, 0))  # transform point from local → world space
t.world_to_local(Vec3(1, 0, 0))  # transform point from world → local space
```

### Transform Multiplication

```python
result = t * Vec3(1, 0, 0)   # rotate then translate a point
result = t * Vec4(1, 0, 0, 1)  # w=1 → point (translation applied)
result = t * Vec4(1, 0, 0, 0)  # w=0 → direction (no translation)
composed = t1 * t2            # compose two transforms
```

### Inverse

```python
inv = t.inverse()   # returns a new Transform that undoes t
inv * (t * point)   # ≈ point
```

---

## Full Example

```python
from linpy import Vec3, Quaternion, Transform

# Build a small scene graph
root  = Transform(Vec3(0, 0, 0), Quaternion.fromEuler(0, 90, 0), "root")
arm   = Transform(Vec3(2, 0, 0), Quaternion.fromEuler(0,  0, 0), "arm")
tip   = Transform(Vec3(1, 0, 0), Quaternion.fromEuler(0,  0, 0), "tip")

arm.parent = root
tip.parent = arm

print(tip.position)     # world position: propagated through tree
print(tip.rotation)     # world rotation: accumulated from root

# Rotate the root and see all descendants update automatically
root.rotate(Quaternion.fromRotationY(45))
print(tip.position)     # updated world position

# Transform a world-space point into the tip's local frame
local_pt = tip.world_to_local(Vec3(5, 0, 0))
```
