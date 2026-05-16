# LINPY

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

Three concrete vector types are provided: `Vector2`, `Vector3`, and `Vector4`. All share the same `Vector` base class.

```python
from linpy import Vector2, Vector3, Vector4

a = Vector3(1, 2, 3)
b = Vector3(4, 5, 6)
```

### Construction

```python
Vector3(1, 2, 3)            # from individual components
Vector3(1.0)                # broadcast — all components set to 1.0
Vector3([1, 2, 3])          # from list or tuple
Vector3(Vector2(1, 2), 3)      # from smaller vector + extra components
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

Components can be read and written using any permutation of `x y z w`.

```python
v = Vector3(1, 2, 3)
v.x            # 1.0  (single component → float)
v.xy           # Vector2(1.0, 2.0)
v.zyx          # Vector3(3.0, 2.0, 1.0)
v.xyzx         # Vector4(1.0, 2.0, 3.0, 1.0)

v.xy = Vector2(10, 20)   # multi-component write
v.z = 99              # single-component write
```

Indexing and slicing also work:

```python
v[0]      # 1.0
v[1:3]    # Vector2(2.0, 3.0)
v[0] = 5  # in-place element assignment
```

### Vector Operations

```python
a.dot(b)          # dot product → float
a.magnitude()     # Euclidean length → float
a.normalize()     # normalise in-place
a.normalized()    # returns a new normalised vector
```

`Vector3` additionally supports the cross product:

```python
Vector3(1, 0, 0).cross(Vector3(0, 1, 0))  # Vector3(0.0, 0.0, 1.0)
```

### Interpolation & Distance

```python
a.lerp(b, 0.5)         # linear interpolation (t=0 → a, t=1 → b)
a.distance(b)          # Euclidean distance → float
a.distance_squared(b)  # squared distance (avoids sqrt) → float
a.angle_between(b)     # angle in degrees between two vectors → float
```

### Projection & Reflection

```python
v.project(onto)        # project v onto another vector
v.reflect(normal)      # reflect v around a surface normal
v.clamp_magnitude(5.0) # cap vector length to 5.0, preserving direction
```

### Component-wise Math

```python
v.abs()          # component-wise absolute value
a.min(b)         # component-wise minimum
a.max(b)         # component-wise maximum
v.floor()        # component-wise floor
v.ceil()         # component-wise ceil
```

### Trigonometric & Conversion Helpers

```python
v.sin()       # component-wise sine (radians)
v.cos()       # component-wise cosine (radians)
v.radians()   # degrees → radians per component
v.degree()    # radians → degrees per component
v.to_list()   # returns a copy as list[float]
v.to_numpy()  # returns a numpy array
```

### Constants (properties on each type)

```python
Vector3.zero    # Vector3(0, 0, 0)
Vector3.one     # Vector3(1, 1, 1)
Vector3.x_one   # Vector3(1, 0, 0)
Vector3.y_one   # Vector3(0, 1, 0)
Vector3.z_one   # Vector3(0, 0, 1)

# Vector4 additionally provides:
Vector4.w_one   # Vector4(0, 0, 0, 1)
```

---

## Quaternions

`Quaternion` represents a rotation in 3D space stored as `(x, y, z, w)`.

```python
from linpy import Quaternion, Vector3
```

### Construction

```python
Quaternion(x, y, z, w)         # from raw components
Quaternion(Vector4(x, y, z, w))   # from Vector4
```

### Factory Methods

```python
Quaternion.identity                     # identity quaternion (no rotation)
Quaternion.from_rotation_x(90)            # 90° rotation about X axis (degrees)
Quaternion.from_rotation_y(45)            # rotation about Y axis
Quaternion.from_rotation_z(30)            # rotation about Z axis
Quaternion.from_angle_axis(axis, deg)     # axis (Vector3) + angle in degrees
Quaternion.from_euler(rx, ry, rz)         # Euler angles in degrees, default order ZXY
Quaternion.from_euler(rx, ry, rz, "XYZ") # explicit rotation order
Quaternion.from_matrix3x3(matrix)         # from a 3×3 numpy rotation matrix
```

Supported Euler orders: `XYZ`, `XZY`, `YXZ`, `YZX`, `ZXY`, `ZYX`.

### Rotating Vectors

```python
q = Quaternion.from_rotation_z(90)
q * Vector3(1, 0, 0)   # Vector3(0.0, 1.0, 0.0)
```

### Composition & Algebra

```python
q1 * q2          # compose rotations
q.inverse()      # inverse rotation
q.normalize()    # normalise in-place
q.normalized()   # returns a new normalised quaternion
q.dot(other)     # dot product with another Quaternion or Vector4
```

### Incremental Rotation

```python
q = q.rotate_x(30)   # apply additional 30° rotation about X
q = q.rotate_y(45)
q = q.rotate_z(60)
```

### Interpolation & Comparison

```python
q1.slerp(q2, 0.5)       # spherical linear interpolation (t=0 → q1, t=1 → q2)
q1.angle_between(q2)    # angle in degrees between two rotations → float
```

### Look Rotation

```python
Quaternion.look_rotation(forward)       # quaternion facing a direction (default up = Y)
Quaternion.look_rotation(forward, up)   # with explicit up vector
```

### Conversion

```python
q.to_euler()              # Vector3 of Euler angles (degrees), default order ZXY
q.to_euler("XYZ")         # explicit order
q.to_matrix3x3()          # 3×3 numpy rotation matrix
q.to_matrix4x4()          # 4×4 homogeneous rotation matrix (numpy)
q.to_angle_axis()         # (axis: Vector3, angle_degrees: float)
```

---

## Transforms

`Transform` combines a position (`Vector3`) and rotation (`Quaternion`) and supports parent–child hierarchies that automatically propagate world-space values.

```python
from linpy import Transform, Vector3, Quaternion

t = Transform(Vector3(1, 2, 3), Quaternion.from_euler(0, 0, 0), "myNode")
# name is optional:
t = Transform(Vector3(1, 2, 3), Quaternion.from_euler(0, 0, 0))
```

### Properties

| Property | Description |
|---|---|
| `position` | World-space position (`Vector3`) |
| `rotation` | World-space rotation (`Quaternion`) |
| `local_position` | Position relative to parent (`Vector3`) |
| `local_rotation` | Rotation relative to parent (`Quaternion`) |
| `parent` | Parent `Transform` or `None` |
| `name` | Optional string label |
| `z_dir` | World-space Z direction (+Z rotated by `rotation`) |
| `x_dir` | World-space X direction (+X rotated by `rotation`) |
| `y_dir` | World-space Y direction (+Y rotated by `rotation`) |

Setting any writable property automatically recomputes the dependent values and cascades the update to all children.

### Parent–Child Hierarchy

```python
parent = Transform(Vector3(10, 0, 0), Quaternion.from_euler(0, 0, 0), "parent")
child  = Transform(Vector3( 5, 0, 0), Quaternion.from_euler(0, 0, 0), "child")

child.parent = parent          # child.position → Vector3(15, 0, 0)
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
t.translate(Vector3(1, 0, 0))            # move by offset in local frame
t.rotate(Quaternion.fromRotationZ(90)) # accumulate additional rotation
```

### Look At

```python
t.look_at(Vector3(10, 0, 5))                    # orient to face a target (default up = Y)
t.look_at(Vector3(10, 0, 5), Vector3(0, 0, 1))  # with explicit up vector
```

### Coordinate Conversion

```python
t.local_to_world(Vector3(1, 0, 0))  # transform point from local → world space
t.world_to_local(Vector3(1, 0, 0))  # transform point from world → local space
```

### Transform Multiplication

```python
result = t * Vector3(1, 0, 0)   # rotate then translate a point
result = t * Vector4(1, 0, 0, 1)  # w=1 → point (translation applied)
result = t * Vector4(1, 0, 0, 0)  # w=0 → direction (no translation)
composed = t1 * t2            # compose two transforms
```

### Inverse

```python
inv = t.inverse()   # returns a new Transform that undoes t
inv * (t * point)   # ≈ point
```

### Matrix Conversion

```python
m = t.to_matrix4x4()                        # 4×4 homogeneous transform matrix (numpy)
t = Transform.from_matrix4x4(m, "name")     # reconstruct Transform from a 4×4 matrix
```

---

## Full Example

```python
from linpy import Vector3, Quaternion, Transform

# Build a small scene graph
root  = Transform(Vector3(0, 0, 0), Quaternion.from_euler(0, 90, 0), "root")
arm   = Transform(Vector3(2, 0, 0), Quaternion.from_euler(0,  0, 0), "arm")
tip   = Transform(Vector3(1, 0, 0), Quaternion.from_euler(0,  0, 0), "tip")

arm.parent = root
tip.parent = arm

print(tip.position)     # world position: propagated through tree
print(tip.rotation)     # world rotation: accumulated from root

# Rotate the root and see all descendants update automatically
root.rotate(Quaternion.from_rotation_y(45))
print(tip.position)     # updated world position

# Transform a world-space point into the tip's local frame
local_pt = tip.world_to_local(Vector3(5, 0, 0))
```

---

## Scene Graph

`SceneGraph` is a managed registry of named `Transform` nodes. It handles out-of-order insertion (a child can be registered before its parent) and provides a single entry point for updating the pose of any node.

```python
from linpy import SceneGraph, Vector3, Quaternion
```

### Construction

```python
sg = SceneGraph()
sg.root        # the implicit root Transform (name "__root__")
```

### Adding / Updating Transforms

```python
sg.apply_transform(
    transform_name,   # str — name of the node to create or update
    parent_name,      # str — name of its parent (use SceneGraph.root_name for root)
    local_position,   # Vector3
    local_rotation,   # Quaternion
)
```

- If the named node does not exist it is created and attached to the parent.
- If the node already exists its pose and parent are updated in place.
- If the parent does not exist yet, a placeholder is created automatically and will be replaced when `apply_transform` is later called for that name.

```python
root_name = SceneGraph.root_name   # "__root__"

sg.apply_transform("torso",  root_name, Vector3(0, 1, 0), Quaternion.identity)
sg.apply_transform("head",   "torso",   Vector3(0, 1, 0), Quaternion.identity)
sg.apply_transform("l_hand", "torso",   Vector3(-1, 0, 0), Quaternion.identity)
sg.apply_transform("r_hand", "torso",   Vector3( 1, 0, 0), Quaternion.identity)
```

### Accessing Nodes

```python
torso = sg["torso"]          # returns the Transform for "torso"
print(torso.position)        # world-space position
print(torso.local_position)  # position relative to parent
```

### Removing Nodes

```python
sg.remove("l_hand")   # remove a node; its children are reparented to its parent
sg.remove("unknown")  # no-op if the name does not exist
```

### Printing the Tree

```python
sg.print_graph()
# __root__
#  torso
#   head
#   l_hand
#   r_hand
```

### Out-of-Order Insertion

Nodes can be registered in any order. The graph inserts a zero-pose placeholder for unknown parents and patches it once the real definition arrives.

```python
# Register the child before the parent exists
sg.apply_transform("wheel", "axle", Vector3(1, 0, 0), Quaternion.identity)

# Now register the parent — the placeholder is promoted automatically
sg.apply_transform("axle", root_name, Vector3(0, 0, 5), Quaternion.identity)

print(sg["wheel"].position)   # Vector3(1.0, 0.0, 5.0) — correctly parented
```