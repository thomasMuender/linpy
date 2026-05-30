# LINPY

A high-performance Cython library for 3D vector math, quaternion rotations, and scene-graph transforms.

## Requirements

- Python 3.10+
- Cython

## Installation

```bash
python setup.py build_ext --inplace
pip install -e .
```

---

## Vector3

`Vector3` is a Cython extension type with `x`, `y`, `z` components stored as C doubles.

```python
from linpy import Vector3

a = Vector3(1, 2, 3)
b = Vector3(4, 5, 6)
```

### Construction

```python
Vector3(1, 2, 3)                    # from individual components
Vector3.from_iterable([1, 2, 3])    # from any iterable of 3 elements
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

Read-only 3-component swizzle properties are available for all permutations of `x y z`:

```python
v = Vector3(1, 2, 3)
v.x            # 1.0  (single component → float)
v.xyz          # Vector3(1.0, 2.0, 3.0)
v.zyx          # Vector3(3.0, 2.0, 1.0)
v.xxy          # Vector3(1.0, 1.0, 2.0)
```

### Vector Operations

```python
a.dot(b)          # dot product → float
a.magnitude()     # Euclidean length → float
a.normalize()     # normalise in-place
a.normalized()    # returns a new normalised vector
a.cross(b)        # cross product → Vector3
```

### Interpolation & Distance

```python
a.lerp(b, 0.5)    # linear interpolation (t=0 → a, t=1 → b)
a.distance(b)     # Euclidean distance → float
```

### Translation & Conversion

```python
a.translate(b)    # alias for a + b → Vector3
a.to_list()       # returns [x, y, z] as list[float]
```

### Constants (static methods)

```python
Vector3.zero()    # Vector3(0, 0, 0)
Vector3.one()     # Vector3(1, 1, 1)
Vector3.x_one()   # Vector3(1, 0, 0)
Vector3.y_one()   # Vector3(0, 1, 0)
Vector3.z_one()   # Vector3(0, 0, 1)
```

---

## Quaternions

`Quaternion` represents a rotation in 3D space stored as `(x, y, z, w)`.

```python
from linpy import Quaternion, Vector3
```

### Construction

```python
Quaternion(x, y, z, w)                   # from raw components
Quaternion.from_iterable([x, y, z, w])   # from any iterable of 4 elements
```

### Factory Methods

```python
Quaternion.identity()                    # identity quaternion (no rotation)
Quaternion.from_euler(rx, ry, rz)        # Euler angles in degrees, default order ZXY
Quaternion.from_euler(rx, ry, rz, "XYZ") # explicit rotation order
```

Supported Euler orders: `XYZ`, `XZY`, `YXZ`, `YZX`, `ZXY`, `ZYX`.

### Rotating Vectors

```python
q = Quaternion.from_euler(0, 0, 90)
q * Vector3(1, 0, 0)   # Vector3(0.0, 1.0, 0.0)
```

### Composition & Algebra

```python
q1 * q2          # compose rotations
q.inverse()      # inverse rotation
q.normalize()    # normalise in-place
q.normalized()   # returns a new normalised quaternion
q.dot(other)     # dot product → float
```

### Incremental Rotation

```python
q = q.rotate_x(30)   # apply additional 30° rotation about X
q = q.rotate_y(45)
q = q.rotate_z(60)
```

### Interpolation

```python
q1.slerp(q2, 0.5)   # spherical linear interpolation (t=0 → q1, t=1 → q2)
```

### Conversion

```python
q.to_euler()         # Vector3 of Euler angles (degrees), default order ZXY
q.to_euler("XYZ")    # explicit order
q.to_list()          # returns [x, y, z, w] as list[float]
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
| `x_dir` | World-space X direction (+X rotated by `rotation`) |
| `y_dir` | World-space Y direction (+Y rotated by `rotation`) |
| `z_dir` | World-space Z direction (+Z rotated by `rotation`) |

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
t.translate(Vector3(1, 0, 0))              # move by offset in local frame
t.rotate(Quaternion.from_euler(0, 0, 90))  # accumulate additional rotation
```

### Batch Update

```python
t.set_local_pos_rot(pos, rot)                # update local position and rotation at once
t.set_local_pos_rot_parent(pos, rot, parent) # update local pose and reparent in one call
```

### Coordinate Conversion

```python
t.local_to_world(Vector3(1, 0, 0))  # transform point from local → world space
t.world_to_local(Vector3(1, 0, 0))  # transform point from world → local space
```

### Transform Multiplication

```python
result = t * Vector3(1, 0, 0)   # rotate then translate a point
composed = t1 * t2              # compose two transforms
```

### Inverse

```python
inv = t.inverse()   # returns a new Transform that undoes t
inv * (t * point)   # ≈ point
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
root.rotate(Quaternion.from_euler(0, 45, 0))
print(tip.position)     # updated world position

# Transform a world-space point into the tip's local frame
local_pt = tip.world_to_local(Vector3(5, 0, 0))
```

---

## Scene Graph

`SceneGraph` is a managed registry of named `Transform` nodes. It handles out-of-order insertion (a child can be registered before its parent) and provides a single entry point for updating the pose of any node.

```python
from linpy.scene_graph import SceneGraph
from linpy import Vector3, Quaternion
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

sg.apply_transform("torso",  root_name, Vector3(0, 1, 0), Quaternion.identity())
sg.apply_transform("head",   "torso",   Vector3(0, 1, 0), Quaternion.identity())
sg.apply_transform("l_hand", "torso",   Vector3(-1, 0, 0), Quaternion.identity())
sg.apply_transform("r_hand", "torso",   Vector3( 1, 0, 0), Quaternion.identity())
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
sg.apply_transform("wheel", "axle", Vector3(1, 0, 0), Quaternion.identity())

# Now register the parent — the placeholder is promoted automatically
sg.apply_transform("axle", root_name, Vector3(0, 0, 5), Quaternion.identity())

print(sg["wheel"].position)   # Vector3(1.0, 0.0, 5.0) — correctly parented
```
