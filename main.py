from linpy.vector3 import Vector3
from linpy.quaternion import Quaternion
from linpy.transform import Transform
from linpy.scene_graph import SceneGraph

# Create a scene graph
sg = SceneGraph()

# Define some transforms: a robot arm with shoulder -> elbow -> wrist
sg.apply_transform(
    "shoulder",
    SceneGraph.root_name,
    Vector3(0.0, 1.0, 0.0),
    Quaternion.from_euler(0.0, 0.0, 45.0),
)

sg.apply_transform(
    "elbow",
    "shoulder",
    Vector3(2.0, 0.0, 0.0),
    Quaternion.from_euler(0.0, 0.0, -30.0),
)

sg.apply_transform(
    "wrist",
    "elbow",
    Vector3(1.5, 0.0, 0.0),
    Quaternion.from_euler(0.0, 0.0, 10.0),
)

# Print the hierarchy
print("=== Scene Graph ===")
sg.print_graph()

# Print world positions of each joint
print("\n=== World Positions ===")
for name in sg:
    t = sg[name]
    print(f"{t.name}: pos={t.position}, rot={t.local_rotation.to_euler()}")

# Update the shoulder rotation — children propagate automatically
print("\n=== After rotating shoulder by 20 deg on Y ===")
sg.apply_transform(
    "shoulder",
    SceneGraph.root_name,
    Vector3(0.0, 1.0, 0.0),
    Quaternion.from_euler(0.0, 20.0, 45.0),
)

for name in sg:
    t = sg[name]
    print(f"{t.name}: pos={t.position}")

