from linpy.vector import Vector3
from linpy.quaternion import Quaternion
from linpy.scene_graph import SceneGraph

import time
import random

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

start = time.time()

for i in range(10000):
    sg.apply_transform(
        "shoulder",
        SceneGraph.root_name,
        Vector3(random.random(), random.random(), random.random()),
        Quaternion(random.random(), random.random(), random.random(), random.random())
    )

print("\n=== Timing ===")
dt = time.time() - start
print(dt, "s")