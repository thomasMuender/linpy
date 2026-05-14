"""
Unit tests for the SceneGraph class.

Covers:
- Initial state
- Adding transforms directly under root
- Adding transforms in tree order
- Adding transforms out of order (child before parent)
- Multi-level out-of-order insertion
- Updating an existing transform's pose
- Re-parenting an existing transform
- Promoting a placeholder when its real apply arrives
- __getitem__ access
"""

import pytest

from linpy.vector import Vector3
from linpy.quaternion import Quaternion
from linpy.scene_graph import SceneGraph


# ------------------------------------------------------------------ helpers --

IDENTITY = Quaternion.identity()


def vec(x, y, z) -> Vector3:
    return Vector3(x, y, z)


def assert_vec_equal(v, expected, tol=1e-6):
    values = list(v)
    for i, (a, b) in enumerate(zip(values, expected)):
        assert abs(a - b) < tol, f"Component {i}: {a} != {b}"


# ============================================================
# Initial state
# ============================================================

class TestSceneGraphInit:

    def test_root_exists(self):
        sg = SceneGraph()
        assert sg.root is not None

    def test_root_name(self):
        sg = SceneGraph()
        assert sg.root.name == SceneGraph.root_name

    def test_root_has_no_children(self):
        sg = SceneGraph()
        assert len(sg.root) == 0

    def test_root_at_origin(self):
        sg = SceneGraph()
        assert_vec_equal(sg.root.position, [0, 0, 0])


# ============================================================
# Adding transforms under root
# ============================================================

class TestAddUnderRoot:

    def test_add_single_transform(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 2, 3), IDENTITY)
        assert sg["A"].name == "A"

    def test_position_stored(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 2, 3), IDENTITY)
        assert_vec_equal(sg["A"].local_position, [1, 2, 3])

    def test_parent_is_root(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        assert sg["A"].parent is sg.root

    def test_root_child_count(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        assert len(sg.root) == 2


# ============================================================
# In-order insertion (parent before child)
# ============================================================

class TestInOrderInsertion:

    def test_child_attached_to_parent(self):
        sg = SceneGraph()
        sg.apply_transform("parent", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        assert sg["child"].parent is sg["parent"]

    def test_grandchild_depth(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", "A", vec(1, 0, 0), IDENTITY)
        sg.apply_transform("C", "B", vec(0, 1, 0), IDENTITY)
        assert sg["C"].parent is sg["B"]
        assert sg["B"].parent is sg["A"]
        assert sg["A"].parent is sg.root

    def test_root_has_only_top_level_children(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", "A", vec(1, 0, 0), IDENTITY)
        # Only A should be directly under root
        assert len(sg.root) == 1


# ============================================================
# Out-of-order insertion (child before parent)
# ============================================================

class TestOutOfOrderInsertion:

    def test_placeholder_created_under_root(self):
        sg = SceneGraph()
        # "parent" hasn't been applied yet
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        # placeholder for "parent" must exist under root
        assert sg["parent"].parent is sg.root

    def test_child_attached_to_placeholder(self):
        sg = SceneGraph()
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        assert sg["child"].parent is sg["parent"]

    def test_placeholder_promoted_when_parent_applied(self):
        sg = SceneGraph()
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        # Now apply the real parent under root
        sg.apply_transform("parent", SceneGraph.root_name, vec(5, 0, 0), IDENTITY)
        assert sg["parent"].parent is sg.root

    def test_child_still_under_parent_after_promotion(self):
        sg = SceneGraph()
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        sg.apply_transform("parent", SceneGraph.root_name, vec(5, 0, 0), IDENTITY)
        assert sg["child"].parent is sg["parent"]

    def test_root_child_count_after_promotion(self):
        sg = SceneGraph()
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        sg.apply_transform("parent", SceneGraph.root_name, vec(5, 0, 0), IDENTITY)
        # Only "parent" should sit directly under root
        assert len(sg.root) == 1

    def test_parent_pose_set_on_promotion(self):
        sg = SceneGraph()
        sg.apply_transform("child", "parent", vec(1, 0, 0), IDENTITY)
        sg.apply_transform("parent", SceneGraph.root_name, vec(5, 6, 7), IDENTITY)
        assert_vec_equal(sg["parent"].local_position, [5, 6, 7])

    def test_multi_level_out_of_order(self):
        # Apply C → B → A in reverse tree order: root → A → B → C
        sg = SceneGraph()
        sg.apply_transform("C", "B", vec(0, 0, 1), IDENTITY)
        sg.apply_transform("B", "A", vec(0, 1, 0), IDENTITY)
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        assert sg["A"].parent is sg.root
        assert sg["B"].parent is sg["A"]
        assert sg["C"].parent is sg["B"]

    def test_multi_level_root_child_count(self):
        sg = SceneGraph()
        sg.apply_transform("C", "B", vec(0, 0, 1), IDENTITY)
        sg.apply_transform("B", "A", vec(0, 1, 0), IDENTITY)
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        assert len(sg.root) == 1


# ============================================================
# Updating an existing transform
# ============================================================

class TestUpdateTransform:

    def test_update_position(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("A", SceneGraph.root_name, vec(9, 8, 7), IDENTITY)
        assert_vec_equal(sg["A"].local_position, [9, 8, 7])

    def test_update_does_not_duplicate_in_root(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        assert len(sg.root) == 1

    def test_reparent_existing_transform(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        sg.apply_transform("child", "A", vec(0, 1, 0), IDENTITY)
        # Move child from A to B
        sg.apply_transform("child", "B", vec(0, 2, 0), IDENTITY)
        assert sg["child"].parent is sg["B"]

    def test_reparent_removes_from_old_parent(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        sg.apply_transform("child", "A", vec(0, 1, 0), IDENTITY)
        sg.apply_transform("child", "B", vec(0, 2, 0), IDENTITY)
        assert len(sg["A"]) == 0
        assert len(sg["B"]) == 1


# ============================================================
# __getitem__
# ============================================================

class TestGetItem:

    def test_getitem_returns_transform(self):
        sg = SceneGraph()
        sg.apply_transform("X", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        from linpy.transform import Transform
        assert isinstance(sg["X"], Transform)

    def test_getitem_missing_raises(self):
        sg = SceneGraph()
        with pytest.raises(KeyError):
            _ = sg["nonexistent"]


# ============================================================
# remove
# ============================================================

class TestRemove:
    def test_remove_leaf(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(1, 0, 0), IDENTITY)
        sg.remove("A")
        with pytest.raises(KeyError):
            _ = sg["A"]
        assert len(sg.root) == 0

    def test_remove_reparents_children(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", "A", vec(1, 0, 0), IDENTITY)
        sg.remove("A")
        # B should now be a child of root
        assert sg["B"].parent is sg.root

    def test_remove_preserves_grandchildren(self):
        sg = SceneGraph()
        sg.apply_transform("A", SceneGraph.root_name, vec(0, 0, 0), IDENTITY)
        sg.apply_transform("B", "A", vec(1, 0, 0), IDENTITY)
        sg.apply_transform("C", "B", vec(2, 0, 0), IDENTITY)
        sg.remove("A")
        # B under root, C still under B
        assert sg["B"].parent is sg.root
        assert sg["C"].parent is sg["B"]

    def test_remove_nonexistent_is_noop(self):
        sg = SceneGraph()
        sg.remove("nope")  # should not raise
