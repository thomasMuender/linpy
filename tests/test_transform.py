"""
Unit tests for the Transform class.

Covers construction, property access, parent/child hierarchy, world/local
coordinate conversion, rotation/translation, multiplication, inverse,
and edge cases related to transform trees.
"""

import math
import pytest
import numpy as np

from linpy.vector import Vector3, Vector4
from linpy.quaternion import Quaternion
from linpy.transform import Transform


# ------------------------------------------------------------------ helpers --

def assert_vec_equal(v, expected, tol=1e-6):
    """Assert each component of a vector matches expected values within tolerance."""
    values = list(v)
    assert len(values) == len(expected), f"Length mismatch: {len(values)} vs {len(expected)}"
    for i, (a, b) in enumerate(zip(values, expected)):
        assert abs(a - b) < tol, f"Component {i}: {a} != {b} (diff={abs(a - b)})"


def make_identity_transform(name: str = "t") -> Transform:
    """Create a transform at the origin with identity rotation."""
    return Transform(Vector3(0, 0, 0), Quaternion.from_euler(0, 0, 0), name)


def make_transform(name: str, x: float, y: float, z: float,
                   rx: float = 0, ry: float = 0, rz: float = 0) -> Transform:
    """Shortcut to create a positioned and rotated transform."""
    return Transform(Vector3(x, y, z), Quaternion.from_euler(rx, ry, rz), name)


# ============================================================
# Construction & Basic Properties
# ============================================================

class TestTransformConstruction:
    """Verify Transform creation and property access."""

    def test_basic_construction(self):
        # A transform stores name, position and rotation.
        t = Transform(Vector3(1, 2, 3), Quaternion.from_euler(0, 0, 0), "root")
        assert t.name == "root"
        assert_vec_equal(t.position, [1, 2, 3])

    def test_initial_local_equals_world(self):
        # Without a parent, local and world values are identical.
        t = make_transform("t", 5, 6, 7)
        assert_vec_equal(t.local_position, [5, 6, 7])
        assert_vec_equal(t.position, [5, 6, 7])

    def test_invalid_pos_type_raises(self):
        # Position must be a Vector3, not a plain list.
        with pytest.raises(TypeError):
            Transform([1, 2, 3], Quaternion.from_euler(0, 0, 0), "t")

    def test_invalid_rot_type_raises(self):
        # Rotation must be a Quaternion, not a Vector4.
        with pytest.raises(TypeError):
            Transform(Vector3(0, 0, 0), Vector4(0, 0, 0, 1), "t")

    def test_parent_is_none_by_default(self):
        # A freshly created transform has no parent.
        t = make_identity_transform()
        assert t.parent is None

    def test_children_empty_by_default(self):
        # A freshly created transform has no children.
        t = make_identity_transform()
        assert len(t) == 0
        assert list(t) == []


# ============================================================
# Parenting (single level)
# ============================================================

class TestSingleLevelParenting:
    """Verify parent-child relationships and world position computation
    for one level of nesting (parent → child)."""

    def test_set_parent_updates_world_position(self):
        # When a child is parented, its world position = parent.pos + local.pos
        # (with identity rotation on the parent).
        parent = make_transform("parent", 10, 0, 0)
        child = Transform(Vector3(5, 0, 0), Quaternion.from_euler(0, 0, 0), "child")
        child.parent = parent
        # World position should be parent (10,0,0) + local (5,0,0) = (15,0,0)
        assert_vec_equal(child.position, [15, 0, 0])

    def test_local_position_preserved_after_parenting(self):
        # After parenting, the child's local position stays the same.
        child = make_transform("child", 3, 4, 5)
        parent = make_identity_transform("parent")
        child.parent = parent
        assert_vec_equal(child.local_position, [3, 4, 5])

    def test_parent_rotation_affects_child_world_position(self):
        # A parent rotated 90° about Z should swing the child's local offset.
        # Local (1,0,0) becomes world (0,1,0) after 90° Z rotation.
        parent = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "parent")
        child = Transform(Vector3(1, 0, 0), Quaternion.from_euler(0, 0, 0), "child")
        child.parent = parent
        assert_vec_equal(child.position, [0, 1, 0])

    def test_parent_rotation_affects_child_world_rotation(self):
        # Child's world rotation = parent.rotation * child.local_rotation.
        parent = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "parent")
        child = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "child")
        child.parent = parent
        # Net rotation = 90+90 = 180° about Z.
        v = child.rotation * Vector3(1, 0, 0)
        assert_vec_equal(v, [-1, 0, 0])

    def test_unparent_restores_local_as_world(self):
        # Setting parent to None reverts world coords to local coords.
        parent = make_transform("parent", 10, 0, 0)
        child = make_transform("child", 5, 0, 0)
        child.parent = parent
        assert_vec_equal(child.position, [15, 0, 0])
        child.parent = None
        # After unparenting, world = local = (5,0,0).
        assert_vec_equal(child.position, [5, 0, 0])

    def test_add_child_method(self):
        # add_child() should parent the child to this transform.
        parent = make_transform("parent", 10, 0, 0)
        child = make_transform("child", 3, 0, 0)
        parent.add_child(child)
        assert child.parent is parent
        assert len(parent) == 1

    def test_reparent_removes_from_old_parent(self):
        # Reparenting a child should remove it from the previous parent.
        parent1 = make_identity_transform("p1")
        parent2 = make_identity_transform("p2")
        child = make_identity_transform("child")
        child.parent = parent1
        assert len(parent1) == 1
        child.parent = parent2
        assert len(parent1) == 0
        assert len(parent2) == 1

    def test_parent_property_getter(self):
        # child.parent should return the parent transform object.
        parent = make_identity_transform("parent")
        child = make_identity_transform("child")
        child.parent = parent
        assert child.parent is parent


# ============================================================
# Parenting (multi-level tree)
# ============================================================

class TestMultiLevelTree:
    """Verify world position/rotation propagation through deeper hierarchies."""

    def test_three_level_position_accumulation(self):
        # grandparent(1,0,0) → parent(2,0,0) → child(3,0,0)
        # All identity rotation, so world positions just add up.
        gp = make_transform("gp", 1, 0, 0)
        p = make_transform("p", 2, 0, 0)
        c = make_transform("c", 3, 0, 0)
        p.parent = gp
        c.parent = p
        # parent world = 1+2 = 3, child world = 3+3 = 6
        assert_vec_equal(p.position, [3, 0, 0])
        assert_vec_equal(c.position, [6, 0, 0])

    def test_moving_root_propagates_to_all_descendants(self):
        # Changing root position should cascade through the tree.
        root = make_transform("root", 0, 0, 0)
        mid = make_transform("mid", 1, 0, 0)
        leaf = make_transform("leaf", 1, 0, 0)
        mid.parent = root
        leaf.parent = mid
        assert_vec_equal(leaf.position, [2, 0, 0])
        # Move root
        root.local_position = Vector3(10, 0, 0)
        # mid world = 10+1 = 11, leaf world = 11+1 = 12
        assert_vec_equal(mid.position, [11, 0, 0])
        assert_vec_equal(leaf.position, [12, 0, 0])

    def test_rotating_root_propagates_to_all_descendants(self):
        # Rotating root 90° Z: child local (1,0,0) → world (0,1,0)
        # Grandchild local (1,0,0) → further rotated relative to child.
        root = make_identity_transform("root")
        child = make_transform("child", 1, 0, 0)
        grandchild = make_transform("gc", 1, 0, 0)
        child.parent = root
        grandchild.parent = child
        root.local_rotation = Quaternion.from_rotation_z(90)
        # child world pos: rot(90Z) * (1,0,0) = (0,1,0)
        assert_vec_equal(child.position, [0, 1, 0])
        # grandchild world pos: root.rot * child.local + child.world
        # = rot(90Z) * (1,0,0) + (0,1,0) = (0,1,0) + (0,1,0) = (0,2,0)
        assert_vec_equal(grandchild.position, [0, 2, 0])

    def test_multiple_children_under_same_parent(self):
        # One parent can have multiple children.
        parent = make_identity_transform("parent")
        c1 = make_transform("c1", 1, 0, 0)
        c2 = make_transform("c2", 0, 1, 0)
        c3 = make_transform("c3", 0, 0, 1)
        parent.add_child(c1)
        parent.add_child(c2)
        parent.add_child(c3)
        assert len(parent) == 3
        assert list(parent) == [c1, c2, c3]

    def test_child_indexing(self):
        # Children should be accessible by index.
        parent = make_identity_transform("parent")
        c0 = make_transform("c0", 1, 0, 0)
        c1 = make_transform("c1", 0, 1, 0)
        parent.add_child(c0)
        parent.add_child(c1)
        assert parent[0] is c0
        assert parent[1] is c1


# ============================================================
# Setting World Position/Rotation (back-computing local)
# ============================================================

class TestWorldToLocalBackCompute:
    """Verify that setting world position/rotation correctly
    back-computes the corresponding local values."""

    def test_set_world_position_updates_local(self):
        # Setting world position on a child should update its local position.
        parent = make_transform("parent", 10, 0, 0)
        child = make_transform("child", 0, 0, 0)
        child.parent = parent
        child.position = Vector3(20, 0, 0)
        # local should be 20 - 10 = 10 (identity parent rotation)
        assert_vec_equal(child.local_position, [10, 0, 0])

    def test_set_world_position_with_rotated_parent(self):
        # With a rotated parent, the local position is in the parent's local frame.
        parent = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "parent")
        child = make_transform("child", 0, 0, 0)
        child.parent = parent
        # Set world position to (0, 5, 0)
        child.position = Vector3(0, 5, 0)
        # Parent rotated 90° Z: invRot * (worldPos - parentPos)
        # invRot(90Z) * (0,5,0) ≈ (5, 0, 0) in parent's local frame
        assert_vec_equal(child.local_position, [5, 0, 0])

    def test_set_world_rotation_updates_local(self):
        # Setting world rotation on a child should update its local rotation.
        parent = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "parent")
        child = make_transform("child", 0, 0, 0)
        child.parent = parent
        # Set child world rotation to 90° Z (same as parent)
        child.rotation = Quaternion.from_rotation_z(90)
        # local_rotation = invParent * worldRot = inv(90Z) * 90Z = identity
        v = child.local_rotation * Vector3(1, 0, 0)
        assert_vec_equal(v, [1, 0, 0])

    def test_set_local_position_updates_world(self):
        # Setting local position should recompute world position.
        parent = make_transform("parent", 10, 0, 0)
        child = make_transform("child", 0, 0, 0)
        child.parent = parent
        child.local_position = Vector3(5, 0, 0)
        assert_vec_equal(child.position, [15, 0, 0])

    def test_set_local_rotation_updates_world(self):
        # Setting local rotation should recompute world rotation.
        parent = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "parent")
        child = make_transform("child", 0, 0, 0)
        child.parent = parent
        child.local_rotation = Quaternion.from_rotation_z(90)
        # world = parent(90Z) * local(90Z) = 180° Z
        v = child.rotation * Vector3(1, 0, 0)
        assert_vec_equal(v, [-1, 0, 0])


# ============================================================
# world_to_local / local_to_world
# ============================================================

class TestCoordinateConversion:
    """Verify world ↔ local coordinate conversion methods."""

    def test_world_to_local_identity(self):
        # With identity transform, world_to_local is a no-op.
        t = make_identity_transform()
        assert_vec_equal(t.world_to_local(Vector3(5, 6, 7)), [5, 6, 7])

    def test_local_to_world_identity(self):
        # With identity transform, local_to_world is a no-op.
        t = make_identity_transform()
        assert_vec_equal(t.local_to_world(Vector3(5, 6, 7)), [5, 6, 7])

    def test_world_to_local_with_translation(self):
        # Transform at (10,0,0): world (15,0,0) → local (5,0,0).
        t = make_transform("t", 10, 0, 0)
        assert_vec_equal(t.world_to_local(Vector3(15, 0, 0)), [5, 0, 0])

    def test_local_to_world_with_translation(self):
        # Transform at (10,0,0): local (5,0,0) → world (15,0,0).
        t = make_transform("t", 10, 0, 0)
        assert_vec_equal(t.local_to_world(Vector3(5, 0, 0)), [15, 0, 0])

    def test_world_to_local_with_rotation(self):
        # 90° Z rotation: world (0,1,0) → local (1,0,0).
        t = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t")
        assert_vec_equal(t.world_to_local(Vector3(0, 1, 0)), [1, 0, 0])

    def test_local_to_world_with_rotation(self):
        # 90° Z rotation: local (1,0,0) → world (0,1,0).
        t = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t")
        assert_vec_equal(t.local_to_world(Vector3(1, 0, 0)), [0, 1, 0])

    def test_roundtrip_world_local_world(self):
        # world → local → world should return the original point.
        t = Transform(Vector3(3, 4, 5), Quaternion.from_euler(30, 45, 60), "t")
        world_pt = Vector3(10, 20, 30)
        local_pt = t.world_to_local(world_pt)
        restored = t.local_to_world(local_pt)
        assert_vec_equal(restored, [10, 20, 30])

    def test_roundtrip_local_world_local(self):
        # local → world → local should return the original point.
        t = Transform(Vector3(3, 4, 5), Quaternion.from_euler(30, 45, 60), "t")
        local_pt = Vector3(1, 2, 3)
        world_pt = t.local_to_world(local_pt)
        restored = t.world_to_local(world_pt)
        assert_vec_equal(restored, [1, 2, 3])


# ============================================================
# rotate() and translate()
# ============================================================

class TestRotateAndTranslate:
    """Verify in-place rotation and translation methods."""

    def test_translate_identity(self):
        # Translating an identity transform by (5,0,0) places it at (5,0,0).
        t = make_identity_transform()
        t.translate(Vector3(5, 0, 0))
        assert_vec_equal(t.position, [5, 0, 0])

    def test_translate_adds_in_local_frame(self):
        # translate() applies the translation in the transform's local frame.
        # 90° Z rotation: local X axis points in world Y direction.
        t = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t")
        t.translate(Vector3(1, 0, 0))
        # (rot * (1,0,0)) + (0,0,0) = (0,1,0)
        assert_vec_equal(t.position, [0, 1, 0])

    def test_rotate_accumulates(self):
        # Two successive 90° rotations about Z should give 180°.
        t = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t")
        t.rotate(Quaternion.from_rotation_z(90))
        v = t.rotation * Vector3(1, 0, 0)
        assert_vec_equal(v, [-1, 0, 0])

    def test_translate_updates_children(self):
        # Moving a parent should cascade to children.
        parent = make_identity_transform("parent")
        child = make_transform("child", 1, 0, 0)
        child.parent = parent
        parent.translate(Vector3(10, 0, 0))
        assert_vec_equal(child.position, [11, 0, 0])


# ============================================================
# Transform Multiplication
# ============================================================

class TestTransformMultiplication:
    """Verify Transform * Transform, Transform * Vector3, Transform * Vector4."""

    def test_mul_Vector3_applies_rotation_and_translation(self):
        # T * v should rotate then translate the vector.
        t = Transform(Vector3(10, 0, 0), Quaternion.from_rotation_z(90), "t")
        result = t * Vector3(1, 0, 0)
        # rot(90Z) * (1,0,0) = (0,1,0), then + (10,0,0) = (10,1,0)
        assert_vec_equal(result, [10, 1, 0])

    def test_mul_Vector3_identity(self):
        # Identity transform should not modify the vector.
        t = make_identity_transform()
        assert_vec_equal(t * Vector3(1, 2, 3), [1, 2, 3])

    def test_mul_Vector4_with_w1(self):
        # Vector4 with w=1 should be treated as a point (translation applied).
        t = make_transform("t", 10, 0, 0)
        result = t * Vector4(1, 0, 0, 1)
        assert_vec_equal(result, [11, 0, 0, 1])

    def test_mul_Vector4_with_w0(self):
        # Vector4 with w=0 should be treated as a direction (no translation).
        t = make_transform("t", 10, 0, 0)
        result = t * Vector4(1, 0, 0, 0)
        assert_vec_equal(result, [1, 0, 0, 0])

    def test_mul_transform_composes(self):
        # Multiplying two transforms composes them.
        t1 = make_transform("t1", 10, 0, 0)
        t2 = make_transform("t2", 5, 0, 0)
        composed = t1 * t2
        # composed position = rot1 * pos2 + pos1 = (5,0,0) + (10,0,0) = (15,0,0)
        assert_vec_equal(composed.position, [15, 0, 0])

    def test_mul_transform_with_rotation(self):
        # Composing a rotated transform with a translated one.
        t1 = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t1")
        t2 = make_transform("t2", 1, 0, 0)
        composed = t1 * t2
        # rot(90Z) * (1,0,0) + (0,0,0) = (0,1,0)
        assert_vec_equal(composed.position, [0, 1, 0])

    def test_mul_invalid_type_raises(self):
        # Multiplying by an unsupported type should raise TypeError.
        t = make_identity_transform()
        with pytest.raises(TypeError):
            t * 42

    def test_mul_composed_name(self):
        # The composed transform's name concatenates the two names.
        t1 = make_identity_transform("A")
        t2 = make_identity_transform("B")
        assert (t1 * t2).name == "A*B"


# ============================================================
# Inverse
# ============================================================

class TestTransformInverse:
    """Verify Transform.inverse() undoes the transform."""

    def test_inverse_position_only(self):
        # Inverse of a pure translation at (5,0,0): applying it to (5,0,0) → (0,0,0).
        t = make_transform("t", 5, 0, 0)
        inv = t.inverse()
        result = inv * Vector3(5, 0, 0)
        assert_vec_equal(result, [0, 0, 0])

    def test_inverse_rotation_only(self):
        # Inverse of a pure rotation should undo it.
        t = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "t")
        inv = t.inverse()
        v = Vector3(1, 0, 0)
        rotated = t * v
        restored = inv * rotated
        assert_vec_equal(restored, [1, 0, 0])

    def test_inverse_combined(self):
        # Inverse of a combined rotation+translation should undo both.
        t = Transform(Vector3(3, 4, 5), Quaternion.from_euler(30, 45, 60), "t")
        inv = t.inverse()
        v = Vector3(10, 20, 30)
        assert_vec_equal(inv * (t * v), [10, 20, 30])

    def test_inverse_name(self):
        # The inverse transform's name has "_inv" appended.
        t = make_identity_transform("root")
        assert t.inverse().name == "root_inv"

    def test_double_inverse_identity(self):
        # Applying inverse twice should be (approximately) identity.
        t = Transform(Vector3(3, 4, 5), Quaternion.from_euler(30, 45, 60), "t")
        double_inv = t.inverse().inverse()
        v = Vector3(1, 2, 3)
        assert_vec_equal(double_inv * v, list(t * v))


# ============================================================
# String Representations
# ============================================================

class TestTransformStr:
    """Verify __str__ output."""

    def test_str_contains_name_and_position(self):
        # __str__ should include the transform's name.
        t = make_transform("myNode", 1, 2, 3)
        s = str(t)
        assert "myNode" in s
        assert "Pos:" in s
        assert "Rot:" in s


# ============================================================
# Edge Cases
# ============================================================

class TestTransformEdgeCases:
    """Test boundary conditions and unusual scenarios."""

    def test_self_parent_raises_value_error(self):
        # A transform cannot be its own parent.
        t = make_identity_transform("t")
        with pytest.raises(ValueError, match="cannot be its own parent"):
            t.parent = t

    def test_descendant_parent_raises_value_error(self):
        # Parenting a node to one of its descendants must be prevented.
        root = make_identity_transform("root")
        child = make_transform("child", 1, 0, 0)
        grandchild = make_transform("grandchild", 1, 0, 0)
        child.parent = root
        grandchild.parent = child
        with pytest.raises(ValueError, match="Cannot set a descendant as parent"):
            root.parent = grandchild

    def test_deep_hierarchy(self):
        # A 10-level hierarchy should correctly accumulate positions.
        transforms = []
        for i in range(10):
            t = make_transform(f"t{i}", 1, 0, 0)
            if i > 0:
                t.parent = transforms[i - 1]
            transforms.append(t)
        # Each level adds (1,0,0), so the leaf should be at (10,0,0).
        assert_vec_equal(transforms[-1].position, [10, 0, 0])

    def test_rotated_deep_hierarchy(self):
        # Root rotated 90° Z, then 3 children each offset by (1,0,0) local.
        root = Transform(Vector3(0, 0, 0), Quaternion.from_rotation_z(90), "root")
        c1 = make_transform("c1", 1, 0, 0)
        c2 = make_transform("c2", 1, 0, 0)
        c3 = make_transform("c3", 1, 0, 0)
        c1.parent = root
        c2.parent = c1
        c3.parent = c2
        # All children inherit root's 90° Z rotation.
        # c1: rot(90Z)*(1,0,0) = (0,1,0)
        # c2: rot(90Z)*(1,0,0) + c1.world = (0,1,0) + (0,1,0) = (0,2,0)
        # c3: rot(90Z)*(1,0,0) + c2.world = (0,1,0) + (0,2,0) = (0,3,0)
        assert_vec_equal(c1.position, [0, 1, 0])
        assert_vec_equal(c2.position, [0, 2, 0])
        assert_vec_equal(c3.position, [0, 3, 0])

    def test_reparent_updates_world_position(self):
        # Moving a child from one parent to another should update world pos.
        p1 = make_transform("p1", 10, 0, 0)
        p2 = make_transform("p2", -10, 0, 0)
        child = make_transform("child", 1, 0, 0)
        child.parent = p1
        assert_vec_equal(child.position, [11, 0, 0])
        child.parent = p2
        # Local pos (1,0,0) + p2 (-10,0,0) = (-9,0,0)
        assert_vec_equal(child.position, [-9, 0, 0])

    def test_removing_parent_preserves_local(self):
        # After unparenting, world == local (the original local_position).
        parent = make_transform("parent", 100, 0, 0)
        child = make_transform("child", 5, 0, 0)
        child.parent = parent
        original_local = list(child.local_position)
        child.parent = None
        assert_vec_equal(child.position, original_local)

    def test_moving_parent_with_rotated_child(self):
        # If a child has local rotation, translating the parent should
        # correctly update the child's world position.
        parent = make_identity_transform("parent")
        child = Transform(Vector3(1, 0, 0), Quaternion.from_rotation_z(90), "child")
        child.parent = parent
        assert_vec_equal(child.position, [1, 0, 0])
        parent.translate(Vector3(5, 0, 0))
        assert_vec_equal(child.position, [6, 0, 0])

    def test_translate_and_rotate_combined(self):
        # Sequential translate + rotate should work correctly.
        t = make_identity_transform()
        t.translate(Vector3(5, 0, 0))
        t.rotate(Quaternion.from_rotation_z(90))
        # Now at (5,0,0) rotated 90° Z.
        # Transforming (1,0,0): rot(90Z)*(1,0,0) + (5,0,0) = (0,1,0) + (5,0,0) = (5,1,0)
        assert_vec_equal(t * Vector3(1, 0, 0), [5, 1, 0])

    def test_position_setter_invalid_type_raises(self):
        # Position setter should reject non-Vector3.
        t = make_identity_transform()
        with pytest.raises(TypeError):
            t.position = [1, 2, 3]

    def test_rotation_setter_invalid_type_raises(self):
        # Rotation setter should reject non-Quaternion.
        t = make_identity_transform()
        with pytest.raises(TypeError):
            t.rotation = Vector3(0, 0, 0)

    def test_local_position_setter_invalid_type_raises(self):
        # local_position setter should reject non-Vector3.
        t = make_identity_transform()
        with pytest.raises(TypeError):
            t.local_position = (1, 2, 3)

    def test_local_rotation_setter_invalid_type_raises(self):
        # local_rotation setter should reject non-Quaternion.
        t = make_identity_transform()
        with pytest.raises(TypeError):
            t.local_rotation = Vector4(0, 0, 0, 1)

    def test_iter_children(self):
        # Iterating over a parent yields its children in insertion order.
        parent = make_identity_transform("parent")
        children = [make_transform(f"c{i}", i, 0, 0) for i in range(5)]
        for c in children:
            parent.add_child(c)
        assert list(parent) == children

    def test_len_updates_on_reparent(self):
        # len() should track the correct number of children.
        parent = make_identity_transform("parent")
        c1 = make_identity_transform("c1")
        c2 = make_identity_transform("c2")
        parent.add_child(c1)
        assert len(parent) == 1
        parent.add_child(c2)
        assert len(parent) == 2
        c1.parent = None
        assert len(parent) == 1


# ============================================================
# forward / right / up properties
# ============================================================

class TestDirectionProperties:
    def test_identity_forward_is_z(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        assert_vec_equal(t.forward, [0, 0, 1])

    def test_identity_right_is_x(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        assert_vec_equal(t.right, [1, 0, 0])

    def test_identity_up_is_y(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        assert_vec_equal(t.up, [0, 1, 0])

    def test_rotated_forward(self):
        t = Transform(Vector3.zero(), Quaternion.from_rotation_y(90))
        assert_vec_equal(t.forward, [1, 0, 0], tol=1e-5)

    def test_rotated_right(self):
        t = Transform(Vector3.zero(), Quaternion.from_rotation_y(90))
        assert_vec_equal(t.right, [0, 0, -1], tol=1e-5)

    def test_rotated_up_unchanged_by_y_rotation(self):
        t = Transform(Vector3.zero(), Quaternion.from_rotation_y(90))
        assert_vec_equal(t.up, [0, 1, 0], tol=1e-5)

    def test_axes_are_orthonormal(self):
        t = Transform(Vector3.zero(), Quaternion.from_rotation_x(37))
        assert t.forward.dot(t.right) == pytest.approx(0.0, abs=1e-6)
        assert t.forward.dot(t.up) == pytest.approx(0.0, abs=1e-6)
        assert t.right.dot(t.up) == pytest.approx(0.0, abs=1e-6)
        assert t.forward.magnitude() == pytest.approx(1.0, abs=1e-6)
        assert t.right.magnitude() == pytest.approx(1.0, abs=1e-6)
        assert t.up.magnitude() == pytest.approx(1.0, abs=1e-6)


# ============================================================
# look_at
# ============================================================

class TestLookAt:
    def test_look_at_forward_z(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.look_at(Vector3(0, 0, 5))
        assert_vec_equal(t.forward, [0, 0, 1], tol=1e-5)

    def test_look_at_positive_x(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.look_at(Vector3(10, 0, 0))
        assert_vec_equal(t.forward, [1, 0, 0], tol=1e-5)

    def test_look_at_negative_z(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.look_at(Vector3(0, 0, -10))
        assert_vec_equal(t.forward, [0, 0, -1], tol=1e-5)

    def test_look_at_with_offset_position(self):
        t = Transform(Vector3(5, 0, 0), Quaternion.identity())
        t.look_at(Vector3(5, 0, 10))
        assert_vec_equal(t.forward, [0, 0, 1], tol=1e-5)

    def test_look_at_same_position_is_noop(self):
        rot = Quaternion.from_rotation_x(45)
        t = Transform(Vector3(1, 2, 3), rot)
        t.look_at(Vector3(1, 2, 3))
        # Rotation should remain unchanged
        assert list(t.rotation) == pytest.approx(list(rot), abs=1e-7)

    def test_look_at_custom_up(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.look_at(Vector3(1, 0, 0), Vector3(0, 0, 1))
        assert_vec_equal(t.forward, [1, 0, 0], tol=1e-5)
        assert_vec_equal(t.up, [0, 0, 1], tol=1e-5)

    def test_look_at_propagates_to_children(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        child = Transform(Vector3(0, 0, 1), Quaternion.identity(), "child")
        parent.add_child(child)
        parent.look_at(Vector3(1, 0, 0))
        # Child world position should have been updated
        expected_child_pos = parent.rotation * Vector3(0, 0, 1) + parent.position
        assert_vec_equal(child.position, list(expected_child_pos), tol=1e-5)


# ============================================================
# to_matrix4x4 / from_matrix4x4
# ============================================================

class TestMatrix4x4:
    def test_identity_gives_identity_matrix(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        np.testing.assert_allclose(t.to_matrix4x4(), np.eye(4), atol=1e-7)

    def test_position_in_last_column(self):
        t = Transform(Vector3(1, 2, 3), Quaternion.identity())
        m = t.to_matrix4x4()
        np.testing.assert_allclose(m[:3, 3], [1, 2, 3], atol=1e-7)
        np.testing.assert_allclose(m[3, :], [0, 0, 0, 1], atol=1e-7)

    def test_rotation_in_upper_left(self):
        q = Quaternion.from_rotation_y(90)
        t = Transform(Vector3.zero(), q)
        m = t.to_matrix4x4()
        np.testing.assert_allclose(m[:3, :3], q.to_matrix3x3(), atol=1e-7)

    def test_roundtrip_to_from(self):
        pos = Vector3(5, -3, 7)
        rot = Quaternion.from_euler(30, 45, 60)
        t = Transform(pos, rot, "test")
        m = t.to_matrix4x4()
        t2 = Transform.from_matrix4x4(m, "restored")
        assert_vec_equal(t2.position, list(pos), tol=1e-5)
        assert_vec_equal(t2.rotation, list(rot), tol=1e-5)
        assert t2.name == "restored"

    def test_matrix_transforms_point(self):
        t = Transform(Vector3(10, 0, 0), Quaternion.from_rotation_z(90))
        m = t.to_matrix4x4()
        pt = np.array([1, 0, 0, 1])
        result = m @ pt
        expected = t * Vector3(1, 0, 0)
        np.testing.assert_allclose(result[:3], list(expected), atol=1e-5)

    def test_from_matrix4x4_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            Transform.from_matrix4x4(np.eye(3))

    def test_from_matrix4x4_not_array_raises(self):
        with pytest.raises(ValueError):
            Transform.from_matrix4x4([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
