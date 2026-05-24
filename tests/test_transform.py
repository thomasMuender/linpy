import math
import pytest
from linpy import Vector3, Transform
from linpy.quaternion import Quaternion


class TestTransformCreation:
    def test_create(self):
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.identity()
        t = Transform(pos, rot, "test")
        assert t.position == pos
        assert t.rotation == rot
        assert t.name == "test"

    def test_create_invalid_pos_raises(self):
        with pytest.raises(TypeError):
            Transform("not_a_vector", Quaternion.identity())

    def test_create_invalid_rot_raises(self):
        with pytest.raises(TypeError):
            Transform(Vector3.zero(), "not_a_quat")

    def test_initial_local_equals_world(self):
        pos = Vector3(5.0, 10.0, 15.0)
        rot = Quaternion.from_euler(30.0, 45.0, 0.0)
        t = Transform(pos, rot)
        assert t.local_position == t.position
        assert t.local_rotation == t.rotation


class TestTransformProperties:
    def test_set_local_position(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.local_position = Vector3(1.0, 2.0, 3.0)
        assert t.position == Vector3(1.0, 2.0, 3.0)

    def test_set_local_position_invalid_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(TypeError):
            t.local_position = "invalid"

    def test_set_local_rotation(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        new_rot = Quaternion.from_euler(0.0, 90.0, 0.0)
        t.local_rotation = new_rot
        assert t.rotation == new_rot

    def test_set_local_rotation_invalid_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(TypeError):
            t.local_rotation = "invalid"

    def test_set_position(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        t.position = Vector3(5.0, 5.0, 5.0)
        assert t.position == Vector3(5.0, 5.0, 5.0)
        assert t.local_position == Vector3(5.0, 5.0, 5.0)

    def test_set_position_invalid_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(TypeError):
            t.position = "invalid"

    def test_set_rotation(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        new_rot = Quaternion.from_euler(0.0, 45.0, 0.0)
        t.rotation = new_rot
        assert t.rotation == new_rot

    def test_set_rotation_invalid_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(TypeError):
            t.rotation = "invalid"


class TestTransformHierarchy:
    def test_parent_child(self):
        parent = Transform(Vector3(10.0, 0.0, 0.0), Quaternion.identity(), "parent")
        child = Transform(Vector3(5.0, 0.0, 0.0), Quaternion.identity(), "child")
        child.parent = parent
        assert child.parent is parent
        assert child.position == Vector3(15.0, 0.0, 0.0)

    def test_child_world_position_with_parent_rotation(self):
        # Parent rotated 90 degrees around Y, child offset in X
        parent = Transform(Vector3.zero(), Quaternion.from_euler(0.0, 90.0, 0.0), "parent")
        child = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "child")
        child.parent = parent
        # Child's local X becomes world -Z after 90 deg Y rotation
        assert child.position == Vector3(0.0, 0.0, -1.0)

    def test_detach_from_parent(self):
        parent = Transform(Vector3(10.0, 0.0, 0.0), Quaternion.identity(), "parent")
        child = Transform(Vector3(5.0, 0.0, 0.0), Quaternion.identity(), "child")
        child.parent = parent
        assert child.position == Vector3(15.0, 0.0, 0.0)
        child.parent = None
        # After detach, world pos = local pos
        assert child.position == Vector3(5.0, 0.0, 0.0)

    def test_self_parent_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(ValueError):
            t.parent = t

    def test_cycle_detection_raises(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        child = Transform(Vector3.zero(), Quaternion.identity(), "child")
        child.parent = parent
        with pytest.raises(ValueError):
            parent.parent = child

    def test_children_count(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        child1 = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "c1")
        child2 = Transform(Vector3(2.0, 0.0, 0.0), Quaternion.identity(), "c2")
        child1.parent = parent
        child2.parent = parent
        assert len(parent) == 2

    def test_iterate_children(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        child1 = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "c1")
        child2 = Transform(Vector3(2.0, 0.0, 0.0), Quaternion.identity(), "c2")
        child1.parent = parent
        child2.parent = parent
        children = list(parent)
        assert child1 in children
        assert child2 in children

    def test_propagation_to_grandchild(self):
        root = Transform(Vector3(0.0, 0.0, 0.0), Quaternion.identity(), "root")
        mid = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "mid")
        leaf = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "leaf")
        mid.parent = root
        leaf.parent = mid
        assert leaf.position == Vector3(2.0, 0.0, 0.0)
        # Now move root
        root.local_position = Vector3(10.0, 0.0, 0.0)
        assert mid.position == Vector3(11.0, 0.0, 0.0)
        assert leaf.position == Vector3(12.0, 0.0, 0.0)

    def test_add_child_method(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        child = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "child")
        parent.add_child(child)
        assert child.parent is parent
        assert len(parent) == 1

    def test_add_child_invalid_raises(self):
        parent = Transform(Vector3.zero(), Quaternion.identity(), "parent")
        with pytest.raises(TypeError):
            parent.add_child("not_a_transform")


class TestTransformOperations:
    def test_mul_transform(self):
        a = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.identity(), "a")
        b = Transform(Vector3(2.0, 0.0, 0.0), Quaternion.identity(), "b")
        result = a * b
        assert result.position == Vector3(3.0, 0.0, 0.0)

    def test_mul_vector(self):
        t = Transform(Vector3(1.0, 2.0, 3.0), Quaternion.identity(), "t")
        v = Vector3(1.0, 0.0, 0.0)
        result = t * v
        assert result == Vector3(2.0, 2.0, 3.0)

    def test_mul_vector_with_rotation(self):
        # 90 deg rotation around Z, then translate by (1, 0, 0)
        t = Transform(Vector3(1.0, 0.0, 0.0), Quaternion.from_euler(0.0, 0.0, 90.0), "t")
        v = Vector3(1.0, 0.0, 0.0)
        result = t * v
        # Rotate (1,0,0) by 90Z -> (0,1,0), then add (1,0,0) -> (1,1,0)
        assert result == Vector3(1.0, 1.0, 0.0)

    def test_mul_invalid_raises(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        with pytest.raises(TypeError):
            t * "invalid"

    def test_inverse(self):
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.from_euler(30.0, 45.0, 60.0)
        t = Transform(pos, rot, "t")
        inv = t.inverse()
        # Applying transform then inverse should give back identity (approximately)
        v = Vector3(5.0, 7.0, 11.0)
        transformed = t * v
        restored = inv * transformed
        assert restored == v

    def test_world_to_local(self):
        t = Transform(Vector3(10.0, 0.0, 0.0), Quaternion.identity(), "t")
        world_pos = Vector3(15.0, 0.0, 0.0)
        local = t.world_to_local(world_pos)
        assert local == Vector3(5.0, 0.0, 0.0)

    def test_local_to_world(self):
        t = Transform(Vector3(10.0, 0.0, 0.0), Quaternion.identity(), "t")
        local_pos = Vector3(5.0, 0.0, 0.0)
        world = t.local_to_world(local_pos)
        assert world == Vector3(15.0, 0.0, 0.0)

    def test_world_to_local_roundtrip(self):
        t = Transform(Vector3(3.0, 5.0, 7.0), Quaternion.from_euler(15.0, 30.0, 45.0), "t")
        world_pos = Vector3(10.0, 20.0, 30.0)
        local = t.world_to_local(world_pos)
        restored = t.local_to_world(local)
        assert restored == world_pos

    def test_x_dir(self):
        t = Transform(Vector3.zero(), Quaternion.identity(), "t")
        assert t.x_dir == Vector3(1.0, 0.0, 0.0)

    def test_y_dir(self):
        t = Transform(Vector3.zero(), Quaternion.identity(), "t")
        assert t.y_dir == Vector3(0.0, 1.0, 0.0)

    def test_z_dir(self):
        t = Transform(Vector3.zero(), Quaternion.identity(), "t")
        assert t.z_dir == Vector3(0.0, 0.0, 1.0)

    def test_x_dir_rotated(self):
        # 90 deg around Z: x_dir should become (0, 1, 0)
        t = Transform(Vector3.zero(), Quaternion.from_euler(0.0, 0.0, 90.0), "t")
        assert t.x_dir == Vector3(0.0, 1.0, 0.0)

    def test_set_local_pos_rot(self):
        t = Transform(Vector3.zero(), Quaternion.identity())
        new_pos = Vector3(1.0, 2.0, 3.0)
        new_rot = Quaternion.from_euler(0.0, 90.0, 0.0)
        t.set_local_pos_rot(new_pos, new_rot)
        assert t.local_position == new_pos
        assert t.local_rotation == new_rot

    def test_translate(self):
        t = Transform(Vector3.zero(), Quaternion.identity(), "t")
        t.translate(Vector3(5.0, 0.0, 0.0))
        assert t.position == Vector3(5.0, 0.0, 0.0)

    def test_translate_with_rotation(self):
        # Rotated 90 deg around Y, translate in local X moves world -Z
        t = Transform(Vector3.zero(), Quaternion.from_euler(0.0, 90.0, 0.0), "t")
        t.translate(Vector3(1.0, 0.0, 0.0))
        assert t.position == Vector3(0.0, 0.0, -1.0)

    def test_rotate(self):
        t = Transform(Vector3.zero(), Quaternion.identity(), "t")
        t.rotate(Quaternion.from_euler(0.0, 90.0, 0.0))
        expected = Quaternion.from_euler(0.0, 90.0, 0.0)
        assert t.rotation == expected


class TestTransformMisc:
    def test_repr(self):
        t = Transform(Vector3(1.0, 2.0, 3.0), Quaternion.identity(), "test")
        r = repr(t)
        assert "Transform" in r
        assert "test" in r

    def test_str(self):
        t = Transform(Vector3(1.0, 2.0, 3.0), Quaternion.identity(), "test")
        s = str(t)
        assert "test" in s
        assert "Pos" in s
        assert "Rot" in s
