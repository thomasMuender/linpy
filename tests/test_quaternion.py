import math
import pytest
from linpy import Vector3
from linpy.quaternion import Quaternion


class TestQuaternionCreation:
    def test_create(self):
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0

    def test_identity(self):
        q = Quaternion.identity()
        assert q == Quaternion(0.0, 0.0, 0.0, 1.0)


class TestQuaternionOperators:
    def test_eq(self):
        a = Quaternion(0.0, 0.0, 0.0, 1.0)
        b = Quaternion(0.0, 0.0, 0.0, 1.0)
        assert a == b

    def test_ne(self):
        a = Quaternion(0.0, 0.0, 0.0, 1.0)
        b = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert a != b

    def test_neg(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert -q == Quaternion(-1.0, -2.0, -3.0, -4.0)

    def test_mul_quaternion(self):
        # Multiplying identity by any quaternion gives that quaternion
        identity = Quaternion.identity()
        q = Quaternion(0.1, 0.2, 0.3, 0.9)
        q_norm = q.normalized()
        result = identity * q_norm
        assert result == q_norm

    def test_mul_vector_identity(self):
        # Identity rotation should not change the vector
        identity = Quaternion.identity()
        v = Vector3(1.0, 2.0, 3.0)
        result = identity * v
        assert result == v

    def test_mul_vector_90_deg_y(self):
        # 90 degree rotation around Y axis: X -> Z, Z -> -X
        q = Quaternion.from_euler(0.0, 90.0, 0.0)
        v = Vector3(1.0, 0.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 0.0, -1.0)

    def test_mul_vector_90_deg_x(self):
        # 90 degree rotation around X axis: Y -> Z, Z -> -Y
        q = Quaternion.from_euler(90.0, 0.0, 0.0)
        v = Vector3(0.0, 1.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 0.0, 1.0)

    def test_mul_vector_90_deg_z(self):
        # 90 degree rotation around Z axis: X -> Y, Y -> -X
        q = Quaternion.from_euler(0.0, 0.0, 90.0)
        v = Vector3(1.0, 0.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 1.0, 0.0)

    def test_mul_invalid_type_raises(self):
        q = Quaternion.identity()
        with pytest.raises(TypeError):
            q * "invalid"


class TestQuaternionMethods:
    def test_dot_identity(self):
        a = Quaternion.identity()
        b = Quaternion.identity()
        assert a.dot(b) == pytest.approx(1.0)

    def test_dot_orthogonal(self):
        a = Quaternion(1.0, 0.0, 0.0, 0.0)
        b = Quaternion(0.0, 1.0, 0.0, 0.0)
        assert a.dot(b) == pytest.approx(0.0)

    def test_normalize(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        q.normalize()
        mag = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert mag == pytest.approx(1.0)

    def test_normalized(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        n = q.normalized()
        expected = 0.5
        assert n.x == pytest.approx(expected)
        assert n.y == pytest.approx(expected)
        assert n.z == pytest.approx(expected)
        assert n.w == pytest.approx(expected)

    def test_inverse(self):
        q = Quaternion.from_euler(45.0, 30.0, 60.0)
        inv = q.inverse()
        # q * q_inv should be identity
        result = q * inv
        identity = Quaternion.identity()
        assert result == identity

    def test_inverse_rotation(self):
        # Rotating a vector and then by inverse should give original
        q = Quaternion.from_euler(30.0, 45.0, 60.0)
        v = Vector3(1.0, 2.0, 3.0)
        rotated = q * v
        restored = q.inverse() * rotated
        assert restored == v

    def test_slerp_t0(self):
        a = Quaternion.identity()
        b = Quaternion.from_euler(0.0, 90.0, 0.0)
        result = a.slerp(b, 0.0)
        assert result == a

    def test_slerp_t1(self):
        a = Quaternion.identity()
        b = Quaternion.from_euler(0.0, 90.0, 0.0)
        result = a.slerp(b, 1.0)
        assert result == b

    def test_slerp_midpoint(self):
        a = Quaternion.identity()
        b = Quaternion.from_euler(0.0, 90.0, 0.0)
        mid = a.slerp(b, 0.5)
        # Should be approximately a 45 degree rotation
        expected = Quaternion.from_euler(0.0, 45.0, 0.0)
        assert mid == expected

    def test_to_list(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert q.to_list() == [1.0, 2.0, 3.0, 4.0]


class TestQuaternionEuler:
    def test_from_euler_identity(self):
        q = Quaternion.from_euler(0.0, 0.0, 0.0)
        assert q == Quaternion.identity()

    def test_from_euler_to_euler_roundtrip(self):
        euler = Vector3(30.0, 45.0, 60.0)
        q = Quaternion.from_euler(euler.x, euler.y, euler.z)
        result = q.to_euler()
        assert result.x == pytest.approx(euler.x, abs=0.01)
        assert result.y == pytest.approx(euler.y, abs=0.01)
        assert result.z == pytest.approx(euler.z, abs=0.01)

    def test_from_euler_90_x(self):
        q = Quaternion.from_euler(90.0, 0.0, 0.0)
        euler = q.to_euler()
        assert euler.x == pytest.approx(90.0, abs=0.01)

    def test_from_euler_90_y(self):
        q = Quaternion.from_euler(0.0, 90.0, 0.0)
        euler = q.to_euler()
        assert euler.y == pytest.approx(90.0, abs=0.01)

    def test_from_euler_xyz_order(self):
        q = Quaternion.from_euler(30.0, 45.0, 60.0, order="XYZ")
        # Just verify it doesn't crash and produces a unit quaternion
        mag = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert mag == pytest.approx(1.0)

    def test_from_euler_invalid_order_raises(self):
        with pytest.raises(ValueError):
            Quaternion.from_euler(0.0, 0.0, 0.0, order="ABC")


class TestQuaternionMisc:
    def test_len(self):
        q = Quaternion.identity()
        assert len(q) == 4

    def test_iter(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert list(q) == [1.0, 2.0, 3.0, 4.0]

    def test_str(self):
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        assert "[0.0, 0.0, 0.0, 1.0]" == str(q)


class TestQuaternionRotateAxis:
    def test_rotate_x_90(self):
        q = Quaternion.identity().rotate_x(90.0)
        expected = Quaternion.from_euler(90.0, 0.0, 0.0)
        assert q == expected

    def test_rotate_y_90(self):
        q = Quaternion.identity().rotate_y(90.0)
        expected = Quaternion.from_euler(0.0, 90.0, 0.0)
        assert q == expected

    def test_rotate_z_90(self):
        q = Quaternion.identity().rotate_z(90.0)
        expected = Quaternion.from_euler(0.0, 0.0, 90.0)
        assert q == expected

    def test_rotate_x_applies_to_vector(self):
        q = Quaternion.identity().rotate_x(90.0)
        v = Vector3(0.0, 1.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 0.0, 1.0)

    def test_rotate_y_applies_to_vector(self):
        q = Quaternion.identity().rotate_y(90.0)
        v = Vector3(1.0, 0.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 0.0, -1.0)

    def test_rotate_z_applies_to_vector(self):
        q = Quaternion.identity().rotate_z(90.0)
        v = Vector3(1.0, 0.0, 0.0)
        result = q * v
        assert result == Vector3(0.0, 1.0, 0.0)

    def test_rotate_x_chained(self):
        q = Quaternion.identity().rotate_x(45.0).rotate_x(45.0)
        expected = Quaternion.from_euler(90.0, 0.0, 0.0)
        assert q == expected

    def test_rotate_y_chained(self):
        q = Quaternion.identity().rotate_y(30.0).rotate_y(60.0)
        expected = Quaternion.from_euler(0.0, 90.0, 0.0)
        assert q == expected

    def test_rotate_z_chained(self):
        q = Quaternion.identity().rotate_z(20.0).rotate_z(70.0)
        expected = Quaternion.from_euler(0.0, 0.0, 90.0)
        assert q == expected

    def test_rotate_x_zero(self):
        q = Quaternion.from_euler(30.0, 45.0, 60.0)
        result = q.rotate_x(0.0)
        assert result == q

    def test_rotate_y_zero(self):
        q = Quaternion.from_euler(30.0, 45.0, 60.0)
        result = q.rotate_y(0.0)
        assert result == q

    def test_rotate_z_zero(self):
        q = Quaternion.from_euler(30.0, 45.0, 60.0)
        result = q.rotate_z(0.0)
        assert result == q
