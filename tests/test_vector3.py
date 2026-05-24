import math
import pytest
from linpy import Vector3


class TestVector3Creation:
    def test_create(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_zero(self):
        v = Vector3.zero()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_one(self):
        v = Vector3.one()
        assert v.x == 1.0 and v.y == 1.0 and v.z == 1.0

    def test_x_one(self):
        v = Vector3.x_one()
        assert v == Vector3(1.0, 0.0, 0.0)

    def test_y_one(self):
        v = Vector3.y_one()
        assert v == Vector3(0.0, 1.0, 0.0)

    def test_z_one(self):
        v = Vector3.z_one()
        assert v == Vector3(0.0, 0.0, 1.0)


class TestVector3Operators:
    def test_add_vector(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        result = a + b
        assert result == Vector3(5.0, 7.0, 9.0)

    def test_add_scalar(self):
        v = Vector3(1.0, 2.0, 3.0)
        result = v + 1.0
        assert result == Vector3(2.0, 3.0, 4.0)

    def test_sub_vector(self):
        a = Vector3(4.0, 5.0, 6.0)
        b = Vector3(1.0, 2.0, 3.0)
        result = a - b
        assert result == Vector3(3.0, 3.0, 3.0)

    def test_sub_scalar(self):
        v = Vector3(4.0, 5.0, 6.0)
        result = v - 1.0
        assert result == Vector3(3.0, 4.0, 5.0)

    def test_mul_vector(self):
        a = Vector3(2.0, 3.0, 4.0)
        b = Vector3(5.0, 6.0, 7.0)
        result = a * b
        assert result == Vector3(10.0, 18.0, 28.0)

    def test_mul_scalar(self):
        v = Vector3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result == Vector3(2.0, 4.0, 6.0)

    def test_rmul_scalar(self):
        v = Vector3(1.0, 2.0, 3.0)
        result = 2.0 * v
        assert result == Vector3(2.0, 4.0, 6.0)

    def test_div_vector(self):
        a = Vector3(10.0, 18.0, 28.0)
        b = Vector3(5.0, 6.0, 7.0)
        result = a / b
        assert result == Vector3(2.0, 3.0, 4.0)

    def test_div_scalar(self):
        v = Vector3(2.0, 4.0, 6.0)
        result = v / 2.0
        assert result == Vector3(1.0, 2.0, 3.0)

    def test_neg(self):
        v = Vector3(1.0, -2.0, 3.0)
        result = -v
        assert result == Vector3(-1.0, 2.0, -3.0)

    def test_eq(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0, 2.0, 3.0)
        assert a == b

    def test_ne(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0, 2.0, 4.0)
        assert a != b

    def test_eq_within_epsilon(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0 + 1e-10, 2.0, 3.0)
        assert a == b

    def test_ne_different_type(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v != "not a vector"

    def test_add_invalid_type_raises(self):
        v = Vector3(1.0, 2.0, 3.0)
        with pytest.raises(TypeError):
            v + "invalid"

    def test_mul_invalid_type_raises(self):
        v = Vector3(1.0, 2.0, 3.0)
        with pytest.raises(TypeError):
            v * "invalid"


class TestVector3Methods:
    def test_dot(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        assert a.dot(b) == pytest.approx(32.0)

    def test_magnitude(self):
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude() == pytest.approx(5.0)

    def test_magnitude_unit(self):
        v = Vector3(1.0, 0.0, 0.0)
        assert v.magnitude() == pytest.approx(1.0)

    def test_normalize_in_place(self):
        v = Vector3(3.0, 0.0, 0.0)
        v.normalize()
        assert v == Vector3(1.0, 0.0, 0.0)

    def test_normalized(self):
        v = Vector3(0.0, 5.0, 0.0)
        n = v.normalized()
        assert n == Vector3(0.0, 1.0, 0.0)
        # Original unchanged
        assert v == Vector3(0.0, 5.0, 0.0)

    def test_normalized_zero_vector(self):
        v = Vector3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert n == Vector3(0.0, 0.0, 0.0)

    def test_cross_product(self):
        x = Vector3(1.0, 0.0, 0.0)
        y = Vector3(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert z == Vector3(0.0, 0.0, 1.0)

    def test_cross_product_anticommutative(self):
        x = Vector3(1.0, 0.0, 0.0)
        y = Vector3(0.0, 1.0, 0.0)
        assert y.cross(x) == Vector3(0.0, 0.0, -1.0)

    def test_inverse(self):
        v = Vector3(1.0, -2.0, 3.0)
        assert v.inverse() == Vector3(-1.0, 2.0, -3.0)

    def test_lerp_start(self):
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        assert a.lerp(b, 0.0) == a

    def test_lerp_end(self):
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        assert a.lerp(b, 1.0) == b

    def test_lerp_mid(self):
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        assert a.lerp(b, 0.5) == Vector3(5.0, 5.0, 5.0)

    def test_distance(self):
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(3.0, 4.0, 0.0)
        assert a.distance(b) == pytest.approx(5.0)

    def test_to_list(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.to_list() == [1.0, 2.0, 3.0]


class TestVector3Misc:
    def test_len(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert len(v) == 3

    def test_iter(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert list(v) == [1.0, 2.0, 3.0]

    def test_str(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert "[1.0, 2.0, 3.0]" == str(v)

    def test_swizzle_xyz(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.xyz == Vector3(1.0, 2.0, 3.0)

    def test_swizzle_yzx(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.yzx == Vector3(2.0, 3.0, 1.0)

    def test_swizzle_zxy(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.zxy == Vector3(3.0, 1.0, 2.0)

    def test_swizzle_xxx(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.xxx == Vector3(1.0, 1.0, 1.0)
