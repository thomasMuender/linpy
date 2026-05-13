"""
Unit tests for vector math (Vector2, Vector3, Vector4).

Covers construction, arithmetic, swizzling, dot product, cross product,
magnitude, normalization, trigonometric helpers, conversions, and edge cases.
"""

import math
import pytest
import numpy as np

from linpy.vector import Vector, Vector2, Vector3, Vector4


# ------------------------------------------------------------------ helpers --

def assert_vec_equal(v, expected, tol=1e-7):
    """Assert each component of a vector approximately equals the expected list."""
    values = list(v)
    assert len(values) == len(expected), f"Length mismatch: {len(values)} vs {len(expected)}"
    for i, (a, b) in enumerate(zip(values, expected)):
        assert abs(a - b) < tol, f"Component {i}: {a} != {b} (diff={abs(a - b)})"


# ============================================================
# Construction
# ============================================================

class TestVectorConstruction:
    """Verify that Vector2/Vector3/Vector4 can be created from various input types
    and that invalid inputs are properly rejected."""

    def test_Vector2_from_two_scalars(self):
        # Two individual floats set x and y directly.
        v = Vector2(1.0, 2.0)
        assert v.x == 1.0 and v.y == 2.0

    def test_Vector3_from_three_scalars(self):
        # Three individual floats set x, y, z directly.
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_Vector4_from_four_scalars(self):
        # Four individual floats set x, y, z, w directly.
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0 and v.w == 4.0

    def test_single_scalar_broadcasts_to_all_components(self):
        # A single int/float argument fills every component with that value.
        assert_vec_equal(Vector2(5), [5, 5])
        assert_vec_equal(Vector3(3), [3, 3, 3])
        assert_vec_equal(Vector4(7), [7, 7, 7, 7])

    def test_from_list(self):
        # A Python list with the correct length should unpack into components.
        v = Vector3([1.0, 2.0, 3.0])
        assert_vec_equal(v, [1.0, 2.0, 3.0])

    def test_from_tuple(self):
        # Tuple input should behave identically to a list.
        v = Vector3((4.0, 5.0, 6.0))
        assert_vec_equal(v, [4.0, 5.0, 6.0])

    def test_from_numpy_array(self):
        # numpy arrays should be unpacked element-by-element.
        v = Vector3(np.array([7.0, 8.0, 9.0]))
        assert_vec_equal(v, [7.0, 8.0, 9.0])

    def test_from_another_vector(self):
        # Constructing from an existing vector copies its values.
        orig = Vector3(1.0, 2.0, 3.0)
        copy = Vector3(orig)
        assert_vec_equal(copy, [1.0, 2.0, 3.0])

    def test_mixed_Vector3_and_scalar_to_Vector4(self):
        # Vector4 can be built from a Vector3 (xyz) plus a trailing scalar (w).
        v4 = Vector4(Vector3(1.0, 2.0, 3.0), 4.0)
        assert_vec_equal(v4, [1.0, 2.0, 3.0, 4.0])

    def test_mixed_Vector2_and_two_scalars_to_Vector4(self):
        # Vector4 from a Vector2 and two additional scalars.
        v4 = Vector4(Vector2(1.0, 2.0), 3.0, 4.0)
        assert_vec_equal(v4, [1.0, 2.0, 3.0, 4.0])

    def test_two_Vector2s_to_Vector4(self):
        # Vector4 from two Vector2s concatenated.
        v4 = Vector4(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
        assert_vec_equal(v4, [1.0, 2.0, 3.0, 4.0])

    def test_wrong_component_count_too_many_raises(self):
        # More components than the vector size should raise AttributeError.
        with pytest.raises(AttributeError):
            Vector2(1.0, 2.0, 3.0)

    def test_wrong_component_count_too_few_raises(self):
        # Fewer components (from a list) than needed should raise AttributeError.
        with pytest.raises(AttributeError):
            Vector3([1.0, 2.0])

    def test_no_args_raises(self):
        # Zero arguments gives zero values => length mismatch => AttributeError.
        with pytest.raises(AttributeError):
            Vector2()

    def test_unsupported_type_raises(self):
        # Non-numeric, non-iterable types should raise NotImplementedError.
        with pytest.raises(NotImplementedError):
            Vector3("abc")

    def test_base_vector_class_raises(self):
        # Instantiating the abstract Vector base directly is forbidden.
        with pytest.raises(ValueError):
            Vector(1.0, 2.0, 3.0)

    def test_integer_inputs_converted_to_float(self):
        # All stored values must be floats regardless of input type.
        v = Vector3(1, 2, 3)
        assert all(isinstance(c, float) for c in v)


# ============================================================
# Arithmetic Operations
# ============================================================

class TestVectorArithmetic:
    """Verify element-wise and scalar arithmetic on vectors."""

    def test_add_same_type(self):
        # Element-wise addition of two Vector3s.
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        assert_vec_equal(a + b, [5.0, 7.0, 9.0])

    def test_add_scalar(self):
        # Adding a scalar adds it to every component.
        assert_vec_equal(Vector3(1, 2, 3) + 10, [11, 12, 13])

    def test_radd_scalar(self):
        # scalar + vector uses __radd__ and should give the same result.
        assert_vec_equal(10 + Vector3(1, 2, 3), [11, 12, 13])

    def test_sub_same_type(self):
        # Element-wise subtraction.
        assert_vec_equal(Vector3(5, 7, 9) - Vector3(1, 2, 3), [4, 5, 6])

    def test_sub_scalar(self):
        # Subtracting a scalar from every component.
        assert_vec_equal(Vector3(10, 20, 30) - 5, [5, 15, 25])

    def test_mul_same_type(self):
        # Element-wise (Hadamard) multiplication.
        assert_vec_equal(Vector3(2, 3, 4) * Vector3(5, 6, 7), [10, 18, 28])

    def test_mul_scalar(self):
        # Scalar multiplication scales every component.
        assert_vec_equal(Vector3(1, 2, 3) * 3, [3, 6, 9])

    def test_rmul_scalar(self):
        # scalar * vector should also work via __rmul__.
        assert_vec_equal(3 * Vector3(1, 2, 3), [3, 6, 9])

    def test_div_same_type(self):
        # Element-wise division.
        assert_vec_equal(Vector3(10, 20, 30) / Vector3(2, 4, 5), [5, 5, 6])

    def test_div_scalar(self):
        # Division by a scalar divides each component.
        assert_vec_equal(Vector3(10, 20, 30) / 10, [1, 2, 3])

    def test_add_mismatched_types_raises(self):
        # Adding vectors of different sizes is not allowed.
        with pytest.raises(TypeError):
            Vector2(1, 2) + Vector3(1, 2, 3)

    def test_mul_mismatched_types_raises(self):
        # Multiplying vectors of different sizes is not allowed.
        with pytest.raises(TypeError):
            Vector2(1, 2) * Vector3(1, 2, 3)

    def test_div_by_zero_scalar_raises(self):
        # Division by zero should propagate ZeroDivisionError.
        with pytest.raises(ZeroDivisionError):
            Vector3(1, 2, 3) / 0

    def test_div_by_zero_component_raises(self):
        # Element-wise division where any denominator is zero should raise.
        with pytest.raises(ZeroDivisionError):
            Vector3(1, 2, 3) / Vector3(1, 0, 1)

    def test_add_preserves_type(self):
        # The result of addition should be the same vector type.
        result = Vector2(1, 2) + Vector2(3, 4)
        assert isinstance(result, Vector2)

    def test_arithmetic_does_not_mutate_operands(self):
        # Binary operations should return new vectors without modifying originals.
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        _ = a + b
        assert_vec_equal(a, [1, 2, 3])
        assert_vec_equal(b, [4, 5, 6])


# ============================================================
# Swizzling (GLSL-style component access / rearrangement)
# ============================================================

class TestVectorSwizzling:
    """Verify reading and writing components by name (e.g. v.xy, v.yzx)."""

    def test_single_component_read(self):
        # Named single-component access returns a float.
        v = Vector3(10.0, 20.0, 30.0)
        assert v.x == 10.0
        assert v.y == 20.0
        assert v.z == 30.0

    def test_two_component_swizzle_returns_Vector2(self):
        # Two-character swizzle returns a Vector2.
        v = Vector3(1, 2, 3)
        result = v.xy
        assert isinstance(result, Vector2)
        assert_vec_equal(result, [1, 2])

    def test_three_component_swizzle_returns_Vector3(self):
        # Three-character swizzle returns a Vector3.
        v = Vector4(1, 2, 3, 4)
        result = v.zyx
        assert isinstance(result, Vector3)
        assert_vec_equal(result, [3, 2, 1])

    def test_four_component_swizzle_returns_Vector4(self):
        # Four-character swizzle on Vector4 returns a Vector4 with reordered components.
        v = Vector4(1, 2, 3, 4)
        result = v.wzyx
        assert isinstance(result, Vector4)
        assert_vec_equal(result, [4, 3, 2, 1])

    def test_repeated_component_in_swizzle(self):
        # Swizzle may repeat the same component (e.g. v.xxx).
        v = Vector3(5, 10, 15)
        assert_vec_equal(v.xxx, [5, 5, 5])

    def test_yzx_cyclic_permutation(self):
        # yzx is used internally for the cross product; verify the cyclic shift.
        v = Vector3(1, 2, 3)
        assert_vec_equal(v.yzx, [2, 3, 1])

    def test_swizzle_set_single_component(self):
        # Writing a single named component should update only that component.
        v = Vector3(1, 2, 3)
        v.x = 99.0
        assert v.x == 99.0
        assert v.y == 2.0 and v.z == 3.0

    def test_swizzle_set_multiple_components(self):
        # Writing a multi-character swizzle updates those components.
        v = Vector3(1, 2, 3)
        v.xy = Vector2(10, 20)
        assert_vec_equal(v, [10, 20, 3])

    def test_swizzle_set_broadcast_scalar(self):
        # Setting multiple components to a single scalar broadcasts.
        v = Vector3(1, 2, 3)
        v.xy = 0
        assert_vec_equal(v, [0, 0, 3])

    def test_out_of_bounds_swizzle_raises(self):
        # Accessing z on a Vector2 is out of range (Vector2 has only x,y).
        # Raises IndexError because name_to_idx returns index 2 which is
        # beyond the internal list length.
        v = Vector2(1, 2)
        with pytest.raises(IndexError):
            _ = v.z

    def test_swizzle_set_with_duplicate_chars_raises(self):
        # Duplicate characters in a set-swizzle (e.g. v.xx = ...) are ambiguous
        # and the implementation rejects them.
        v = Vector3(1, 2, 3)
        with pytest.raises(IndexError):
            v.xx = Vector2(1, 2)

    def test_w_component_on_Vector4(self):
        # Vector4 supports the w component.
        v = Vector4(1, 2, 3, 4)
        assert v.w == 4.0


# ============================================================
# Indexing
# ============================================================

class TestVectorIndexing:
    """Verify integer indexing and slicing on vectors."""

    def test_single_index(self):
        # Integer index returns a single float.
        v = Vector3(10, 20, 30)
        assert v[0] == 10.0
        assert v[1] == 20.0
        assert v[2] == 30.0

    def test_negative_index(self):
        # Negative indexing wraps around as in standard Python.
        v = Vector3(10, 20, 30)
        assert v[-1] == 30.0

    def test_slice_to_Vector3(self):
        # Slicing a Vector4 to 3 elements returns a Vector3.
        v = Vector4(1, 2, 3, 4)
        result = v[:3]
        assert isinstance(result, Vector3)
        assert_vec_equal(result, [1, 2, 3])

    def test_slice_to_Vector2(self):
        # Slicing to 2 elements returns a Vector2.
        v = Vector3(1, 2, 3)
        result = v[:2]
        assert isinstance(result, Vector2)
        assert_vec_equal(result, [1, 2])

    def test_setitem(self):
        # Setting a component via index.
        v = Vector3(1, 2, 3)
        v[1] = 99.0
        assert v[1] == 99.0

    def test_len(self):
        # len() returns the number of components.
        assert len(Vector2(0, 0)) == 2
        assert len(Vector3(0, 0, 0)) == 3
        assert len(Vector4(0, 0, 0, 0)) == 4

    def test_iter(self):
        # Iterating yields all components in order.
        v = Vector3(1, 2, 3)
        assert list(v) == [1.0, 2.0, 3.0]


# ============================================================
# Dot Product
# ============================================================

class TestDotProduct:
    """Verify the dot (inner) product on vectors."""

    def test_orthogonal_vectors_yield_zero(self):
        # Perpendicular vectors have dot product zero.
        assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == pytest.approx(0.0)

    def test_parallel_unit_vectors_yield_one(self):
        # Aligned unit vectors have dot product 1.
        assert Vector3(1, 0, 0).dot(Vector3(1, 0, 0)) == pytest.approx(1.0)

    def test_anti_parallel_unit_vectors_yield_negative_one(self):
        # Opposite unit vectors have dot product -1.
        assert Vector3(1, 0, 0).dot(Vector3(-1, 0, 0)) == pytest.approx(-1.0)

    def test_self_dot_equals_magnitude_squared(self):
        # v · v = |v|² ; for (3,4,0) that is 25.
        v = Vector3(3, 4, 0)
        assert v.dot(v) == pytest.approx(25.0)

    def test_known_value(self):
        # (1,2,3) · (4,5,6) = 4 + 10 + 18 = 32.
        assert Vector3(1, 2, 3).dot(Vector3(4, 5, 6)) == pytest.approx(32.0)

    def test_commutative(self):
        # Dot product must be commutative: a · b == b · a.
        a, b = Vector3(1, 2, 3), Vector3(4, 5, 6)
        assert a.dot(b) == pytest.approx(b.dot(a))

    def test_dot_Vector2(self):
        # Dot product should also work on Vector2.
        assert Vector2(3, 4).dot(Vector2(4, 3)) == pytest.approx(24.0)


# ============================================================
# Magnitude & Normalization
# ============================================================

class TestMagnitudeAndNormalization:
    """Verify length calculation and normalization behaviour."""

    def test_unit_vector_magnitude_is_one(self):
        # A basis vector has magnitude 1.
        assert Vector3(1, 0, 0).magnitude() == pytest.approx(1.0)

    def test_known_magnitude(self):
        # |[3,4,0]| = 5 (Pythagorean triple).
        assert Vector3(3, 4, 0).magnitude() == pytest.approx(5.0)

    def test_zero_vector_magnitude_is_zero(self):
        # The zero vector has length 0.
        assert Vector3(0, 0, 0).magnitude() == pytest.approx(0.0)

    def test_normalize_in_place(self):
        # normalize() modifies the vector so its magnitude becomes 1
        # while preserving direction.
        v = Vector3(3, 4, 0)
        v.normalize()
        assert v.magnitude() == pytest.approx(1.0)
        assert_vec_equal(v, [0.6, 0.8, 0.0])

    def test_normalized_returns_new_vector(self):
        # normalized() returns a new unit-length vector; original is unchanged.
        v = Vector3(3, 4, 0)
        n = v.normalized()
        assert n.magnitude() == pytest.approx(1.0)
        assert_vec_equal(v, [3, 4, 0])  # original unchanged

    def test_normalize_already_unit(self):
        # Normalizing a unit vector should leave it unchanged.
        n = Vector3(0, 1, 0).normalized()
        assert_vec_equal(n, [0, 1, 0])

    def test_normalize_zero_vector_raises(self):
        # Normalizing a zero vector requires dividing by zero → ZeroDivisionError.
        with pytest.raises(ZeroDivisionError):
            Vector3(0, 0, 0).normalize()


# ============================================================
# Inverse (negation)
# ============================================================

class TestVectorInverse:
    """Verify v.inverse() negates all components."""

    def test_negates_all_components(self):
        # inverse() multiplies by -1.
        assert_vec_equal(Vector3(1, -2, 3).inverse(), [-1, 2, -3])

    def test_inverse_of_zero(self):
        # -0 is still 0 for each component.
        assert_vec_equal(Vector3(0, 0, 0).inverse(), [0, 0, 0])

    def test_double_inverse_restores_original(self):
        # Negating twice returns the original vector.
        v = Vector3(1, 2, 3)
        assert_vec_equal(v.inverse().inverse(), [1, 2, 3])


# ============================================================
# Element-wise Trig & Angle Conversion
# ============================================================

class TestVectorTrigAndConversion:
    """Verify element-wise sin/cos/degree/radians and format conversions."""

    def test_sin(self):
        # sin(0) = 0, sin(π/2) = 1.
        v = Vector2(0.0, math.pi / 2).sin()
        assert v.x == pytest.approx(0.0)
        assert v.y == pytest.approx(1.0)

    def test_cos(self):
        # cos(0) = 1, cos(π) = -1.
        v = Vector2(0.0, math.pi).cos()
        assert v.x == pytest.approx(1.0)
        assert v.y == pytest.approx(-1.0)

    def test_degree_conversion(self):
        # π radians → 180°, π/2 → 90°.
        v = Vector2(math.pi, math.pi / 2).degree()
        assert v.x == pytest.approx(180.0)
        assert v.y == pytest.approx(90.0)

    def test_radians_conversion(self):
        # 180° → π, 90° → π/2.
        v = Vector2(180, 90).radians()
        assert v.x == pytest.approx(math.pi)
        assert v.y == pytest.approx(math.pi / 2)

    def test_toList(self):
        # toList() must return a plain Python list.
        assert Vector3(1, 2, 3).to_list() == [1.0, 2.0, 3.0]

    def test_toNumpy(self):
        # toNumpy() must return a numpy ndarray with matching values.
        arr = Vector3(1, 2, 3).to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])


# ============================================================
# Cross Product (Vector3 only)
# ============================================================

class TestCrossProduct:
    """Verify the 3D cross product on Vector3."""

    def test_x_cross_y_equals_z(self):
        # Standard right-hand rule: x × y = z.
        assert_vec_equal(Vector3(1, 0, 0).cross(Vector3(0, 1, 0)), [0, 0, 1])

    def test_y_cross_z_equals_x(self):
        # y × z = x.
        assert_vec_equal(Vector3(0, 1, 0).cross(Vector3(0, 0, 1)), [1, 0, 0])

    def test_z_cross_x_equals_y(self):
        # z × x = y.
        assert_vec_equal(Vector3(0, 0, 1).cross(Vector3(1, 0, 0)), [0, 1, 0])

    def test_anti_commutative(self):
        # a × b = -(b × a).
        a, b = Vector3(1, 2, 3), Vector3(4, 5, 6)
        ab = a.cross(b)
        ba = b.cross(a)
        assert_vec_equal(ab, [-v for v in ba])

    def test_parallel_vectors_give_zero(self):
        # Parallel (same direction) vectors have zero cross product.
        assert_vec_equal(Vector3(2, 0, 0).cross(Vector3(5, 0, 0)), [0, 0, 0])

    def test_self_cross_is_zero(self):
        # Any vector crossed with itself is the zero vector.
        v = Vector3(1, 2, 3)
        assert_vec_equal(v.cross(v), [0, 0, 0])

    def test_result_perpendicular_to_both_inputs(self):
        # The cross product must be orthogonal to both operands.
        a, b = Vector3(1, 2, 3), Vector3(4, 5, 6)
        c = a.cross(b)
        assert c.dot(a) == pytest.approx(0.0)
        assert c.dot(b) == pytest.approx(0.0)

    def test_known_cross_product(self):
        # (1,2,3) × (4,5,6) = (2·6-3·5, 3·4-1·6, 1·5-2·4) = (-3, 6, -3).
        assert_vec_equal(Vector3(1, 2, 3).cross(Vector3(4, 5, 6)), [-3, 6, -3])

    def test_cross_product_magnitude_equals_area(self):
        # |a × b| = |a|·|b|·sin(θ).
        # For perpendicular unit vectors the magnitude should be 1.
        c = Vector3(1, 0, 0).cross(Vector3(0, 1, 0))
        assert c.magnitude() == pytest.approx(1.0)


# ============================================================
# Constant Properties
# ============================================================

class TestVectorConstants:
    """Verify convenience constant-vector properties (zero, one, axis units)."""

    def test_Vector3_zero(self):
        # zero returns the origin vector.
        assert_vec_equal(Vector3(1, 1, 1).zero(), [0, 0, 0])

    def test_Vector3_one(self):
        # one returns the all-ones vector.
        assert_vec_equal(Vector3(0, 0, 0).one(), [1, 1, 1])

    def test_Vector3_axis_unit_vectors(self):
        # x_one, y_one, z_one are the standard basis vectors.
        v = Vector3(0, 0, 0)
        assert_vec_equal(v.x_one(), [1, 0, 0])
        assert_vec_equal(v.y_one(), [0, 1, 0])
        assert_vec_equal(v.z_one(), [0, 0, 1])

    def test_Vector4_w_one(self):
        # Vector4 additionally provides w_one.
        assert_vec_equal(Vector4(0, 0, 0, 0).w_one(), [0, 0, 0, 1])


# ============================================================
# String Representation
# ============================================================

class TestVectorStr:
    """Verify __str__ output format."""

    def test_str_format(self):
        # Should produce bracket-enclosed, comma-separated values.
        assert str(Vector3(1.0, 2.0, 3.0)) == "[1.0, 2.0, 3.0]"


# ============================================================
# Edge Cases
# ============================================================

class TestVectorEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_very_large_values(self):
        # Arithmetic should work with very large floats.
        v = Vector3(1e15, 1e15, 1e15)
        assert_vec_equal(v + v, [2e15, 2e15, 2e15])

    def test_very_small_values(self):
        # Arithmetic should work with very small floats.
        v = Vector3(1e-15, 1e-15, 1e-15)
        assert_vec_equal(v * 2, [2e-15, 2e-15, 2e-15], tol=1e-30)

    def test_negative_zero_magnitude(self):
        # Negative zero is still zero; magnitude should be 0.
        v = Vector3(-0.0, -0.0, -0.0)
        assert v.magnitude() == pytest.approx(0.0)

    def test_mixed_positive_negative_magnitude(self):
        # Magnitude with mixed signs: |(-1,2,-3)| = √14.
        assert Vector3(-1, 2, -3).magnitude() == pytest.approx(math.sqrt(14))

    def test_normalize_negative_components(self):
        # Normalizing a vector with negative components should keep direction.
        v = Vector3(-3, 0, 4)
        n = v.normalized()
        assert n.magnitude() == pytest.approx(1.0)
        # Direction: (-3/5, 0, 4/5)
        assert_vec_equal(n, [-0.6, 0.0, 0.8])

    def test_scalar_operations_with_int_and_float(self):
        # Both int and float scalars should work in arithmetic.
        v = Vector3(1, 2, 3)
        assert_vec_equal(v * 2, [2, 4, 6])
        assert_vec_equal(v * 2.0, [2, 4, 6])

    def test_copy_independence(self):
        # Constructing a new vector from an existing one should create independent data.
        a = Vector3(1, 2, 3)
        b = Vector3(a)
        b.x = 99
        assert a.x == 1.0  # original must be unaffected
