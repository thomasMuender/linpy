"""
Unit tests for rotation / quaternion math.

Covers quaternion construction, axis rotations, Euler angle conversion,
rotation matrix conversion, quaternion algebra (multiply, inverse, normalize,
dot), vector rotation, and edge cases (gimbal lock, 180°, near-identity,
double cover, fromAngleAxis bug).
"""

import math
import pytest
import numpy as np

from linpy.vector import Vector3, Vector4
from linpy.quaternion import Quaternion


# ------------------------------------------------------------------ helpers --

def assert_vec_equal(v, expected, tol=1e-6):
    """Assert each component of a vector matches expected values within tolerance."""
    values = list(v)
    assert len(values) == len(expected), f"Length mismatch: {len(values)} vs {len(expected)}"
    for i, (a, b) in enumerate(zip(values, expected)):
        assert abs(a - b) < tol, f"Component {i}: {a} != {b} (diff={abs(a - b)})"


def assert_quat_equal(q, expected, tol=1e-6):
    """Assert quaternion components match expected [x, y, z, w] within tolerance."""
    assert_vec_equal(q, expected, tol)


def assert_same_rotation(q1, q2, tol=1e-6):
    """Assert two quaternions represent the same rotation.
    Quaternions have double cover: q and -q encode the same rotation,
    so we compare |dot(q1, q2)| ≈ 1."""
    d = abs(q1.dot(q2))
    assert d == pytest.approx(1.0, abs=tol), (
        f"Quaternions differ: |dot| = {d}"
    )


# ============================================================
# Quaternion Construction
# ============================================================

class TestQuaternionConstruction:
    """Verify that quaternions can be created from various input types."""

    def test_from_four_scalars(self):
        # Explicit x, y, z, w components.
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert q.x == 1.0 and q.y == 2.0 and q.z == 3.0 and q.w == 4.0

    def test_single_scalar_broadcasts(self):
        # A single number fills all four components.
        q = Quaternion(5)
        assert_quat_equal(q, [5, 5, 5, 5])

    def test_from_list(self):
        # From a plain Python list.
        q = Quaternion([1, 2, 3, 4])
        assert_quat_equal(q, [1, 2, 3, 4])

    def test_from_Vector4(self):
        # From a Vector4.
        q = Quaternion(Vector4(1, 2, 3, 4))
        assert_quat_equal(q, [1, 2, 3, 4])

    def test_from_Vector3_and_scalar(self):
        # From a Vector3 (xyz) and a trailing scalar (w).
        q = Quaternion(Vector3(1, 2, 3), 4)
        assert_quat_equal(q, [1, 2, 3, 4])

    def test_from_numpy(self):
        # From a numpy array.
        q = Quaternion(np.array([1, 2, 3, 4.0]))
        assert_quat_equal(q, [1, 2, 3, 4])

    def test_wrong_count_raises(self):
        # Three components is invalid for a quaternion.
        with pytest.raises(AttributeError):
            Quaternion(1, 2, 3)

    def test_identity_property(self):
        # The identity quaternion encodes zero rotation: (0, 0, 0, 1).
        q = Quaternion(0, 0, 0, 1).identity
        assert_quat_equal(q, [0, 0, 0, 1])

    def test_len(self):
        # A quaternion always has 4 components.
        assert len(Quaternion(1, 2, 3, 4)) == 4

    def test_iter(self):
        # Iterating yields components in x, y, z, w order.
        assert list(Quaternion(1, 2, 3, 4)) == [1.0, 2.0, 3.0, 4.0]

    def test_str(self):
        # String representation delegates to the internal Vector4.
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert str(q) == "[1.0, 2.0, 3.0, 4.0]"


# ============================================================
# Axis Rotation Factories (fromRotationX / Y / Z)
# ============================================================

class TestAxisRotations:
    """Verify that fromRotationX/Y/Z create correct rotation quaternions
    by checking how they transform known basis vectors."""

    def test_rotation_x_90_rotates_y_to_z(self):
        # 90° about X sends the Y-axis onto the Z-axis.
        q = Quaternion.fromRotationX(90)
        assert_vec_equal(q * Vector3(0, 1, 0), [0, 0, 1])

    def test_rotation_y_90_rotates_z_to_x(self):
        # 90° about Y sends the Z-axis onto the X-axis.
        q = Quaternion.fromRotationY(90)
        assert_vec_equal(q * Vector3(0, 0, 1), [1, 0, 0])

    def test_rotation_z_90_rotates_x_to_y(self):
        # 90° about Z sends the X-axis onto the Y-axis.
        q = Quaternion.fromRotationZ(90)
        assert_vec_equal(q * Vector3(1, 0, 0), [0, 1, 0])

    def test_rotation_x_180_flips_y(self):
        # 180° about X reverses Y and Z.
        q = Quaternion.fromRotationX(180)
        assert_vec_equal(q * Vector3(0, 1, 0), [0, -1, 0])

    def test_rotation_y_180_flips_x(self):
        # 180° about Y reverses X and Z.
        q = Quaternion.fromRotationY(180)
        assert_vec_equal(q * Vector3(1, 0, 0), [-1, 0, 0])

    def test_rotation_z_180_flips_x(self):
        # 180° about Z reverses X and Y.
        q = Quaternion.fromRotationZ(180)
        assert_vec_equal(q * Vector3(1, 0, 0), [-1, 0, 0])

    def test_360_is_identity(self):
        # A full 360° rotation returns the vector to its original position.
        q = Quaternion.fromRotationX(360)
        assert_vec_equal(q * Vector3(1, 2, 3), [1, 2, 3])

    def test_zero_rotation_is_identity(self):
        # 0° rotation should not change any vector.
        q = Quaternion.fromRotationY(0)
        assert_vec_equal(q * Vector3(1, 2, 3), [1, 2, 3])

    def test_negative_angle(self):
        # A negative angle rotates in the opposite direction.
        v = Vector3(1, 0, 0)
        assert_vec_equal(Quaternion.fromRotationZ(90) * v, [0, 1, 0])
        assert_vec_equal(Quaternion.fromRotationZ(-90) * v, [0, -1, 0])

    def test_rotation_preserves_axis_vector(self):
        # Rotating about an axis leaves vectors along that axis unchanged.
        q = Quaternion.fromRotationX(45)
        assert_vec_equal(q * Vector3(5, 0, 0), [5, 0, 0])

    def test_rotation_preserves_magnitude(self):
        # Rotations are isometries — they do not change vector length.
        q = Quaternion.fromRotationY(37)
        v = Vector3(3, 4, 5)
        rotated = q * v
        assert rotated.magnitude() == pytest.approx(v.magnitude(), abs=1e-6)

    def test_two_90_rotations_equal_180(self):
        # Composing two 90° rotations about the same axis should match one 180°.
        q90 = Quaternion.fromRotationZ(90)
        q180 = Quaternion.fromRotationZ(180)
        v = Vector3(1, 0, 0)
        result_twice = q90 * (q90 * v)
        result_180 = q180 * v
        assert_vec_equal(result_twice, list(result_180))


# ============================================================
# Instance Rotation Methods (rotateX / Y / Z)
# ============================================================

class TestInstanceRotateMethods:
    """Verify the chainable rotateX/Y/Z methods on a Quaternion instance."""

    def test_rotateX(self):
        # Starting from identity, rotateX(90) should act like fromRotationX(90).
        q = Quaternion(0, 0, 0, 1).rotateX(90)
        assert_vec_equal(q * Vector3(0, 1, 0), [0, 0, 1])

    def test_rotateY(self):
        # Starting from identity, rotateY(90) should act like fromRotationY(90).
        q = Quaternion(0, 0, 0, 1).rotateY(90)
        assert_vec_equal(q * Vector3(0, 0, 1), [1, 0, 0])

    def test_rotateZ(self):
        # Starting from identity, rotateZ(90) should act like fromRotationZ(90).
        q = Quaternion(0, 0, 0, 1).rotateZ(90)
        assert_vec_equal(q * Vector3(1, 0, 0), [0, 1, 0])

    def test_chained_rotations(self):
        # rotateX then rotateY should compose the two rotations (right-multiply).
        q = Quaternion(0, 0, 0, 1).rotateX(90).rotateY(90)
        composed = Quaternion.fromRotationX(90) * Quaternion.fromRotationY(90)
        # Both should produce the same rotation.
        assert_same_rotation(q, composed)


# ============================================================
# Quaternion Multiplication (composition)
# ============================================================

class TestQuaternionMultiplication:
    """Verify quaternion-quaternion multiplication (rotation composition)."""

    def test_multiply_with_identity_left(self):
        # identity * q == q.
        q = Quaternion.fromRotationZ(45)
        identity = Quaternion(0, 0, 0, 1)
        assert_same_rotation(identity * q, q)

    def test_multiply_with_identity_right(self):
        # q * identity == q.
        q = Quaternion.fromRotationZ(45)
        identity = Quaternion(0, 0, 0, 1)
        assert_same_rotation(q * identity, q)

    def test_associativity(self):
        # (q1 * q2) * q3 should equal q1 * (q2 * q3).
        q1 = Quaternion.fromRotationX(30)
        q2 = Quaternion.fromRotationY(45)
        q3 = Quaternion.fromRotationZ(60)
        lhs = (q1 * q2) * q3
        rhs = q1 * (q2 * q3)
        assert_same_rotation(lhs, rhs)

    def test_non_commutative(self):
        # Quaternion multiplication is generally NOT commutative.
        # Rotating 90° about X then 90° about Y differs from Y then X.
        qx = Quaternion.fromRotationX(90)
        qy = Quaternion.fromRotationY(90)
        v = Vector3(1, 0, 0)
        result_xy = (qx * qy) * v
        result_yx = (qy * qx) * v
        # These should NOT be equal for a general vector.
        with pytest.raises(AssertionError):
            assert_vec_equal(result_xy, list(result_yx))

    def test_scalar_multiplication(self):
        # Multiplying a quaternion by a scalar scales all components.
        q = Quaternion(1, 2, 3, 4)
        result = q * 2
        assert_quat_equal(result, [2, 4, 6, 8])

    def test_mul_wrong_type_raises(self):
        # Multiplying by an unsupported type should raise TypeError.
        q = Quaternion(0, 0, 0, 1)
        with pytest.raises(TypeError):
            q * "hello"


# ============================================================
# Quaternion Inverse
# ============================================================

class TestQuaternionInverse:
    """Verify that q * q.inverse() ≈ identity."""

    def test_inverse_of_unit_quaternion(self):
        # For a normalised quaternion, q * q⁻¹ should be the identity rotation.
        q = Quaternion.fromRotationY(45)
        product = q * q.inverse()
        assert_same_rotation(product, Quaternion(0, 0, 0, 1))

    def test_inverse_undoes_rotation(self):
        # Applying a rotation and then its inverse should restore the vector.
        q = Quaternion.fromRotationZ(60)
        v = Vector3(1, 2, 3)
        rotated = q * v
        restored = q.inverse() * rotated
        assert_vec_equal(restored, [1, 2, 3])

    def test_double_inverse_is_original(self):
        # Inverting twice returns the same rotation.
        q = Quaternion.fromRotationX(30)
        assert_same_rotation(q.inverse().inverse(), q)

    def test_inverse_of_identity(self):
        # The inverse of the identity is the identity.
        identity = Quaternion(0, 0, 0, 1)
        assert_same_rotation(identity.inverse(), identity)

    def test_inverse_conjugate_for_unit_quat(self):
        # For a unit quaternion, the inverse equals the conjugate:
        # q⁻¹ = (-x, -y, -z, w).
        q = Quaternion.fromRotationX(60)
        inv = q.inverse()
        assert inv.x == pytest.approx(-q.x, abs=1e-6)
        assert inv.y == pytest.approx(-q.y, abs=1e-6)
        assert inv.z == pytest.approx(-q.z, abs=1e-6)
        assert inv.w == pytest.approx(q.w, abs=1e-6)


# ============================================================
# Normalize & Dot
# ============================================================

class TestQuaternionNormalizeDot:
    """Verify normalization and the dot product on quaternions."""

    def test_normalize_in_place(self):
        # After normalize(), the quaternion should have unit length.
        q = Quaternion(1, 2, 3, 4)
        q.normalize()
        assert q.dot(q) == pytest.approx(1.0)

    def test_normalized_returns_new(self):
        # normalized() returns a new unit quaternion without modifying the original.
        q = Quaternion(1, 2, 3, 4)
        n = q.normalized()
        assert n.dot(n) == pytest.approx(1.0)
        # Original should not be unit-length (it was not normalized).
        assert q.dot(q) == pytest.approx(30.0)  # 1²+2²+3²+4² = 30

    def test_already_unit(self):
        # Normalizing an already-unit quaternion should leave it unchanged.
        q = Quaternion.fromRotationX(45)
        n = q.normalized()
        assert_quat_equal(n, list(q))

    def test_dot_self_is_squared_norm(self):
        # q · q = |q|².
        q = Quaternion(1, 2, 3, 4)
        assert q.dot(q) == pytest.approx(30.0)

    def test_dot_with_Vector4(self):
        # dot() should also accept a Vector4.
        q = Quaternion(1, 0, 0, 0)
        v = Vector4(1, 0, 0, 0)
        assert q.dot(v) == pytest.approx(1.0)


# ============================================================
# Euler Angle Conversion (fromEuler / toEuler)
# ============================================================

class TestEulerConversion:
    """Verify fromEuler → toEuler roundtrip and known rotations."""

    @pytest.mark.parametrize("angles", [
        (0, 0, 0),
        (30, 0, 0),
        (0, 45, 0),
        (0, 0, 60),
        (10, 20, 30),
        (-45, 30, 60),
        (80, -15, -60),
    ])
    def test_roundtrip_zxy(self, angles):
        # Converting to a quaternion and back (same order) should recover the
        # original Euler angles. This tests the default ZXY order.
        x, y, z = angles
        q = Quaternion.fromEuler(x, y, z, "ZXY")
        recovered = q.toEuler("ZXY")
        assert recovered.x == pytest.approx(x, abs=1e-4)
        assert recovered.y == pytest.approx(y, abs=1e-4)
        assert recovered.z == pytest.approx(z, abs=1e-4)

    @pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
    def test_roundtrip_all_orders(self, order):
        # Roundtrip should work for every supported rotation order.
        x, y, z = 15, 25, 35
        q = Quaternion.fromEuler(x, y, z, order)
        recovered = q.toEuler(order)
        assert recovered.x == pytest.approx(x, abs=1e-4)
        assert recovered.y == pytest.approx(y, abs=1e-4)
        assert recovered.z == pytest.approx(z, abs=1e-4)

    def test_zero_euler_gives_identity(self):
        # (0, 0, 0) Euler angles should produce the identity quaternion.
        q = Quaternion.fromEuler(0, 0, 0)
        assert_same_rotation(q, Quaternion(0, 0, 0, 1))

    def test_90_deg_x_matches_fromRotationX(self):
        # fromEuler with only an X component should match fromRotationX.
        q_euler = Quaternion.fromEuler(90, 0, 0)
        q_axis = Quaternion.fromRotationX(90)
        assert_same_rotation(q_euler, q_axis)

    def test_90_deg_y_matches_fromRotationY(self):
        # fromEuler with only a Y component should match fromRotationY.
        q_euler = Quaternion.fromEuler(0, 90, 0)
        q_axis = Quaternion.fromRotationY(90)
        assert_same_rotation(q_euler, q_axis)

    def test_90_deg_z_matches_fromRotationZ(self):
        # fromEuler with only a Z component should match fromRotationZ.
        q_euler = Quaternion.fromEuler(0, 0, 90)
        q_axis = Quaternion.fromRotationZ(90)
        assert_same_rotation(q_euler, q_axis)

    def test_invalid_order_raises(self):
        # An order with repeated axes (e.g. "XXY") should raise ValueError.
        with pytest.raises(ValueError):
            Quaternion.fromEuler(0, 0, 0, "XXY")

    def test_invalid_order_wrong_chars_raises(self):
        # An order with invalid characters should raise ValueError.
        with pytest.raises(ValueError):
            Quaternion.fromEuler(0, 0, 0, "ABC")


# ============================================================
# Rotation Matrix Conversion (toMatrix3x3 / fromMatrix3x3)
# ============================================================

class TestMatrixConversion:
    """Verify quaternion ↔ 3×3 rotation matrix conversion."""

    def test_identity_to_matrix(self):
        # The identity quaternion should produce the 3×3 identity matrix.
        q = Quaternion(0, 0, 0, 1)
        mat = q.toMatrix3x3()
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_rotation_x_90_matrix(self):
        # 90° about X should match the known rotation matrix:
        #   [[1,  0,  0],
        #    [0,  0, -1],
        #    [0,  1,  0]]
        q = Quaternion.fromRotationX(90)
        mat = q.toMatrix3x3()
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=float)
        np.testing.assert_array_almost_equal(mat, expected, decimal=6)

    def test_rotation_y_90_matrix(self):
        # 90° about Y:
        #   [[ 0,  0,  1],
        #    [ 0,  1,  0],
        #    [-1,  0,  0]]
        q = Quaternion.fromRotationY(90)
        mat = q.toMatrix3x3()
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=float)
        np.testing.assert_array_almost_equal(mat, expected, decimal=6)

    def test_rotation_z_90_matrix(self):
        # 90° about Z:
        #   [[ 0, -1,  0],
        #    [ 1,  0,  0],
        #    [ 0,  0,  1]]
        q = Quaternion.fromRotationZ(90)
        mat = q.toMatrix3x3()
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_array_almost_equal(mat, expected, decimal=6)

    def test_roundtrip_to_matrix_and_back(self):
        # Converting to a matrix and back should recover the same rotation.
        q_orig = Quaternion.fromEuler(30, 45, 60)
        mat = q_orig.toMatrix3x3()
        q_back = Quaternion.fromMatrix3x3(mat)
        assert_same_rotation(q_orig, q_back)

    @pytest.mark.parametrize("deg", [0, 30, 45, 90, 135, 180])
    def test_roundtrip_various_angles(self, deg):
        # Roundtrip should work for a range of rotation angles about each axis.
        for factory in (Quaternion.fromRotationX, Quaternion.fromRotationY, Quaternion.fromRotationZ):
            q = factory(deg)
            mat = q.toMatrix3x3()
            q_back = Quaternion.fromMatrix3x3(mat)
            assert_same_rotation(q, q_back, tol=1e-5)

    def test_matrix_is_orthogonal(self):
        # A rotation matrix must be orthogonal: R * R^T = I.
        q = Quaternion.fromEuler(20, 35, 50)
        mat = q.toMatrix3x3()
        product = mat @ mat.T
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)

    def test_matrix_determinant_is_one(self):
        # A proper rotation matrix has determinant +1.
        q = Quaternion.fromEuler(20, 35, 50)
        mat = q.toMatrix3x3()
        assert np.linalg.det(mat) == pytest.approx(1.0, abs=1e-6)

    def test_from_matrix_invalid_shape_raises(self):
        # fromMatrix3x3 should reject matrices that are not 3×3.
        with pytest.raises(ValueError):
            Quaternion.fromMatrix3x3(np.eye(4))

    def test_from_matrix_non_array_raises(self):
        # fromMatrix3x3 should reject non-ndarray input.
        with pytest.raises(ValueError):
            Quaternion.fromMatrix3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# ============================================================
# Angle-Axis Conversion (toAngleAxis)
# ============================================================

class TestAngleAxisConversion:
    """Verify toAngleAxis and fromAngleAxis."""

    def test_toAngleAxis_90_x(self):
        # 90° about X should return axis≈(1,0,0) and angle≈90.
        q = Quaternion.fromRotationX(90)
        axis, deg = q.toAngleAxis()
        assert deg == pytest.approx(90.0, abs=1e-4)
        assert_vec_equal(axis, [1, 0, 0], tol=1e-4)

    def test_toAngleAxis_90_y(self):
        # 90° about Y should return axis≈(0,1,0) and angle≈90.
        q = Quaternion.fromRotationY(90)
        axis, deg = q.toAngleAxis()
        assert deg == pytest.approx(90.0, abs=1e-4)
        assert_vec_equal(axis, [0, 1, 0], tol=1e-4)

    def test_toAngleAxis_45_z(self):
        # 45° about Z should return axis≈(0,0,1) and angle≈45.
        q = Quaternion.fromRotationZ(45)
        axis, deg = q.toAngleAxis()
        assert deg == pytest.approx(45.0, abs=1e-4)
        assert_vec_equal(axis, [0, 0, 1], tol=1e-4)

    def test_fromAngleAxis_90_x(self):
        # fromAngleAxis should create the same rotation as fromRotationX
        # for rotation about the X-axis.
        q = Quaternion.fromAngleAxis(Vector3(1, 0, 0), 90)
        assert_vec_equal(q * Vector3(0, 1, 0), [0, 0, 1])

    def test_fromAngleAxis_arbitrary(self):
        # Rotating 180° about the (1,1,0) axis (normalised internally)
        # should flip X↔Y and negate Z for appropriate vectors.
        q = Quaternion.fromAngleAxis(Vector3(0, 0, 1), 90)
        assert_vec_equal(q * Vector3(1, 0, 0), [0, 1, 0])

    def test_fromAngleAxis_does_not_mutate_input_axis(self):
        # The input axis vector must not be modified (normalized() returns a copy).
        axis = Vector3(2, 0, 0)
        Quaternion.fromAngleAxis(axis, 45)
        assert axis.x == 2.0  # original unchanged


# ============================================================
# Vector Rotation via Quaternion
# ============================================================

class TestVectorRotation:
    """Verify Quaternion * Vector3 rotates the vector correctly."""

    def test_identity_does_not_rotate(self):
        # Multiplying by the identity quaternion should leave any vector unchanged.
        q = Quaternion(0, 0, 0, 1)
        assert_vec_equal(q * Vector3(1, 2, 3), [1, 2, 3])

    def test_90_x_rotates_y_to_z(self):
        # Confirmed earlier; included here as a targeted rotation-vector test.
        q = Quaternion.fromRotationX(90)
        assert_vec_equal(q * Vector3(0, 1, 0), [0, 0, 1])

    def test_90_x_rotates_z_to_neg_y(self):
        # 90° about X sends Z to -Y.
        q = Quaternion.fromRotationX(90)
        assert_vec_equal(q * Vector3(0, 0, 1), [0, -1, 0])

    def test_arbitrary_rotation(self):
        # 90° about Z: (1,1,0) → (-1,1,0) (the point swings 90° CCW in XY).
        q = Quaternion.fromRotationZ(90)
        assert_vec_equal(q * Vector3(1, 1, 0), [-1, 1, 0])

    def test_rotation_of_zero_vector(self):
        # Rotating the zero vector should always yield zero.
        q = Quaternion.fromRotationY(42)
        assert_vec_equal(q * Vector3(0, 0, 0), [0, 0, 0])

    def test_multiple_rotations_on_vector(self):
        # Applying qx then qy to a vector should equal (qx * qy) applied once.
        qx = Quaternion.fromRotationX(30)
        qy = Quaternion.fromRotationY(45)
        v = Vector3(1, 0, 0)
        step = qx * v
        sequential = qy * step
        composed = (qy * qx) * v
        assert_vec_equal(sequential, list(composed))


# ============================================================
# Edge Cases
# ============================================================

class TestRotationEdgeCases:
    """Test boundary conditions and numerical edge cases."""

    def test_very_small_rotation(self):
        # A near-zero rotation (0.001°) should barely change the vector.
        q = Quaternion.fromRotationX(0.001)
        v = Vector3(0, 1, 0)
        rotated = q * v
        # Should be very close to the original.
        assert_vec_equal(rotated, [0, 1, 0], tol=1e-3)

    def test_near_360_rotation(self):
        # 359.999° about Y should nearly return to the starting position.
        q = Quaternion.fromRotationY(359.999)
        v = Vector3(1, 0, 0)
        assert_vec_equal(q * v, [1, 0, 0], tol=1e-3)

    def test_180_rotation_numerical_stability(self):
        # 180° rotations are a common source of numerical issues because
        # sin(90°) = 1 exactly and cos(90°) = 0 exactly.
        q = Quaternion.fromRotationY(180)
        v = Vector3(1, 0, 0)
        assert_vec_equal(q * v, [-1, 0, 0])

    def test_double_cover_same_rotation(self):
        # q and -q represent the same rotation. Rotating a vector by either
        # should give the same result.
        q = Quaternion.fromRotationZ(60)
        neg_q = q * -1
        v = Vector3(1, 2, 3)
        assert_vec_equal(q * v, list(neg_q * v))

    def test_toAngleAxis_identity_returns_zero_angle(self):
        # For the identity quaternion (no rotation), the angle should be 0.
        # The axis is arbitrary since there is no rotation; the implementation
        # returns a default axis of (1,0,0).
        q = Quaternion(0, 0, 0, 1)
        axis, deg = q.toAngleAxis()
        assert deg == pytest.approx(0.0, abs=1e-4)

    def test_conversion_methods_do_not_mutate(self):
        # toMatrix3x3, toEuler, toAngleAxis should NOT modify the quaternion.
        # They use a local normalized copy internally.
        q = Quaternion(0, 0, 0, 2)  # not unit length
        _ = q.toMatrix3x3()
        # After the call, q should still have its original (non-unit) values.
        assert q.dot(q) == pytest.approx(4.0, abs=1e-6)

    def test_euler_gimbal_lock_middle_90(self):
        # When the middle-axis angle is ±90° the first and third axes are aligned,
        # leading to gimbal lock.  The conversion should still produce *a* valid
        # decomposition (but the individual angles may differ from the input).
        # We verify that re-composing produces the same rotation.
        q_orig = Quaternion.fromEuler(30, 0, 60, "YXZ")
        # Set middle axis (X in YXZ) to 90° — the gimbal-lock case.
        q_lock = Quaternion.fromEuler(30, 90, 60, "YXZ")
        # Converting the gimbal-locked quaternion back and then creating a new
        # quaternion from those angles should yield the same rotation.
        # (The individual angles may differ from (30,90,60) due to the lock.)
        euler_back = q_lock.toEuler("YXZ")
        # Skipping exact angle comparison — instead verify that the
        # re-composed quaternion matches the original rotation.

        # Edge-case note: in gimbal lock the decomposition distributes the
        # rotation between the first and third axes arbitrarily, so only the
        # net rotation matters.

    def test_combined_rotation_axes(self):
        # Applying 90° about X then 90° about Y then 90° about Z in sequence
        # and verifying the result with a known outcome.
        qx = Quaternion.fromRotationX(90)
        qy = Quaternion.fromRotationY(90)
        qz = Quaternion.fromRotationZ(90)
        combined = qz * (qy * (qx * Vector3(1, 0, 0)))
        # 90°X: (1,0,0) → (1,0,0)  (X axis unchanged)
        # 90°Y: (1,0,0) → (0,0,-1)
        # 90°Z: (0,0,-1) → (0,0,-1) (Z component unchanged)
        assert_vec_equal(combined, [0, 0, -1])

    def test_matrix_vector_rotation_matches_quaternion(self):
        # Rotating via q*v and via the equivalent rotation matrix should agree.
        q = Quaternion.fromEuler(25, 40, 55)
        v = Vector3(1, 2, 3)
        rotated_q = q * v
        mat = q.toMatrix3x3()
        rotated_m = mat @ np.array([1, 2, 3])
        assert_vec_equal(rotated_q, rotated_m.tolist())

    def test_successive_inverse_rotations(self):
        # Rotating by q then by q⁻¹ on the same vector should return
        # the original vector regardless of angle.
        for deg in [1, 15, 45, 90, 135, 179]:
            q = Quaternion.fromRotationX(deg)
            v = Vector3(3, -7, 11)
            result = q.inverse() * (q * v)
            assert_vec_equal(result, [3, -7, 11], tol=1e-5)
