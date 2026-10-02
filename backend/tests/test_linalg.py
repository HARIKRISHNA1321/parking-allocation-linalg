"""
Pytest Test Suite for Scratch Linear Algebra Routines.

Validates from-scratch algorithms against NumPy and SymPy:
- RREF and Rank vs SymPy Matrix.rref()
- LU Decomposition & Solver vs np.linalg.solve
- Rank & Nullity vs Rank-Nullity theorem and SymPy nullspace()
- Gram-Schmidt QR (CGS and MGS) vs np.linalg.qr (accounting for column sign convention)
- Orthogonal projection and residual orthogonality At @ (b - b_hat) ~ 0
- Least squares parameter estimates vs np.linalg.lstsq
- Power iteration & deflation eigenvalues vs np.linalg.eigh
- Jacobi symmetric diagonalization and reconstruction error monotonicity
"""

import pytest
import numpy as np
import sympy as sp

from backend.core.linalg import (
    rref,
    lu_decomposition,
    forward_substitution,
    backward_substitution,
    lu_solve,
    nullspace,
    gram_schmidt_cgs,
    gram_schmidt_mgs,
    orthogonal_projection,
    least_squares_fit,
    power_iteration,
    eigen_deflation_top_k,
    jacobi_eigen,
    vector_norms
)


def test_rref_and_rank_vs_sympy():
    """Cross-checks scratch RREF and rank against SymPy rref."""
    rng = np.random.RandomState(42)
    # Test multiple shapes and rank configurations
    test_matrices = [
        # Invertible 3x3
        np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]),
        # Rank-deficient 3x4 (row 3 is sum of 1 and 2)
        np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 1.0], [3.0, 2.0, 4.0, 5.0]]),
        # Rectangular tall 5x3
        rng.randn(5, 3),
        # Matrix with dependent column (Total column)
        np.column_stack([rng.randn(6, 3), np.zeros(6)])
    ]
    # Add dependent sum column to the last one
    test_matrices[-1][:, 3] = np.sum(test_matrices[-1][:, :3], axis=1)

    for A in test_matrices:
        R_scratch, pivots_scratch, rank_scratch = rref(A, tol=1e-8)

        # SymPy reference with numerical zero-tolerance
        sp_mat = sp.Matrix(A)
        R_sympy, pivots_sympy = sp_mat.rref(iszerofunc=lambda x: abs(float(x)) < 1e-8)
        R_sympy_arr = np.array(R_sympy).astype(np.float64)

        assert rank_scratch == len(pivots_sympy), f"Rank mismatch: {rank_scratch} vs {len(pivots_sympy)}"
        assert list(pivots_scratch) == list(pivots_sympy), f"Pivots mismatch: {pivots_scratch} vs {pivots_sympy}"
        np.testing.assert_allclose(R_scratch, R_sympy_arr, atol=1e-7, err_msg="RREF matrix content mismatch")


def test_lu_decomposition_and_solve_vs_numpy():
    """Cross-checks LU decomposition PA = LU and lu_solve against np.linalg.solve."""
    rng = np.random.RandomState(101)
    for dim in [3, 4, 6]:
        # Generate well-conditioned random square matrix
        A = rng.randn(dim, dim) + dim * np.eye(dim)
        b = rng.randn(dim)

        P, L, U = lu_decomposition(A)

        # 1. Verify PA = LU
        np.testing.assert_allclose(P @ A, L @ U, atol=1e-9, err_msg="PA != LU decomposition failed")

        # 2. Verify L is unit lower-triangular
        np.testing.assert_allclose(np.diag(L), np.ones(dim), atol=1e-9)
        assert np.allclose(np.triu(L, k=1), 0.0), "L is not lower triangular"

        # 3. Verify U is upper-triangular
        assert np.allclose(np.tril(U, k=-1), 0.0), "U is not upper triangular"

        # 4. Solve Ax = b without matrix inversion
        x_scratch = lu_solve(A, b)
        x_numpy = np.linalg.solve(A, b)

        np.testing.assert_allclose(x_scratch, x_numpy, atol=1e-9, err_msg="lu_solve mismatch with np.linalg.solve")
        np.testing.assert_allclose(A @ x_scratch, b, atol=1e-9, err_msg="Ax != b residual error")


def test_nullspace_and_rank_nullity():
    """Verifies null-space computation and the Rank-Nullity Theorem."""
    rng = np.random.RandomState(7)
    # Create matrix where column 4 is exactly col0 + col1 + col2 + col3
    cols = rng.randn(10, 4)
    total_col = np.sum(cols, axis=1, keepdims=True)
    X = np.hstack([cols, total_col])  # 10 x 5

    R, pivots, rank = rref(X)
    null_basis = nullspace(X)

    # 1. Rank-Nullity Theorem
    nullity = null_basis.shape[1]
    assert rank + nullity == X.shape[1], f"Rank-Nullity violated: {rank} + {nullity} != {X.shape[1]}"
    assert rank == 4
    assert nullity == 1

    # 2. Check X @ v == 0
    v = null_basis[:, 0]
    np.testing.assert_allclose(X @ v, np.zeros(X.shape[0]), atol=1e-8, err_msg="X @ v != 0")

    # 3. Null vector should be proportional to (1, 1, 1, 1, -1)
    normalized_v = v / v[-1]  # scale so last element is 1
    expected_v = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(normalized_v, expected_v, atol=1e-7, err_msg="Null vector != (-1,-1,-1,-1,1)")


def test_gram_schmidt_qr_vs_numpy():
    """Verifies Classical & Modified Gram-Schmidt QR vs np.linalg.qr."""
    rng = np.random.RandomState(88)
    A = rng.randn(15, 4)

    # 1. Classical GS
    Q_cgs, R_cgs, defect_cgs = gram_schmidt_cgs(A)
    # Check Q^T Q == I
    np.testing.assert_allclose(Q_cgs.T @ Q_cgs, np.eye(4), atol=1e-12)
    # Check Q R == A
    np.testing.assert_allclose(Q_cgs @ R_cgs, A, atol=1e-12)

    # 2. Modified GS
    Q_mgs, R_mgs, defect_mgs = gram_schmidt_mgs(A)
    np.testing.assert_allclose(Q_mgs.T @ Q_mgs, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(Q_mgs @ R_mgs, A, atol=1e-12)

    # 3. Compare with numpy.linalg.qr (up to column sign flips)
    Q_np, R_np = np.linalg.qr(A)
    for i in range(4):
        # Column i must be parallel to NumPy's column i
        sign = np.sign(np.dot(Q_mgs[:, i], Q_np[:, i]))
        np.testing.assert_allclose(Q_mgs[:, i] * sign, Q_np[:, i], atol=1e-10)
        np.testing.assert_allclose(R_mgs[i, :] * sign, R_np[i, :], atol=1e-10)


def test_orthogonal_projection_residual():
    """Verifies orthogonal projection onto col(A) and that A^T @ residual == 0."""
    rng = np.random.RandomState(99)
    A = rng.randn(20, 3)
    b = rng.randn(20)

    proj = orthogonal_projection(A, b)
    b_hat = proj["b_hat"]
    residual = proj["residual"]

    # 1. Verify residual = b - b_hat
    np.testing.assert_allclose(residual, b - b_hat, atol=1e-12)

    # 2. Orthogonality check: A^T @ residual == 0
    np.testing.assert_allclose(A.T @ residual, np.zeros(3), atol=1e-10, err_msg="Residual not orthogonal to col(A)")

    # 3. Idempotence: projecting b_hat again should yield b_hat
    proj_again = orthogonal_projection(A, b_hat)
    np.testing.assert_allclose(proj_again["b_hat"], b_hat, atol=1e-10, err_msg="Projection is not idempotent")


def test_least_squares_vs_numpy():
    """Verifies that scratch least squares normal equation solver matches np.linalg.lstsq."""
    rng = np.random.RandomState(456)
    m, n = 50, 4
    X = rng.randn(m, n)
    true_beta = np.array([2.5, -1.8, 0.7, 4.2])
    y = X @ true_beta + 0.1 * rng.randn(m)

    fit_scratch = least_squares_fit(X, y)
    beta_scratch = fit_scratch["beta"]

    beta_np, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    np.testing.assert_allclose(
        beta_scratch, beta_np, atol=1e-9, err_msg="Scratch least squares beta != np.linalg.lstsq"
    )
    assert fit_scratch["rmse"] > 0


def test_power_iteration_and_deflation_vs_numpy():
    """Verifies dominant eigenvalues from power iteration + deflation against np.linalg.eigh."""
    rng = np.random.RandomState(789)
    M = rng.randn(5, 5)
    S = M.T @ M  # Symmetric positive semi-definite

    top2 = eigen_deflation_top_k(S, k=2)
    val1_sc, vec1_sc, res1 = top2[0]
    val2_sc, vec2_sc, res2 = top2[1]

    # Verify S v = lambda v
    np.testing.assert_allclose(S @ vec1_sc, val1_sc * vec1_sc, atol=1e-7)
    np.testing.assert_allclose(S @ vec2_sc, val2_sc * vec2_sc, atol=1e-7)

    # Compare with np.linalg.eigh
    w_np, _ = np.linalg.eigh(S)
    sorted_w_np = np.sort(w_np)[::-1]

    np.testing.assert_allclose(val1_sc, sorted_w_np[0], rtol=1e-5)
    np.testing.assert_allclose(val2_sc, sorted_w_np[1], rtol=1e-5)


def test_jacobi_eigen_and_reconstruction():
    """Verifies full spectral decomposition S = V Lambda V^T and reconstruction error monotonicity."""
    rng = np.random.RandomState(321)
    M = rng.randn(4, 4)
    S = M.T @ M

    evals, V = jacobi_eigen(S)

    # 1. Orthonormality of eigenvectors
    np.testing.assert_allclose(V.T @ V, np.eye(4), atol=1e-9)

    # 2. Spectral reconstruction: S = V Lambda V^T
    S_reconstructed = V @ np.diag(evals) @ V.T
    np.testing.assert_allclose(S_reconstructed, S, atol=1e-9)

    # 3. Monotonically decreasing Frobenius error
    errors = []
    for k in range(1, 5):
        Sk = np.zeros_like(S)
        for i in range(k):
            Sk += evals[i] * np.outer(V[:, i], V[:, i])
        err = np.sqrt(np.sum((S - Sk) ** 2))
        errors.append(err)

    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1], f"Reconstruction error did not decrease: {errors}"
    assert np.isclose(errors[-1], 0.0, atol=1e-9)


def test_vector_norms():
    """Verifies vector norms L1, L2, Linf."""
    v = np.array([3.0, -4.0])
    norms = vector_norms(v)
    assert norms["l1_manhattan"] == 7.0
    assert norms["l2_euclidean"] == 5.0
    assert norms["linf_chebyshev"] == 4.0
