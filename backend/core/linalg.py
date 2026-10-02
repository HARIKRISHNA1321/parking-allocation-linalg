"""
Linear Algebra Routines - Implemented from scratch using NumPy arrays.
NOTE: As required by project guidelines, np.linalg is NOT used here.
np.linalg is only used inside pytest test suites for cross-validation.
"""

from typing import Tuple, List, Dict, Any
import numpy as np


def rref(A: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, List[int], int]:
    """
    Computes the Reduced Row Echelon Form (RREF) of matrix A using Gaussian elimination
    with partial pivoting.

    Parameters:
        A: 2D numpy array (m x n)
        tol: Tolerance for treating floating point values as zero

    Returns:
        R: Matrix in RREF
        pivot_cols: List of column indices containing leading 1s (pivots)
        rank: Rank of the matrix (number of pivots)
    """
    R = A.astype(np.float64, copy=True)
    m, n = R.shape
    pivot_row = 0
    pivot_cols = []

    for c in range(n):
        if pivot_row >= m:
            break

        # Partial pivoting: select row with max absolute value in current column
        max_idx = pivot_row + np.argmax(np.abs(R[pivot_row:, c]))
        max_val = np.abs(R[max_idx, c])

        if max_val < tol:
            # Free column, no pivot in this column
            continue

        # Swap rows to bring pivot to current pivot_row
        if max_idx != pivot_row:
            R[[pivot_row, max_idx]] = R[[max_idx, pivot_row]]

        # Normalize pivot row so pivot element is 1
        pivot_elem = R[pivot_row, c]
        R[pivot_row] = R[pivot_row] / pivot_elem

        # Eliminate current column in all other rows
        for r in range(m):
            if r != pivot_row and np.abs(R[r, c]) > tol:
                factor = R[r, c]
                R[r] -= factor * R[pivot_row]

        # Clean tiny values
        R[np.abs(R) < tol] = 0.0

        pivot_cols.append(c)
        pivot_row += 1

    rank = len(pivot_cols)
    return R, pivot_cols, rank


def lu_decomposition(A: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the LU decomposition with partial pivoting: P @ A = L @ U
    where P is a permutation matrix, L is unit lower-triangular, and U is upper-triangular.

    Parameters:
        A: Square matrix (n x n)
        tol: Tolerance for singular pivot check

    Returns:
        P: Permutation matrix (n x n)
        L: Unit lower-triangular matrix (n x n)
        U: Upper-triangular matrix (n x n)
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU decomposition requires a square matrix")

    U = A.astype(np.float64, copy=True)
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        # Find pivot in column k below diagonal
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if np.abs(U[max_row, k]) < tol:
            continue

        # Swap rows in U, P, and previously calculated columns of L
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        # Elimination
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return P, L, U


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves L y = b where L is lower-triangular.
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - s) / L[i, i]
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Solves U x = y where U is upper-triangular.
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if np.abs(U[i, i]) < tol:
            x[i] = 0.0
        else:
            s = np.dot(U[i, i + 1:], x[i + 1:])
            x[i] = (y[i] - s) / U[i, i]
    return x


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves A x = b using LU decomposition with partial pivoting: P A = L U.
    P A x = P b  ==>  L U x = P b.
    1. y = forward_substitution(L, P b)
    2. x = backward_substitution(U, y)
    Does NOT invert any matrix.
    """
    P, L, U = lu_decomposition(A)
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = backward_substitution(U, y)
    return x


def nullspace(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Computes a basis for the null space of A (i.e. all x such that A x = 0)
    using the RREF of A.

    Returns:
        null_basis: 2D array where each column is a null-space basis vector.
                    If nullity is 0, returns an empty array of shape (n, 0).
    """
    R, pivot_cols, rank = rref(A, tol=tol)
    m, n = A.shape
    free_cols = [c for c in range(n) if c not in pivot_cols]
    nullity = len(free_cols)

    if nullity == 0:
        return np.zeros((n, 0), dtype=np.float64)

    null_basis = np.zeros((n, nullity), dtype=np.float64)
    for idx, f in enumerate(free_cols):
        # Set free variable to 1
        null_basis[f, idx] = 1.0
        # Determine dependent variables from RREF equations: x_p + sum(R[r, f] * x_f) = 0
        for r, p in enumerate(pivot_cols):
            null_basis[p, idx] = -R[r, f]

    return null_basis


def gram_schmidt_cgs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Classical Gram-Schmidt QR decomposition of A (m x n, m >= n).
    Returns Q, R, and the orthogonality defect max|Q^T Q - I|.
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        v = A[:, j].astype(np.float64)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        norm_v = np.sqrt(np.dot(v, v))
        R[j, j] = norm_v
        if norm_v > 1e-12:
            Q[:, j] = v / norm_v

    defect = float(np.max(np.abs(Q.T @ Q - np.eye(n))))
    return Q, R, defect


def gram_schmidt_mgs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Modified Gram-Schmidt QR decomposition of A (m x n, m >= n).
    Numerically more stable than classical GS by orthogonalizing remaining vectors
    successively against each new unit vector.
    Returns Q, R, and the orthogonality defect max|Q^T Q - I|.
    """
    m, n = A.shape
    V = A.astype(np.float64, copy=True)
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        norm_v = np.sqrt(np.dot(V[:, i], V[:, i]))
        R[i, i] = norm_v
        if norm_v > 1e-12:
            Q[:, i] = V[:, i] / norm_v
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    defect = float(np.max(np.abs(Q.T @ Q - np.eye(n))))
    return Q, R, defect


def orthogonal_projection(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Computes the orthogonal projection of vector b onto col(A).
    Solves normal equations (A^T A) x_hat = A^T b using our LU solver (no explicit inverse).
    b_hat = A @ x_hat
    residual = b - b_hat
    Checks orthogonality: A^T @ residual ~ 0.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    x_hat = lu_solve(AtA, Atb)
    b_hat = A @ x_hat
    residual = b - b_hat
    ortho_error = float(np.max(np.abs(A.T @ residual)))
    res_norm = float(np.sqrt(np.sum(residual ** 2)))

    return {
        "x_hat": x_hat,
        "b_hat": b_hat,
        "residual": residual,
        "orthogonality_error": ortho_error,
        "residual_norm": res_norm
    }


def least_squares_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fits least squares linear model y ~ X beta using normal equations (X^T X) beta = X^T y
    solved via from-scratch LU decomposition.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = lu_solve(XtX, Xty)
    y_pred = X @ beta
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "beta": beta,
        "y_pred": y_pred,
        "residuals": residuals,
        "rmse": rmse
    }


def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    Power iteration algorithm to compute dominant eigenvalue and eigenvector
    for a symmetric real matrix A.
    """
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64)
    v /= np.sqrt(np.dot(v, v))

    eigenvalue = 0.0
    for _ in range(max_iter):
        w = A @ v
        norm_w = np.sqrt(np.dot(w, w))
        if norm_w < 1e-14:
            break
        v_next = w / norm_w
        eigenvalue = float(v_next.T @ A @ v_next)

        diff = np.sqrt(np.sum((v_next - v) ** 2))
        if diff < tol:
            v = v_next
            break
        v = v_next

    return eigenvalue, v


def eigen_deflation_top_k(A: np.ndarray, k: int = 2) -> List[Tuple[float, np.ndarray, float]]:
    """
    Extracts top k eigenpairs (lambda_i, v_i) of symmetric matrix A
    using power iteration with Hotelling's deflation:
    A_{i+1} = A_i - lambda_i * (v_i @ v_i^T)

    Returns:
        List of tuples: (eigenvalue, eigenvector, verification_residual_norm)
    """
    A_curr = A.astype(np.float64, copy=True)
    results = []

    for _ in range(k):
        val, vec = power_iteration(A_curr)
        res = float(np.sqrt(np.sum((A @ vec - val * vec) ** 2)))
        results.append((val, vec, res))
        A_curr = A_curr - val * np.outer(vec, vec)

    return results


def jacobi_eigen(A: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes complete eigendecomposition of symmetric matrix A using cyclic Jacobi method.
    Returns:
        eigenvalues: sorted descending
        eigenvectors: matrix V whose columns are corresponding eigenvectors (A = V diag(lambda) V^T)
    """
    n = A.shape[0]
    D = A.astype(np.float64, copy=True)
    V = np.eye(n, dtype=np.float64)

    for _ in range(max_iter):
        off_diag = np.copy(D)
        np.fill_diagonal(off_diag, 0.0)
        p, q = np.unravel_index(np.argmax(np.abs(off_diag)), (n, n))

        if np.abs(D[p, q]) < tol:
            break

        if np.abs(D[p, p] - D[q, q]) < 1e-14:
            theta = np.pi / 4.0 if D[p, q] > 0 else -np.pi / 4.0
        else:
            tau = (D[q, q] - D[p, p]) / (2.0 * D[p, q])
            t = np.sign(tau) / (np.abs(tau) + np.sqrt(1.0 + tau**2))
            theta = np.arctan(t)

        c = np.cos(theta)
        s = np.sin(theta)

        J = np.eye(n, dtype=np.float64)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s

        D = J.T @ D @ J
        V = V @ J

    evals = np.diag(D)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    V = V[:, order]

    return evals, V


def vector_norms(v: np.ndarray) -> Dict[str, float]:
    """
    Computes L1 (Manhattan), L2 (Euclidean), and Linf (Chebyshev) norms of vector v.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    l1 = float(np.sum(np.abs(v_arr)))
    l2 = float(np.sqrt(np.sum(v_arr ** 2)))
    linf = float(np.max(np.abs(v_arr)))
    return {
        "l1_manhattan": l1,
        "l2_euclidean": l2,
        "linf_chebyshev": linf
    }
