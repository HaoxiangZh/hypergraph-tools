"""t-product algebra for tensors of any order p >= 3.

A tensor has shape ``(n1, n2, n3, ..., np)``. The first two axes index the
"matrix" part of each frontal slice; axes 2..p-1 are the tube axes. Every
operation below is carried out slice-wise after an FFT along the tube axes,
which is equivalent to the block-circulant definition of the t-product
(Kilmer & Martin 2011; Martin, Shafer & LaRue 2013).
"""

import warnings

import numpy as np

__all__ = [
    "t_fft",
    "t_ifft",
    "t_product",
    "t_product_bcirc",
    "t_transpose",
    "t_eye",
    "t_inverse",
    "t_power",
    "t_hadamard",
    "t_norm",
    "t_eig",
    "t_eigh",
    "bcirc",
    "unfold",
    "fold",
    "diag_circ",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tube_axes(A):
    return tuple(range(2, A.ndim))


def _to_stack(A):
    """(n1, n2, *tube) -> (*tube, n1, n2), so numpy's batched linalg applies."""
    return np.moveaxis(A, (0, 1), (-2, -1))


def _from_stack(A):
    return np.moveaxis(A, (-2, -1), (0, 1))


def _mirror(A):
    """Return B with B[:, :, k] = A[:, :, -k] (indices mod n) on every tube axis."""
    for ax in _tube_axes(A):
        A = np.roll(np.flip(A, axis=ax), 1, axis=ax)
    return A


def _maybe_real(C, tol=1e-10):
    """Drop the imaginary part when it is round-off only."""
    scale = max(np.abs(C).max(initial=0.0), 1.0)
    if np.abs(C.imag).max(initial=0.0) <= tol * scale:
        return C.real
    return C


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def t_fft(A):
    """FFT of ``A`` along all tube axes (axes 2..p-1)."""
    A = np.asarray(A)
    if A.ndim < 3:
        return A.astype(np.result_type(A, np.complex128))
    return np.fft.fftn(A, axes=_tube_axes(A))


def t_ifft(A, real=False):
    """Inverse of :func:`t_fft`. With ``real=True`` the real part is returned."""
    A = np.asarray(A)
    C = A if A.ndim < 3 else np.fft.ifftn(A, axes=_tube_axes(A))
    return C.real if real else C


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------

def t_product(*tensors):
    """t-product ``A * B * ...`` of two or more tensors, computed with the FFT.

    All tensors must share the same tube shape (axes 2..p-1), and consecutive
    tensors must have matching inner dimensions. The result is real when all
    inputs are real.
    """
    if len(tensors) < 2:
        raise ValueError("t_product needs at least two tensors")
    arrays = [np.asarray(T) for T in tensors]
    tube = arrays[0].shape[2:]
    for k, T in enumerate(arrays[1:], start=2):
        if T.ndim != arrays[0].ndim or T.shape[2:] != tube:
            raise ValueError(
                f"tensor {k} has shape {T.shape}; expected tube shape {tube}"
            )

    C = _to_stack(t_fft(arrays[0]))
    for k, T in enumerate(arrays[1:], start=2):
        if C.shape[-1] != T.shape[0]:
            raise ValueError(
                f"inner dimensions do not match: {C.shape[-1]} vs {T.shape[0]} (tensor {k})"
            )
        C = C @ _to_stack(t_fft(T))

    real = all(np.isrealobj(T) for T in arrays)
    return t_ifft(_from_stack(C), real=real)


def bcirc(A):
    """Block-circulant matrix of ``A`` along its last axis.

    Block ``(i, j)`` equals ``A[..., (i - j) % n]``. For a 3rd-order
    ``n1 x n2 x n3`` tensor the result is ``(n1*n3) x (n2*n3)``; for higher
    orders the blocks are themselves tensors.
    """
    A = np.asarray(A)
    n = A.shape[-1]
    rows = [
        np.concatenate([A[..., (i - j) % n] for j in range(n)], axis=1)
        for i in range(n)
    ]
    return np.concatenate(rows, axis=0)


def unfold(A):
    """Stack the slices of ``A`` along its last axis vertically."""
    A = np.asarray(A)
    return np.concatenate([A[..., k] for k in range(A.shape[-1])], axis=0)


def fold(M, p):
    """Inverse of :func:`unfold`: split ``M`` into ``p`` row blocks, stack them on a new last axis."""
    M = np.asarray(M)
    if M.shape[0] % p:
        raise ValueError(f"first dimension {M.shape[0]} is not divisible by {p}")
    n = M.shape[0] // p
    return np.stack([M[k * n:(k + 1) * n] for k in range(p)], axis=-1)


def t_product_bcirc(A, B):
    """t-product by definition, ``fold(bcirc(A) @ unfold(B))`` (recursive for p > 3).

    Much slower than :func:`t_product`; useful as a reference.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim == 2:
        return A @ B
    inner = t_product_bcirc(bcirc(A), unfold(B)) if A.ndim > 3 else bcirc(A) @ unfold(B)
    return fold(inner, A.shape[-1])


def t_hadamard(A, B):
    """Element-wise product of ``A`` and ``B`` in the Fourier domain."""
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError(f"shapes do not match: {A.shape} vs {B.shape}")
    real = np.isrealobj(A) and np.isrealobj(B)
    return t_ifft(t_fft(A) * t_fft(B), real=real)


# ---------------------------------------------------------------------------
# transpose, identity, inverse, power
# ---------------------------------------------------------------------------

def t_transpose(A):
    """Tensor (conjugate) transpose under the t-product.

    Conjugate-transposes every frontal slice and reverses the order of slices
    ``1..n-1`` along every tube axis, so that
    ``t_transpose(t_product(A, B)) == t_product(t_transpose(B), t_transpose(A))``.
    """
    A = np.asarray(A)
    return _mirror(np.conj(np.swapaxes(A, 0, 1)))


def t_eye(n, tube_shape):
    """Identity tensor of size ``n x n x tube_shape``."""
    tube_shape = tuple(np.atleast_1d(tube_shape))
    I = np.zeros((n, n) + tube_shape)
    I[(slice(None), slice(None)) + (0,) * len(tube_shape)] = np.eye(n)
    return I


def t_inverse(A):
    """Tensor (pseudo-)inverse: ``t_product(A, t_inverse(A))`` is the identity for invertible ``A``."""
    A = np.asarray(A)
    C = np.linalg.pinv(_to_stack(t_fft(A)))
    return t_ifft(_from_stack(C), real=np.isrealobj(A))


def t_power(A, k):
    """Tensor power ``A * A * ... * A`` (``k`` times) under the t-product.

    Integer ``k`` (including negative) uses repeated products / the inverse;
    non-integer ``k`` uses the eigendecomposition of each Fourier slice.
    """
    A = np.asarray(A)
    Ah = _to_stack(t_fft(A))
    if float(k).is_integer():
        C = np.linalg.matrix_power(Ah, int(k))
        return t_ifft(_from_stack(C), real=np.isrealobj(A))
    w, V = np.linalg.eig(Ah)
    C = (V * (w.astype(complex) ** k)[..., None, :]) @ np.linalg.inv(V)
    return _maybe_real(t_ifft(_from_stack(C)))


def t_norm(X):
    """Tensor magnitude ``|X| = (X^T * X)^(1/2)`` under the t-product.

    For a tensor signal of shape ``(N, 1, ...)`` this is its t-norm tube.
    """
    X = np.asarray(X)
    Xh = _to_stack(t_fft(X))
    G = np.conj(np.swapaxes(Xh, -1, -2)) @ Xh
    w, V = np.linalg.eigh(G)
    S = (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    return t_ifft(_from_stack(S), real=np.isrealobj(X))


# ---------------------------------------------------------------------------
# eigendecomposition
# ---------------------------------------------------------------------------

def t_eig(A):
    """Eigendecomposition ``A * V = V * L`` of a square tensor.

    Returns ``(V, L)`` where ``L`` has diagonal frontal slices in the Fourier
    domain. Both are complex in general. For t-symmetric tensors prefer
    :func:`t_eigh`, which returns real, orthonormal eigenvectors.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square in its first two axes, got {A.shape}")
    w, V = np.linalg.eig(_to_stack(t_fft(A)))
    if np.max(np.linalg.cond(V)) > 1e12:
        warnings.warn("some Fourier slices of A look defective (not diagonalizable)", RuntimeWarning)
    n = A.shape[0]
    L = np.zeros_like(V)
    L[..., np.arange(n), np.arange(n)] = w
    return t_ifft(_from_stack(V)), t_ifft(_from_stack(L))


def t_eigh(A):
    """Eigendecomposition of a t-symmetric tensor (``A == t_transpose(A)``).

    Returns ``(V, L)`` with ``A * V = V * L`` and ``V^T * V = I``. Eigenvalues
    in each Fourier slice are sorted in ascending order. For real ``A`` both
    ``V`` and ``L`` are real, so ``V`` can be used directly as a Fourier basis.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square in its first two axes, got {A.shape}")
    if not np.allclose(A, t_transpose(A)):
        raise ValueError("A is not t-symmetric; use t_eig instead")

    Ah = _to_stack(t_fft(A))
    w, V = np.linalg.eigh(Ah)
    real = np.isrealobj(A) and A.ndim > 2
    if real:
        # Slices k and -k are complex conjugates. Self-conjugate slices are
        # real symmetric, so solve them in real arithmetic, and give the
        # remaining pairs conjugate eigenvectors, so the inverse FFT is real.
        tube = A.shape[2:]
        idx = np.arange(int(np.prod(tube))).reshape(tube)
        mirror_idx = _mirror(idx[None, None])[0, 0]
        selfconj = idx == mirror_idx
        w[selfconj], V[selfconj] = np.linalg.eigh(Ah[selfconj].real)
        V_mirror = _to_stack(_mirror(_from_stack(V)))
        V = np.where((idx <= mirror_idx)[..., None, None], V, np.conj(V_mirror))

    n = A.shape[0]
    L = np.zeros(V.shape, dtype=complex)
    L[..., np.arange(n), np.arange(n)] = w
    return t_ifft(_from_stack(V), real=real), t_ifft(_from_stack(L), real=real)


def diag_circ(A):
    """Block-diagonalization of ``bcirc(A)`` for a 3rd-order tensor.

    Equals ``(F kron I) @ bcirc(A) @ (F kron I)^H / n3``: a block-diagonal
    matrix whose blocks are the frontal slices of ``t_fft(A)``.
    """
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError("diag_circ expects a 3rd-order tensor")
    n1, n2, n3 = A.shape
    Ah = t_fft(A)
    D = np.zeros((n1 * n3, n2 * n3), dtype=complex)
    for k in range(n3):
        D[k * n1:(k + 1) * n1, k * n2:(k + 1) * n2] = Ah[:, :, k]
    return D
