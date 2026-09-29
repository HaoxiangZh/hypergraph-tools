"""Helpers for hypergraph signal processing with the t-product (t-HGSP)."""

import numpy as np

from .tproduct import t_norm, t_product, t_transpose

__all__ = ["t_sym", "signal_tensor", "t_tv", "t_tv_laplacian", "random_tensor"]


def t_sym(A):
    """Symmetrize a tensor along its tube axes.

    An ``N x N x N`` tensor becomes ``N x N x (2N+1)`` with frontal slices
    ``[0, A[:, :, 0]/2, ..., A[:, :, N-1]/2, A[:, :, N-1]/2, ..., A[:, :, 0]/2]``.
    For higher orders the rule is applied recursively along each tube axis,
    so ``N x N x N x N`` becomes ``N x N x (2N+1) x (2N+1)``.
    """
    A = np.asarray(A)
    if A.ndim < 3:
        raise ValueError("t_sym expects a tensor of order >= 3")
    if A.ndim == 3:
        halves = [0.5 * A[..., k] for k in range(A.shape[-1])]
    else:
        halves = [0.5 * t_sym(A[..., k]) for k in range(A.shape[-1])]
    zero = np.zeros_like(halves[0])
    return np.stack([zero] + halves + halves[::-1], axis=-1)


def signal_tensor(x, M):
    """Tensor signal of a hypergraph with maximum cardinality ``M``.

    Returns the ``(M-1)``-fold outer product of ``x`` with itself, reshaped to
    ``N x 1 x N x ... x N`` (order ``M``); for ``M = 2`` this is ``x`` as an
    ``N x 1`` column.
    """
    x = np.asarray(x).ravel()
    if M < 2:
        raise ValueError("M must be at least 2")
    X = x
    for _ in range(M - 2):
        X = np.multiply.outer(X, x)
    return X.reshape((x.size, 1) + X.shape[1:])


def t_tv(A, X):
    """Total variation ``|X - A * X|`` of a tensor signal ``X`` w.r.t. a shifting operator ``A``."""
    return t_norm(np.asarray(X) - t_product(A, X))


def t_tv_laplacian(L, X):
    """Laplacian quadratic form ``X^T * L * X`` of a tensor signal ``X``."""
    return t_product(t_transpose(X), L, X)


def random_tensor(shape, rng=None):
    """Random tensor with symmetric frontal slices, symmetrized by :func:`t_sym`.

    ``shape`` is the size before symmetrization, e.g. ``(N, N, N)`` gives an
    ``N x N x (2N+1)`` result. ``rng`` is a seed or ``numpy.random.Generator``.
    """
    rng = np.random.default_rng(rng)
    A = rng.random(shape)
    return t_sym(A + np.swapaxes(A, 0, 1))
