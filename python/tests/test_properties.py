"""Algebraic identities of the t-product for 3rd- to 5th-order tensors."""

import numpy as np
import pytest

import hypergraph_tools as ht

TUBES = [(5,), (3, 2), (2, 3, 2)]


def rand(rng, *shape, complex_=False):
    A = rng.random(shape)
    return A + 1j * rng.random(shape) if complex_ else A


def reference_product(A, B):
    """Slice-by-slice FFT reference, independent of the package internals."""
    tube = A.shape[2:]
    Ah = np.fft.fftn(A, axes=range(2, A.ndim)).reshape(A.shape[:2] + (-1,))
    Bh = np.fft.fftn(B, axes=range(2, B.ndim)).reshape(B.shape[:2] + (-1,))
    Ch = np.stack([Ah[:, :, k] @ Bh[:, :, k] for k in range(Ah.shape[2])], axis=-1)
    return np.fft.ifftn(Ch.reshape(Ch.shape[:2] + tube), axes=range(2, A.ndim))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("tube", TUBES)
@pytest.mark.parametrize("complex_", [False, True])
def test_product_matches_reference_and_definition(rng, tube, complex_):
    A = rand(rng, 3, 4, *tube, complex_=complex_)
    B = rand(rng, 4, 2, *tube, complex_=complex_)
    C = ht.t_product(A, B)
    np.testing.assert_allclose(C, reference_product(A, B), atol=1e-12)
    np.testing.assert_allclose(C, ht.t_product_bcirc(A, B), atol=1e-12)
    assert np.isrealobj(C) != complex_


def test_product_is_associative_and_multi_arg(rng):
    A, B, C = rand(rng, 3, 4, 5), rand(rng, 4, 2, 5), rand(rng, 2, 3, 5)
    np.testing.assert_allclose(ht.t_product(A, B, C), ht.t_product(ht.t_product(A, B), C), atol=1e-12)


def test_product_checks_shapes(rng):
    with pytest.raises(ValueError):
        ht.t_product(rand(rng, 3, 4, 5), rand(rng, 3, 2, 5))
    with pytest.raises(ValueError):
        ht.t_product(rand(rng, 3, 4, 5), rand(rng, 4, 2, 4))


@pytest.mark.parametrize("tube", TUBES)
def test_transpose_reverses_products(rng, tube):
    A = rand(rng, 3, 4, *tube, complex_=True)
    B = rand(rng, 4, 2, *tube, complex_=True)
    lhs = ht.t_transpose(ht.t_product(A, B))
    rhs = ht.t_product(ht.t_transpose(B), ht.t_transpose(A))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)
    np.testing.assert_array_equal(ht.t_transpose(ht.t_transpose(A)), A)


@pytest.mark.parametrize("tube", TUBES)
def test_inverse_and_power(rng, tube):
    S = rand(rng, 3, 3, *tube)
    I = ht.t_eye(3, tube)
    np.testing.assert_allclose(ht.t_product(S, ht.t_inverse(S)), I, atol=1e-10)
    np.testing.assert_allclose(ht.t_power(S, 3), ht.t_product(S, S, S), atol=1e-10)
    np.testing.assert_allclose(ht.t_power(S, -1), ht.t_inverse(S), atol=1e-8)
    np.testing.assert_allclose(ht.t_power(S, 0), I, atol=1e-12)


def test_fractional_power(rng):
    X = rand(rng, 4, 4, 5)
    P = ht.t_product(ht.t_transpose(X), X)  # t-symmetric positive definite
    R = ht.t_power(P, 0.5)
    assert np.isrealobj(R)
    np.testing.assert_allclose(ht.t_product(R, R), P, atol=1e-10)


@pytest.mark.parametrize("tube", TUBES)
def test_norm_squares_to_gram(rng, tube):
    X = rand(rng, 4, 2, *tube)
    N = ht.t_norm(X)
    np.testing.assert_allclose(ht.t_product(N, N), ht.t_product(ht.t_transpose(X), X), atol=1e-10)


@pytest.mark.parametrize("tube", TUBES)
def test_eig(rng, tube):
    A = rand(rng, 3, 3, *tube)
    V, L = ht.t_eig(A)
    np.testing.assert_allclose(ht.t_product(A, V), ht.t_product(V, L), atol=1e-10)


@pytest.mark.parametrize("tube", TUBES)
def test_eigh_is_real_and_orthonormal(rng, tube):
    S = rand(rng, 4, 4, *tube)
    A = S + ht.t_transpose(S)
    V, L = ht.t_eigh(A)
    assert np.isrealobj(V) and np.isrealobj(L)
    np.testing.assert_allclose(ht.t_product(A, V), ht.t_product(V, L), atol=1e-10)
    np.testing.assert_allclose(ht.t_product(ht.t_transpose(V), V), ht.t_eye(4, tube), atol=1e-10)
    np.testing.assert_allclose(ht.t_product(V, L, ht.t_transpose(V)), A, atol=1e-10)


def test_eigh_rejects_non_symmetric(rng):
    with pytest.raises(ValueError):
        ht.t_eigh(rand(rng, 3, 3, 4))


def test_hadamard(rng):
    A, B = rand(rng, 3, 3, 5), rand(rng, 3, 3, 5)
    expected = np.fft.ifft(np.fft.fft(A, axis=2) * np.fft.fft(B, axis=2), axis=2).real
    np.testing.assert_allclose(ht.t_hadamard(A, B), expected, atol=1e-12)


def test_fold_unfold_bcirc(rng):
    A, B = rand(rng, 3, 4, 5), rand(rng, 4, 2, 5)
    np.testing.assert_array_equal(ht.fold(ht.unfold(A), 5), A)
    np.testing.assert_allclose(ht.unfold(ht.t_product(A, B)), ht.bcirc(A) @ ht.unfold(B), atol=1e-12)


def test_diag_circ(rng):
    A = rand(rng, 3, 4, 5)
    n3 = 5
    F = np.kron(np.fft.fft(np.eye(n3)), np.eye(3))
    G = np.kron(np.fft.fft(np.eye(n3)), np.eye(4))
    np.testing.assert_allclose(ht.diag_circ(A), F @ ht.bcirc(A) @ G.conj().T / n3, atol=1e-12)


def test_t_sym_and_signal_tensor(rng):
    A = rand(rng, 3, 3, 3)
    S = ht.t_sym(A)
    assert S.shape == (3, 3, 7)
    np.testing.assert_array_equal(S[:, :, 0], 0)
    np.testing.assert_allclose(S[:, :, 1:4], A / 2)
    np.testing.assert_allclose(S[:, :, 4:], A[:, :, ::-1] / 2)
    assert ht.t_sym(rand(rng, 3, 3, 3, 3)).shape == (3, 3, 7, 7)
    assert ht.random_tensor((3, 3, 3), rng=1).shape == (3, 3, 7)

    x = np.array([1.0, 2.0, 3.0])
    X = ht.signal_tensor(x, 4)
    assert X.shape == (3, 1, 3, 3)
    assert X[1, 0, 2, 0] == x[1] * x[2] * x[0]
    assert ht.signal_tensor(x, 2).shape == (3, 1)


def test_tv(rng):
    S = rand(rng, 4, 4, 5)
    A = S + ht.t_transpose(S)
    X = rand(rng, 4, 1, 5)
    assert ht.t_tv(A, X).shape == (1, 1, 5)
    q = ht.t_tv_laplacian(A, X)
    assert q.shape == (1, 1, 5) and np.isrealobj(q)
