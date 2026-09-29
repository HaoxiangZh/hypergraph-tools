# hypergraph-tools

MATLAB utilities for the **tensor t-product** and its use in hypergraph signal processing (t-HGSP).
All operations work on tensors of any order p ≥ 3: dimensions 3..p are treated as the "tube" modes,
and products are carried out slice-wise in the Fourier domain.

## Contents

### `tproduct/` — t-product algebra

| Function | Description |
|---|---|
| `t_product_fft(A, B, ...)` | t-product of two or more tensors via FFT (use this one) |
| `t_product_fft_gpu(A, B, ...)` | GPU version (Parallel Computing Toolbox) |
| `t_product(A, B)` | t-product by definition, `fold(bcirc(A) * unfold(B))`; slow, for reference |
| `ttranspose(A)` | tensor transpose |
| `t_inverse(A)` | tensor (pseudo-)inverse |
| `t_power(A, k)` | tensor power `A * A * ... * A`, any real `k` |
| `t_hadamard(A, B)` | element-wise product in the Fourier domain |
| `t_norm(X)` | tensor magnitude `(X^T * X)^(1/2)` |
| `t_eigendecomposition(A)` | `A * V = V * L`; falls back to Jordan form for defective slices (needs Symbolic Math Toolbox) |
| `bcirc(A)`, `unfold(A)`, `fold(A, p)` | block-circulant matrix and (un)folding along the last mode |
| `diag_circ(A)` | DFT block-diagonalization of `bcirc(A)` |

### `hgsp/` — hypergraph signal processing helpers

| Function | Description |
|---|---|
| `t_sym(A)` | symmetrize a tensor along its tube modes (`N x N x N` → `N x N x (2N+1)`) |
| `signal_tensor(x, M)` | build the `N x 1 x N x ... x N` tensor signal from a vector `x` |
| `t_tv(A, X)` | total variation `|X - A * X|` w.r.t. a shifting operator |
| `t_tv_laplacian(L, X)` | Laplacian quadratic form `X^T * L * X` |
| `random_tensor(tsize)` | random tensor with symmetric slices, symmetrized by `t_sym` |

### `visualization/` — understanding the t-product entry by entry (3rd order)

| Function | Description |
|---|---|
| `t_product_entrywise(A, B)` | t-product that also records which entries of `A` and `B` produce each `C(i,j,k)` |
| `printEquations(C, info)` | print the scalar equation behind every entry of `C` |
| `visualization_t_product(C, A, B, info)` | interactive 3-D view; click an entry of `C` to highlight its contributors |

## Usage

```matlab
addpath tproduct hgsp visualization

A = rand(4, 4, 5);
B = rand(4, 2, 5);
C = t_product_fft(A, B);          % 4 x 2 x 5

S  = A + ttranspose(A);
Si = t_inverse(S);                % t_product_fft(S, Si) is the identity tensor
[V, L] = t_eigendecomposition(S); % t_product_fft(S, V) == t_product_fft(V, L)

% higher order works the same way
A4 = rand(3, 3, 4, 2);
B4 = rand(3, 1, 4, 2);
C4 = t_product_fft(A4, B4);       % 3 x 1 x 4 x 2

% see how each entry of a 3rd-order product is formed
[C, info] = t_product_entrywise(rand(2,2,3), rand(2,2,3));
printEquations(C, info);
```

## Tests

```matlab
run('tests/run_tests.m')
```

The tests check algebraic identities (inverse, transpose, power, eigendecomposition) for 3rd- to
5th-order tensors, and compare every product against an independent FFT reference (`tests/ref_tprod.m`).

## References

- M. E. Kilmer and C. D. Martin, "Factorization strategies for third-order tensors," *Linear Algebra and its Applications*, 2011.
- C. D. Martin, R. Shafer, and B. LaRue, "An order-p tensor factorization with applications in imaging," *SIAM Journal on Scientific Computing*, 2013.
- K. Pena-Pena, D. L. Lau, and G. R. Arce, "t-HGSP: Hypergraph signal processing using t-product tensor decompositions," *IEEE Transactions on Signal and Information Processing over Networks*, 2023.

## License

MIT — see [LICENSE](LICENSE).
