"""Compare the Python port with outputs saved from the MATLAB toolbox.

The reference file is produced by ``tests/data/generate_matlab_reference.m``.
"""

from pathlib import Path

import numpy as np
import pytest

import hypergraph_tools as ht

scipy_io = pytest.importorskip("scipy.io")

REF = Path(__file__).parent / "data" / "matlab_reference.mat"


@pytest.fixture(scope="module")
def ref():
    data = scipy_io.loadmat(REF, squeeze_me=False, struct_as_record=False)
    out = data.pop("out")[0, 0]
    return data, out


CASES = {
    "t_product_3": lambda d: ht.t_product(d["A3"], d["B3"]),
    "t_product_4": lambda d: ht.t_product(d["A4"], d["B4"]),
    "t_product_5": lambda d: ht.t_product(d["A5"], d["B5"]),
    "t_product_complex_4": lambda d: ht.t_product(d["Ac4"], d["Bc4"]),
    "t_product_multi_3": lambda d: ht.t_product(d["A3"], d["B3"], d["C3"]),
    "t_product_def_4": lambda d: ht.t_product_bcirc(d["A4"], d["B4"]),
    "ttranspose_3": lambda d: ht.t_transpose(d["A3"]),
    "ttranspose_4": lambda d: ht.t_transpose(d["A4"]),
    "ttranspose_complex_4": lambda d: ht.t_transpose(d["Ac4"]),
    "t_inverse_3": lambda d: ht.t_inverse(d["S3"]),
    "t_inverse_4": lambda d: ht.t_inverse(d["S4"]),
    "t_power_3_2": lambda d: ht.t_power(d["S3"], 2),
    "t_power_3_m1": lambda d: ht.t_power(d["S3"], -1),
    "t_power_4_3": lambda d: ht.t_power(d["S4"], 3),
    "t_hadamard_3": lambda d: ht.t_hadamard(d["S3"], d["Ss3"]),
    "t_norm_3": lambda d: ht.t_norm(d["B3"]),
    "bcirc_3": lambda d: ht.bcirc(d["A3"]),
    "bcirc_4": lambda d: ht.bcirc(d["A4"]),
    "unfold_4": lambda d: ht.unfold(d["A4"]),
    "diag_circ_3": lambda d: ht.diag_circ(d["A3"]),
    "t_sym_3": lambda d: ht.t_sym(d["T3"]),
    "t_sym_4": lambda d: ht.t_sym(d["T4"]),
    "signal_tensor_4": lambda d: ht.signal_tensor(d["x"], 4),
    "t_tv_3": lambda d: ht.t_tv(d["Ss3"], d["xs"]),
    "t_tv_laplacian_3": lambda d: ht.t_tv_laplacian(d["Ss3"], d["xs"]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_matlab(ref, name):
    data, out = ref
    expected = getattr(out, name)
    got = CASES[name](data)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10)
