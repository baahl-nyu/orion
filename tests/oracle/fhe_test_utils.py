"""Shared utilities for FHE differential tests."""

import torch


def assert_fhe_close(actual, expected, atol=1e-2, rtol=0, msg=""):
    """Compare FHE output against expected cleartext values.

    Parameters
    ----------
    actual : torch.Tensor | list
        Decrypted + decoded result from the FHE pipeline.
    expected : torch.Tensor | list
        Cleartext ground-truth values.
    atol : float
        Absolute tolerance (default 0.01 — generous for CKKS).
    rtol : float
        Relative tolerance.
    msg : str
        Optional message prepended to assertion errors.
    """
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float64)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=torch.float64)

    actual = actual.double().flatten()
    expected = expected.double().flatten()

    # Trim to the shorter length (FHE may pad to slot boundary)
    n = min(len(actual), len(expected))
    actual = actual[:n]
    expected = expected[:n]

    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    failures = diff > tol

    if failures.any():
        max_err_idx = diff.argmax().item()
        detail = (
            f"Max error: {diff[max_err_idx]:.6e} at index {max_err_idx} "
            f"(actual={actual[max_err_idx]:.6f}, "
            f"expected={expected[max_err_idx]:.6f}).  "
            f"{failures.sum().item()}/{n} slots exceeded atol={atol}."
        )
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(f"{prefix}{detail}")


def encrypt_values(scheme, values, level=None):
    """Encode → encrypt a list of floats."""
    ptxt = scheme.encode(values, level=level)
    return scheme.encrypt(ptxt)


def decrypt_values(scheme, ctxt):
    """Decrypt → decode a CipherTensor back to torch.Tensor."""
    return scheme.decode(scheme.decrypt(ctxt))
