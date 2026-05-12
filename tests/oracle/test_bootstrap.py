"""Tests for bootstrap (level refresh).

Bootstrap requires larger parameters (LogN=16) and dedicated boot_logp,
so these tests are significantly slower than the others. They are marked
with @pytest.mark.slow and skipped by default.

Run them explicitly with:  pytest -m slow tests/oracle/test_bootstrap.py

The tests run against both backends. They mirror the prescale/postscale
discipline implemented by ``orion.nn.operations.Bootstrap`` so they remain
backend-agnostic: the backend is invoked at level 0 with values already in
[-1, 1], matching the production contract.
"""

import math

import pytest
import torch
from orion.core.orion import Scheme
from fhe_test_utils import assert_fhe_close


def _make_bootstrap_config(backend):
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {
            "LogP": [61, 61, 61, 61, 61, 61, 61, 61],
        },
        "orion": {
            "backend": backend,
            "io_mode": "none",
            "debug": False,
        },
    }


@pytest.fixture(scope="module", params=["lattigo", "desilo"])
def boot_scheme(request):
    """Module-scoped scheme with bootstrap support, one per backend."""
    s = Scheme()
    s.init_scheme(_make_bootstrap_config(request.param))
    yield s
    s.delete_scheme()


def _normalize_and_bootstrap(scheme, ctxt, values, num_active_slots, margin=2.0):
    """Mirror nn.operations.Bootstrap: shift + scale into [-1, 1], bootstrap,
    then undo the scale and shift. Drives the backend at level 0 with
    in-range values, which is the contract every Orion bootstrap caller
    must satisfy.

    The prescale is encoded as a *sparse* plaintext (zeros in unused slots).
    Multiplying by it both scales the active slots and zeros out the slots
    that the constant-add disturbed. Sparse bootstrap relies on those
    inactive slots being zero, so a scalar-prescale variant breaks under
    Lattigo's sparse path.
    """
    input_min = float(min(values))
    input_max = float(max(values))

    center = (input_min + input_max) / 2
    half_range = (input_max - input_min) / 2
    low = center - margin * half_range
    high = center + margin * half_range

    if high - low > 2:
        postscale = math.ceil((high - low) / 2)
        prescale = 1.0 / postscale
    else:
        postscale = 1
        prescale = 1.0

    constant = -(low + high) / 2

    slots = scheme.params.get_slots()
    prescale_vec = torch.zeros(slots, dtype=torch.float64)
    prescale_vec[:num_active_slots] = prescale

    if constant != 0:
        ctxt = ctxt + constant

    prescale_ptxt = scheme.encode(prescale_vec, level=ctxt.level())
    ctxt = ctxt * prescale_ptxt

    ctxt_btp = ctxt.bootstrap()

    if postscale != 1:
        ctxt_btp = ctxt_btp * postscale
    if constant != 0:
        ctxt_btp = ctxt_btp - constant

    return ctxt_btp


@pytest.mark.slow
class TestBootstrap:

    def test_bootstrap_restores_level(self, boot_scheme):
        """After draining levels, bootstrap should restore them."""
        scheme = boot_scheme
        slots = scheme.params.get_slots()

        # Generate a bootstrapper for the full slot count
        scheme.bootstrapper.generate_bootstrapper(slots)

        values = [float(i % 10) for i in range(slots)]
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)

        # Drain levels via repeated float multiplications. The prescale
        # multiply inside _normalize_and_bootstrap consumes the final level
        # so the backend sees ct.level == 0.
        max_level = scheme.params.get_max_level()
        drained = list(values)
        for _ in range(max_level - 1):
            ctxt = ctxt * 1.1
            drained = [v * 1.1 for v in drained]

        level_before_btp = ctxt.level()
        assert level_before_btp <= 1, (
            f"Expected low level before bootstrap, got {level_before_btp}")

        ctxt_btp = _normalize_and_bootstrap(scheme, ctxt, drained, slots)
        level_after_btp = ctxt_btp.level()

        assert level_after_btp > level_before_btp, (
            f"Bootstrap should raise level: was {level_before_btp}, "
            f"now {level_after_btp}")

        result = scheme.decode(scheme.decrypt(ctxt_btp))

        assert_fhe_close(
            result, drained, atol=5e-1, msg="bootstrap values")

    def test_bootstrap_sparse(self, boot_scheme):
        """Sparse bootstrap (fewer slots than available)."""
        scheme = boot_scheme
        num_values = 64  # fewer than full slots

        scheme.bootstrapper.generate_bootstrapper(num_values)

        values = [float(i + 1) for i in range(num_values)]
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)

        max_level = scheme.params.get_max_level()
        drained = list(values)
        for _ in range(max_level - 1):
            ctxt = ctxt * 1.1
            drained = [v * 1.1 for v in drained]

        ctxt_btp = _normalize_and_bootstrap(scheme, ctxt, drained, num_values)
        assert ctxt_btp.level() > 1

        result = scheme.decode(scheme.decrypt(ctxt_btp))

        assert_fhe_close(
            result[:num_values], drained, atol=5e-1,
            msg="sparse bootstrap values")
