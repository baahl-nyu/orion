import sys
import os

import pytest
import torch
from orion.core.orion import Scheme

# Make fhe_test_utils importable from within this package
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Minimal CKKS config for fast oracle tests.
# LogN=13 (ring degree 8192), 4 levels (3 usable), 1 special prime.
# This keeps key-gen fast while still supporting polynomial eval (degree ≤ 3)
# and a few chained multiplications.
# ---------------------------------------------------------------------------
TEST_CKKS_CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [29, 26, 26, 26],
        "LogP": [29],
        "LogScale": 26,
        "H": 8192,
        "RingType": "ConjugateInvariant",
    },
    "orion": {
        "backend": "lattigo",
        "io_mode": "none",
        "debug": False,
    },
}


@pytest.fixture(scope="session")
def scheme():
    """Session-scoped Lattigo scheme shared by all tests."""
    s = Scheme()
    s.init_scheme(TEST_CKKS_CONFIG)
    yield s
    s.delete_scheme()


@pytest.fixture(scope="session")
def slots(scheme):
    """Number of plaintext slots."""
    return scheme.params.get_slots()


@pytest.fixture(scope="session")
def max_level(scheme):
    """Maximum ciphertext level."""
    return scheme.params.get_max_level()


@pytest.fixture(scope="session")
def default_scale(scheme):
    """Default plaintext scale (2^LogScale)."""
    return scheme.params.get_default_scale()
