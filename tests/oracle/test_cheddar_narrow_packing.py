"""Correctness tests for per-layer narrow slot-width packing on the
Cheddar backend (see packing.propagate_physical_slots).

These are engineered to specifically hit the failure mode caught during
design review: a layer's "hybrid" packing fold only *logically* compresses
its output (via output-rotation summation), it never narrows the
ciphertext's *physical* width -- so a downstream layer that computed its
own packing width independently from its own (narrower) logical shape
would build diagonals for the wrong physical width, causing silently
wrong rotation wraparound rather than a crash. Two scenarios are covered:

  1. A layer whose output is narrower than its input (forces
     output_rotations > 0 under the hybrid embedding method).
  2. A residual-style two-branch merge (Add) followed by another layer.
     With fold-retag narrowing active (the default), both branches'
     folded outputs are retagged down to their common 512-periodic
     width and the downstream layer packs at that folded width; the
     end-to-end MAE assertion is what verifies the merge stays correct
     across the retag.

Run explicitly with:  pytest -m slow tests/oracle/test_cheddar_narrow_packing.py
"""

import math

import pytest
import torch

import orion
import orion.nn as on
from orion.core.utils import mae


# Small ring (LogN=13, 4096 slots) for fast key-gen; enough usable levels
# (6) that a 3-deep linear chain never needs to bootstrap, so this stays
# focused purely on packing/rotation correctness. No boot_params on
# purpose -- this network should never need one.
CHEDDAR_CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [55, 40, 40, 40, 40, 40],
        "LogP": [61, 61, 61],
        "LogScale": 40,
        "H": 192,
        "RingType": "Standard",
    },
    "orion": {
        "backend": "cheddar",
        "device": "gpu",
        "io_mode": "none",
        "debug": False,
        "margin": 2,
    },
}


class NarrowPackingNet(on.Module):
    """fc1 widens (512->1024), fc2a/fc2b independently narrow the same
    1024-wide input down to 512 (forcing output_rotations > 0 on both --
    scenario 1), their outputs merge via Add (scenario 2: the merge must
    reconcile both branches' physical width), and fc3 narrows further
    still (256), inheriting whatever physical width the merge left it
    with rather than its own logical 512 input shape.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = on.Linear(512, 1024)
        self.fc2a = on.Linear(1024, 512)
        self.fc2b = on.Linear(1024, 512)
        self.add = on.Add()
        self.fc3 = on.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        a = self.fc2a(x)
        b = self.fc2b(x)
        x = self.add(a, b)
        return self.fc3(x)


@pytest.mark.slow
class TestCheddarNarrowPacking:

    def test_width_mismatch_and_branch_merge(self):
        torch.manual_seed(42)

        orion.init_scheme(CHEDDAR_CONFIG)
        net = NarrowPackingNet()
        net.eval()

        inp = torch.randn(1, 512)

        out_clear = net(inp)

        orion.fit(net, inp)
        input_level = orion.compile(net)

        # Sanity-check the propagation pass actually did something
        # non-trivial before trusting the FHE-vs-cleartext comparison
        # below to catch a regression -- if these are all still the
        # scheme's global max (4096), the test isn't exercising the new
        # code path at all.
        max_slots = net.scheme.params.get_slots()
        assert net.fc1.slots < max_slots
        assert net.fc2a.output_rotations > 0, (
            "expected fc2a to hit the hybrid output-rotation fold "
            "(narrower own-output than inherited physical input width)")
        assert net.fc2b.output_rotations > 0
        # Fold-retag narrowing (see propagate_physical_slots): both
        # branches' hybrid folds leave their outputs 512-periodic, so
        # their ciphertexts are retagged down to 512 at runtime and fc3
        # correctly packs at that *folded* width -- narrower than the
        # branches' own 1024 pack width. (Before the fold-retag resize
        # primitive existed, the invariant here was the opposite: fc3
        # had to inherit the full 1024 because nothing could narrow a
        # ciphertext mid-network. ORION_CHEDDAR_DISABLE_FOLD_RETAG=1
        # restores that behavior.)
        assert net.fc2a.folded_slots == net.fc2b.folded_slots == 512, (
            "expected both branches to be marked for fold-retag "
            "narrowing to 512")
        assert net.fc3.slots == 512, (
            "fc3 must pack at the branches' folded (retagged) width, "
            "which the Add merge carries forward")
        assert net.fc3.folded_slots is None, (
            "the last LinearTransform uses the square method (no fold) "
            "and must never be marked for retag")

        vec_ptxt = orion.encode(inp, input_level)
        vec_ctxt = orion.encrypt(vec_ptxt)
        net.he()

        out_ctxt = net(vec_ctxt)
        out_fhe = out_ctxt.decrypt().decode()

        dist = mae(out_clear, out_fhe)
        assert dist < 0.01, f"FHE vs cleartext MAE too high: {dist}"

        orion.delete_scheme()
