"""End-to-end correctness test for the real ResNet stride-2 downsampling
transitions on the Cheddar backend, under narrow (per-layer) slot-width
packing (see packing.propagate_physical_slots).

Uses a minimal 1-block-per-stage ResNet (same ResNet/BasicBlock classes
as the real ResNet20 in orion.models.resnet, just num_blocks=[1,1,1]
instead of [3,3,3]) so it still exercises both real stride-2 transitions
-- stage2's first block (16ch@32x32 -> 32ch@16x16) and stage3's first
block (32ch@16x16 -> 64ch@8x8), each with the shortcut branch's own
stride-2 1x1 conv merging via Add -- without the cost of the full
20-layer network. This is the highest-risk part of the change: each
transition changes channel count and shrinks spatial size in the same
step, so the branch reconciliation in propagate_physical_slots must
correctly widen the shortcut's 1x1 conv to match the main path's
physical width (or vice versa) before they merge.

Run explicitly with:  pytest -m slow tests/oracle/test_resnet_stage_transition.py
"""

from pathlib import Path

import pytest
import torch

import orion
from orion.core.utils import mae
from orion.models.resnet import ResNet, BasicBlock, get_resnet_config


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


@pytest.mark.slow
class TestResNetStageTransition:

    def test_mini_resnet_stage_transitions(self):
        torch.manual_seed(42)

        # This test cares about packing/rotation correctness, not the
        # per-layer decrypt-and-verify debug tax (see debug=True in
        # nn/module.py's timer()) -- keep it off so the run stays fast
        # and focused.
        import yaml
        with open(get_config_path("resnet_cheddar.yml")) as f:
            cfg = yaml.safe_load(f)
        cfg["orion"]["debug"] = False
        orion.init_scheme(cfg)

        conv1_params, num_classes = get_resnet_config("cifar10")
        net = ResNet("cifar10", BasicBlock, [1, 1, 1], [16, 32, 64],
                     conv1_params, num_classes)
        net.eval()

        inp = torch.randn(1, 3, 32, 32)
        out_clear = net(inp)

        orion.fit(net, inp)
        input_level = orion.compile(net)

        # Confirm both stride-2 transitions actually exercise narrower-
        # than-global packing and that the shortcut/main-path branches
        # reconciled to the same physical width before their Add --
        # otherwise this test wouldn't be exercising the scenario it's
        # meant to cover.
        max_slots = net.scheme.params.get_slots()
        stage2_block = net.layers[1][0]
        stage3_block = net.layers[2][0]
        for stage_name, block in [("stage2", stage2_block), ("stage3", stage3_block)]:
            assert block.conv1.slots < max_slots, (
                f"{stage_name} conv1 should pack narrower than the global max")
            shortcut_conv = block.shortcut[0]
            assert shortcut_conv.physical_slots == block.conv2.physical_slots, (
                f"{stage_name}: main path (conv2) and shortcut branch must "
                f"reconcile to the same physical width before Add merges them")

        vec_ptxt = orion.encode(inp, input_level)
        vec_ctxt = orion.encrypt(vec_ptxt)
        net.he()

        out_ctxt = net(vec_ctxt)
        out_fhe = out_ctxt.decrypt().decode()

        dist = mae(out_clear, out_fhe)
        assert dist < 0.05, f"FHE vs cleartext MAE too high: {dist}"

        orion.delete_scheme()
