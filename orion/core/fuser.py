import logging
import torch
import networkx as nx

from orion.nn.activation import Chebyshev
from orion.nn.linear import Linear, Conv2d
from orion.nn.normalization import BatchNorm1d, BatchNorm2d

logger = logging.getLogger("orion")


def get_class_name(obj):
    """Get the class name of an object."""
    return obj.__class__.__name__


def format_node_tag(node_id, module):
    """Format a node identifier as 'node_id:ClassName'."""
    return f"{node_id}:{get_class_name(module)}"


class Fuser:
    """Fuse simple layer pairs inside a module DAG."""

    def __init__(self, network_dag: nx.DiGraph):
        self.network_dag = network_dag
        self.bn_linear_fusion_count = 0
        self.chebyshev_fusion_count = 0

    # ----------------------- pairwise fusions -----------------------

    def _fuse_linear_chebyshev(self, linear, cheb, parent_id=None, child_id=None):
        logger.debug(
            f"Fuse {format_node_tag(parent_id, linear)} -> {child_id}:Chebyshev  "
            f"prescale={cheb.prescale} constant={cheb.constant} "
            f"cheb.depth(before)={cheb.depth}"
        )

        linear.on_weight = linear.on_weight * cheb.prescale
        linear.on_bias = linear.on_bias * cheb.prescale + cheb.constant
        cheb.fused = True
        self.chebyshev_fusion_count += 1

        if cheb.prescale != 1:
            before = cheb.depth
            cheb.depth -= 1
            logger.debug(f"cheb.depth: {before} -> {cheb.depth} (prescale!=1)")
        else:
            logger.debug("cheb.depth unchanged (prescale==1)")

    def _fuse_bn_chebyshev(self, bn, cheb, parent_id=None, child_id=None):
        logger.debug(
            f"Fuse {format_node_tag(parent_id, bn)} -> {child_id}:Chebyshev  "
            f"prescale={cheb.prescale} constant={cheb.constant} "
            f"bn.affine={bn.affine} cheb.depth(before)={cheb.depth}"
        )

        if bn.affine:
            bn.on_weight = bn.on_weight * cheb.prescale
            bn.on_bias = bn.on_bias * cheb.prescale + cheb.constant
        else:
            bn.affine = True
            bn.on_weight = torch.ones(bn.num_features) * cheb.prescale
            bn.on_bias = torch.ones(bn.num_features) * cheb.constant

        cheb.fused = True
        self.chebyshev_fusion_count += 1

        if cheb.prescale != 1:
            before = cheb.depth
            cheb.depth -= 1
            logger.debug(f"cheb.depth: {before} -> {cheb.depth} (prescale!=1)")
        else:
            logger.debug("cheb.depth unchanged (prescale==1)")

    def _fuse_linear_bn(self, linear, bn, parent_id=None, child_id=None):
        logger.debug(
            f"Fuse {format_node_tag(parent_id, linear)} -> {child_id}:BatchNorm  "
            f"bn.affine={bn.affine} bn.depth(before)={bn.depth}"
        )

        inv_std = 1 / torch.sqrt(bn.on_running_var + bn.eps)
        scale = bn.on_weight * inv_std

        if len(linear.on_weight.shape) == 2:  # (out, in)
            linear.on_weight *= scale.reshape(-1, 1)
        else:  # (out, in, kH, kW)
            linear.on_weight *= scale.reshape(-1, 1, 1, 1)

        linear.on_bias = scale * (linear.on_bias - bn.running_mean) + bn.on_bias

        bn.fused = True
        before = bn.depth
        bn.depth -= (2 if bn.affine else 1)

        self.bn_linear_fusion_count += 1
        logger.debug(f"bn.depth: {before} -> {bn.depth} (affine={bn.affine})")

    # ----------------------- traversal helpers ----------------------

    def fuse_two_layers(self, parent_class, child_class, fuse_fn):
        """
        For every unique edge parent->child that matches the given classes,
        apply fuse_fn(parent_module, child_module, parent_id, child_id).
        """
        def find_parents(node_id):
            matches = []
            for p in self.network_dag.predecessors(node_id):
                mod = self.network_dag.nodes[p]["module"]
                if isinstance(mod, parent_class):
                    matches.append((p, mod))
            return matches

        for node_id in self.network_dag.nodes:
            child = self.network_dag.nodes[node_id]["module"]
            if not isinstance(child, child_class):
                continue

            parents = find_parents(node_id)

            if len(parents) == 0:
                logger.debug(f"Skip {format_node_tag(node_id, child)} — no parent")
                continue

            if len(parents) > 1:
                readable = ", ".join(format_node_tag(pid, pm) for pid, pm in parents)
                logger.debug(f"Skip {format_node_tag(node_id, child)} — >1 parents ({readable})")
                continue

            parent_id, parent_mod = parents[0]
            logger.debug(f"Apply fuse {format_node_tag(parent_id, parent_mod)} -> "
                         f"{format_node_tag(node_id, child)}")
            fuse_fn(parent_mod, child, parent_id, node_id)

    # --------------------------- passes -----------------------------

    def fuse_linear_chebyshev(self):
        logger.debug("Pass: Linear/Conv2d -> Chebyshev")
        self.fuse_two_layers(
            (Linear, Conv2d), Chebyshev, self._fuse_linear_chebyshev
        )

    def fuse_bn_chebyshev(self):
        logger.debug("Pass: BatchNorm -> Chebyshev")
        self.fuse_two_layers(
            (BatchNorm1d, BatchNorm2d), Chebyshev, self._fuse_bn_chebyshev
        )

    def fuse_linear_bn(self):
        logger.debug("Pass: Linear/Conv2d -> BatchNorm")
        self.fuse_two_layers(Linear, BatchNorm1d, self._fuse_linear_bn)
        self.fuse_two_layers(Conv2d, BatchNorm2d, self._fuse_linear_bn)

    # --------------------------- driver -----------------------------

    def fuse_modules(self):
        """
        Order matters:
          1) Push Chebyshev into Linear/BN.
          2) Fold BN into Linear/Conv.

        This way if we have the chain Cheb -> BN -> Conv/Linear, which
        we often do, it'll first fuse Cheb into BN, then the updated
        BN parameters into Linear.
        """
        self.bn_linear_fusion_count = 0
        self.chebyshev_fusion_count = 0

        logger.debug("Begin fusing modules.")
        self.fuse_linear_chebyshev()
        self.fuse_bn_chebyshev()
        self.fuse_linear_bn()
        logger.debug("End fusing modules.")

        logger.info(
            f"Fuser summary: BN->Conv2d fused={self.bn_linear_fusion_count}, "
            f"Chebyshev fused={self.chebyshev_fusion_count}"
        )