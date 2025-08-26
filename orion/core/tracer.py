import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.fx as fx
from torch.utils.data import DataLoader

import orion.nn as on
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.nn.normalization import BatchNormNd
from orion.nn.operations import Add, Mult

logger = logging.getLogger("orion")


# ------------------------------- utilities -------------------------------- #

def iter_tensors(obj):
    """Yield all tensors found recursively in obj."""
    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if isinstance(obj, nn.Module):
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from iter_tensors(v)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from iter_tensors(v)


def shape_tree(obj):
    """Return a shape-structured mirror of obj."""
    if isinstance(obj, torch.Tensor):
        return obj.shape
    if isinstance(obj, Mapping):
        return {k: shape_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        shapes = [shape_tree(v) for v in obj]
        return shapes[0] if len(shapes) == 1 else (
            tuple(shapes) if isinstance(obj, tuple) else shapes
        )
    return None


def tensors_min_max(tensors):
    """Return (min, max) across tensors; inf/-inf if none."""
    mn, mx = float("inf"), float("-inf")
    for t in tensors:
        if t.numel() == 0:
            continue
        x = t.detach()
        mn = min(mn, x.amin())
        mx = max(mx, x.amax())
    return mn, mx


# ------------------------------- tracer ----------------------------------- #

class OrionTracer(fx.Tracer):
    """
    Deeper trace than default: non-container modules with no children are
    considered leaves.
    """
    def is_leaf_module(self, m, _):
        if not isinstance(m, nn.Module):
            return False
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        return not any(m.children())

    def trace_model(self, model):
        logger.info(f"Starting trace of model: {model.__class__.__name__}")

        # Wrap single-leaf models for consistent tracing
        if self.is_leaf_module(model, ""):
            logger.debug(
                f"Wrapping leaf module {model.__class__.__name__} for tracing"
            )
            model = ModuleWrapper(model)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = super().trace(model)

        gm = fx.GraphModule(model, graph)
        logger.info(f"Trace completed. Graph has {len(gm.graph.nodes)} nodes")
        return gm


class ModuleWrapper(on.Module):
    """Wrapper for leaf modules to make them traceable."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


# ------------------------------- interpreter ------------------------------- #

class StatsTracker(fx.Interpreter):
    """Run the graph and record ranges, shapes, and FHE shape/gap metadata."""

    def __init__(self, module, batch_size):
        super().__init__(module)
        logger.info(
            "Initializing StatsTracker for graph with "
            f"{len(module.graph.nodes)} nodes"
        )
        self.batch_size = batch_size
        self._init_node_attributes()

    def _init_node_attributes(self):
        attrs = dict(
            input_min=float("inf"),
            input_max=float("-inf"),
            output_min=float("inf"),
            output_max=float("-inf"),
            input_shape=None,
            output_shape=None,
            fhe_input_shape=None,
            fhe_output_shape=None,
            input_gap=1,
            output_gap=1,
        )
        for node in self.module.graph.nodes:
            for k, v in attrs.items():
                setattr(node, k, v)

    def run_node(self, node):
        parents = [p.name for p in node.all_input_nodes]
        if parents:
            logger.debug(
                f"\n→ Running {node.name} (op: {node.op}) "
                f"with inputs from: {parents}"
            )
        else:
            logger.debug(f"\n→ Running {node.name} (op: {node.op})")

        self._validate_node(node)

        args = self.map_nodes_to_values(node.args, node)
        kwargs = self.map_nodes_to_values(node.kwargs, node)

        if args or kwargs:
            self._update_input_stats(node, args, kwargs)

        result = super().run_node(node)
        self._update_output_stats(node, result)

        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            self._sync_module_attributes(node, mod)

        return result

    def _validate_node(self, node):
        parents = node.all_input_nodes
        if parents:
            gaps = [p.output_gap for p in parents if hasattr(p, "output_gap")]
            if len(set(gaps)) > 1:
                msg = f"Inconsistent input gaps for {node.name}: {set(gaps)}"
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)
            if gaps:
                logger.debug(f"  ✓ Gap check passed (gap={gaps[0]})")

        if node.op == "call_module":
            sub = self.module.get_submodule(node.target)

            stride = getattr(sub, "stride", None)
            if stride and len(set(stride)) > 1:
                msg = (
                    f"Stride for {node.name} must be equal in all directions: "
                    f"{stride}"
                )
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)

            if isinstance(sub, BatchNormNd) and len(node.all_input_nodes) > 1:
                msg = (
                    f"BatchNorm node {node} has multiple parents which "
                    "prevents fusion"
                )
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)

    def _update_input_stats(self, node, args, kwargs):
        tensors = list(iter_tensors((args, kwargs)))
        mn, mx = tensors_min_max(tensors)
        node.input_min = min(node.input_min, mn)
        node.input_max = max(node.input_max, mx)

        # Fast path: binary pointwise call_function with shared FHE shape
        if node.op == "call_function" and len(node.all_input_nodes) == 2:
            p0, p1 = node.all_input_nodes
            if p0.fhe_output_shape != p1.fhe_output_shape:
                msg = (
                    f"call_function {node.name} expects equal FHE shapes: "
                    f"{p0.fhe_output_shape} vs {p1.fhe_output_shape}"
                )
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)

            node.input_shape = p0.output_shape
            node.fhe_input_shape = p0.fhe_output_shape
            node.input_gap = p0.output_gap
            logger.debug(
                "  ← call_function unified FHE shape: "
                f"{node.fhe_input_shape}"
            )

        else:
            if node.all_input_nodes:
                if len(node.all_input_nodes) == 1:
                    p = node.all_input_nodes[0]
                    node.input_shape = p.output_shape
                    node.input_gap = p.output_gap
                    node.fhe_input_shape = p.fhe_output_shape
                    logger.debug(
                        f"  ← Inherited from {p.name}: "
                        f"shape={node.input_shape}, gap={node.input_gap}"
                    )
                    if node.fhe_input_shape != node.input_shape:
                        logger.debug(
                            "  ← FHE shape from "
                            f"{p.name}: {node.fhe_input_shape}"
                        )
                else:
                    node.input_shape = [
                        p.output_shape for p in node.all_input_nodes
                    ]
                    node.input_gap = node.all_input_nodes[0].output_gap
                    node.fhe_input_shape = [
                        p.fhe_output_shape for p in node.all_input_nodes
                    ]
                    names = [p.name for p in node.all_input_nodes]
                    logger.debug(
                        f"  ← Multiple inputs from {names}: "
                        f"shapes={node.input_shape}, gap={node.input_gap}"
                    )
            else:
                node.input_shape = shape_tree(args)
                node.fhe_input_shape = node.input_shape
                logger.debug(
                    "  ← Input placeholder: "
                    f"shape={node.input_shape}, "
                    f"fhe_shape={node.fhe_input_shape}"
                )

        logger.debug(
            f"  📊 Input stats: [{node.input_min:.4f}, {node.input_max:.4f}]"
        )

    def _update_output_stats(self, node, result):
        tensors = list(iter_tensors(result))
        mn, mx = tensors_min_max(tensors)
        node.output_min = min(node.output_min, mn)
        node.output_max = max(node.output_max, mx)

        result_shapes = shape_tree(result)
        node.output_shape = self.compute_clear_output_shape(
            node, result_shapes
        )
        node.fhe_output_shape = self.compute_fhe_output_shape(node)
        node.output_gap = self.compute_fhe_output_gap(node)

        # Preserve FHE shape for getitem when parent exposes a list
        if (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "getitem"
        ):
            parent = node.all_input_nodes[0] if node.all_input_nodes else None
            if parent and isinstance(parent.fhe_output_shape, list):
                if len(node.args) > 1:
                    idx = node.args[1]
                    if isinstance(idx, int) and idx < len(
                        parent.fhe_output_shape
                    ):
                        node.fhe_output_shape = parent.fhe_output_shape[idx]
                        logger.debug(
                            "  → getitem extracted FHE shape at index "
                            f"{idx}: {node.fhe_output_shape}"
                        )

        logger.debug(
            f"  → Output: shape={node.output_shape}, gap={node.output_gap}"
        )
        if node.fhe_output_shape != node.output_shape:
            logger.debug(f"  → FHE shape: {node.fhe_output_shape}")
        logger.debug(
            f"  📊 Output stats: [{node.output_min:.4f}, "
            f"{node.output_max:.4f}]"
        )

    def compute_clear_output_shape(self, node, result_shapes):
        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, LinearTransform):
                in_shape = (
                    node.input_shape[0]
                    if isinstance(node.input_shape, list)
                    else node.input_shape
                )
                logger.debug(
                    f"  🔄 LinearTransform shape change: "
                    f"{in_shape} → {result_shapes}"
                )
                return result_shapes
        return result_shapes if result_shapes is not None else node.input_shape

    def compute_fhe_output_gap(self, node):
        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, LinearTransform):
                in_shape = (
                    node.input_shape[0]
                    if isinstance(node.input_shape, list)
                    else node.input_shape
                )
                out_shape = (
                    node.output_shape[0]
                    if isinstance(node.output_shape, list)
                    else node.output_shape
                )
                new_gap = mod.compute_fhe_output_gap(
                    input_gap=node.input_gap,
                    input_shape=in_shape,
                    output_shape=out_shape,
                )
                logger.debug(
                    f"  🔄 FHE gap change: {node.input_gap} → {new_gap}"
                )
                return new_gap
        return node.input_gap

    def compute_fhe_output_shape(self, node):
        # Placeholders won't have input_shape yet; carry clear output.
        if not node.input_shape:
            return node.output_shape

        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, (LinearTransform, Add, Mult)):
                fhe_shape = mod.compute_fhe_output_shape(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                    fhe_input_shape=node.fhe_input_shape,
                    output_gap=node.output_gap,
                    clear_output_shape=node.output_shape,
                )
                logger.debug(
                    "  🔄 FHE shape transformation: "
                    f"{node.fhe_input_shape} → {fhe_shape}"
                )
                return fhe_shape

        return node.fhe_input_shape

    def _sync_module_attributes(self, node, module):
        logger.debug(
            "  Module type check: %s, is Module: %s",
            type(module).__name__, isinstance(module, Module)
        )
        if not isinstance(module, Module):
            logger.debug(
                "  ⊘ Skipping sync for PyTorch module: %s",
                type(module).__name__,
            )
            return

        module.name = node.name

        module.input_min = node.input_min
        module.input_max = node.input_max
        module.output_min = node.output_min
        module.output_max = node.output_max

        module.input_shape = node.input_shape
        module.output_shape = node.output_shape
        module.fhe_input_shape = node.fhe_input_shape
        module.fhe_output_shape = node.fhe_output_shape

        module.input_gap = node.input_gap
        module.output_gap = node.output_gap

        logger.debug(
            "  ✓ Synced to Orion module: %s (type: %s)",
            node.name, type(module).__name__,
        )
        logger.debug(
            "    - input_shape: %s, fhe_input_shape: %s",
            module.input_shape, module.fhe_input_shape,
        )
        logger.debug(
            "    - output_shape: %s, fhe_output_shape: %s",
            module.output_shape, module.fhe_output_shape,
        )

    def _update_shape_batch_size(self, shape):
        if shape is None:
            return None
        if isinstance(shape, torch.Size):
            return torch.Size([self.batch_size] + list(shape[1:]))
        if isinstance(shape, (list, tuple)):
            updated = [self._update_shape_batch_size(s) for s in shape]
            return tuple(updated) if isinstance(shape, tuple) else updated
        return shape

    def update_batch_size(self):
        logger.info(
            f"\nUpdating batch size to {self.batch_size} for all "
            "Orion modules..."
        )
        updated = 0
        for node in self.module.graph.nodes:
            if node.op != "call_module":
                continue
            mod = self.module.get_submodule(node.target)
            if not isinstance(mod, Module):
                continue

            old_in = mod.input_shape

            mod.input_shape = self._update_shape_batch_size(mod.input_shape)
            mod.output_shape = self._update_shape_batch_size(mod.output_shape)
            mod.fhe_input_shape = self._update_shape_batch_size(
                mod.fhe_input_shape
            )
            mod.fhe_output_shape = self._update_shape_batch_size(
                mod.fhe_output_shape
            )

            logger.debug(f"  {node.name}: {old_in} → {mod.input_shape}")
            updated += 1
        logger.info(f"✓ Updated batch size for {updated} Orion modules\n")

    def propagate(self, *args):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting propagation with {len(args)} input(s)")
        logger.info(f"{'='*60}")
        self.run(*args)
        logger.info(f"\n{'='*60}")
        logger.info("✓ Propagation completed successfully")
        logger.info(f"{'='*60}")

    def propagate_all(self, input_data, device='cpu', show_progress=True):
        """Run on tensors (or lists of tensors). DataLoader support deferred."""
        if not isinstance(input_data, list):
            input_data = [input_data]

        all_tensors = all(isinstance(x, (torch.Tensor, list))
                          for x in input_data)
        all_loaders = all(isinstance(x, DataLoader) for x in input_data)

        if all_tensors:
            logger.info(
                "Propagating input data structure: " +
                str([
                    ("List[{0}]".format(len(x)) if isinstance(x, list)
                     else type(x).__name__)
                    for x in input_data
                ])
            )
            inputs = []
            for x in input_data:
                if isinstance(x, torch.Tensor):
                    inputs.append(x.to(device))
                else:
                    inputs.append([t.to(device) for t in x])
            self.propagate(*inputs)

        elif all_loaders:
            logger.info("Propagating through DataLoader(s)")
            raise NotImplementedError(
                "DataLoader support needs updating for new structure"
            )
        else:
            types = [type(x).__name__ for x in input_data]
            raise ValueError(
                "All inputs must be either Tensors or DataLoaders, not mixed. "
                f"Got: {types}"
            )

        self.update_batch_size()