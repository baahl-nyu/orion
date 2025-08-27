import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.fx as fx
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        mn = min(mn, x.amin().item())
        mx = max(mx, x.amax().item())
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
        """Initialize tracking attributes for all nodes."""
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
        """Execute a node and track its statistics."""
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
            if isinstance(mod, Module):
                self._sync_module_attributes(node, mod)

        return result

    def _validate_node(self, node):
        """Validate FHE compatibility constraints."""
        parents = node.all_input_nodes
        
        if parents:
            gaps = [p.output_gap for p in parents]
            if len(set(gaps)) > 1:
                msg = f"Inconsistent input gaps for {node.name}: {set(gaps)}"
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)
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

            if isinstance(sub, BatchNormNd) and len(parents) > 1:
                msg = (
                    f"BatchNorm node {node} has multiple parents which "
                    "prevents fusion"
                )
                logger.error(f"  ✗ {msg}")
                raise ValueError(msg)

    def _update_input_stats(self, node, args, kwargs):
        """Update node's input statistics."""
        tensors = list(iter_tensors((args, kwargs)))
        mn, mx = tensors_min_max(tensors)
        node.input_min = min(node.input_min, mn)
        node.input_max = max(node.input_max, mx)

        parents = node.all_input_nodes
        if parents:
            if len(parents) == 1:
                p = parents[0]
                node.input_shape = p.output_shape
                node.input_gap = p.output_gap
                node.fhe_input_shape = p.fhe_output_shape
                logger.debug(
                    f"  ← Inherited from {p.name}: "
                    f"shape={node.input_shape}, gap={node.input_gap}"
                )
                if node.fhe_input_shape != node.input_shape:
                    logger.debug(
                        f"  ← FHE shape from {p.name}: {node.fhe_input_shape}"
                    )
            else:
                node.input_shape = [p.output_shape for p in parents]
                node.input_gap = parents[0].output_gap
                node.fhe_input_shape = [p.fhe_output_shape for p in parents]
                names = [p.name for p in parents]
                logger.debug(
                    f"  ← Multiple inputs from {names}: "
                    f"shapes={node.input_shape}, gap={node.input_gap}"
                )
        else:
            node.input_shape = shape_tree(args)
            node.fhe_input_shape = node.input_shape
            logger.debug(
                f"  ← Input placeholder: "
                f"shape={node.input_shape}, fhe_shape={node.fhe_input_shape}"
            )

        logger.debug(
            f"  📊 Input stats: [{node.input_min:.4f}, {node.input_max:.4f}]"
        )

    def _update_output_stats(self, node, result):
        """Update node's output statistics."""
        tensors = list(iter_tensors(result))
        mn, mx = tensors_min_max(tensors)
        node.output_min = min(node.output_min, mn)
        node.output_max = max(node.output_max, mx)

        result_shapes = shape_tree(result)
        node.output_shape = self._compute_clear_shape(node, result_shapes)
        node.fhe_output_shape = self._compute_fhe_shape(node)
        node.output_gap = self._compute_output_gap(node)

        if (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "getitem"
        ):
            parent = node.all_input_nodes[0] if node.all_input_nodes else None
            if parent and isinstance(parent.fhe_output_shape, list):
                if len(node.args) > 1:
                    idx = node.args[1]
                    if isinstance(idx, int) and idx < len(parent.fhe_output_shape):
                        node.fhe_output_shape = parent.fhe_output_shape[idx]
                        logger.debug(
                            f"  → getitem extracted FHE shape at index {idx}: "
                            f"{node.fhe_output_shape}"
                        )

        logger.debug(
            f"  → Output: shape={node.output_shape}, gap={node.output_gap}"
        )
        if node.fhe_output_shape != node.output_shape:
            logger.debug(f"  → FHE shape: {node.fhe_output_shape}")
        logger.debug(
            f"  📊 Output stats: [{node.output_min:.4f}, {node.output_max:.4f}]"
        )

    def _compute_clear_shape(self, node, result_shapes):
        """
        Compute clear output shape.
        Only LinearTransform changes clear shape.
        """
        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, LinearTransform):
                logger.debug(
                    f"  🔄 LinearTransform shape change: "
                    f"{node.input_shape} → {result_shapes}"
                )
                return result_shapes
        
        return node.input_shape if node.input_shape else result_shapes

    def _compute_output_gap(self, node):
        """Compute output gap."""
        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, LinearTransform):
                new_gap = mod.compute_fhe_output_gap(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                )
                logger.debug(
                    f"  🔄 FHE gap change: {node.input_gap} → {new_gap}"
                )
                return new_gap
        return node.input_gap

    def _compute_fhe_shape(self, node):
        """Compute FHE output shape."""
        if not node.input_shape:
            return node.output_shape

        if node.op == "call_module":
            mod = self.module.get_submodule(node.target)
            if isinstance(mod, (LinearTransform, Add, Mult)):
                fhe_output_shape = mod.compute_fhe_output_shape(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                    fhe_input_shape=node.fhe_input_shape,
                    output_gap=node.output_gap,
                    clear_output_shape=node.output_shape,
                )
                logger.debug(
                    f"  🔄 FHE shape transformation: {node.fhe_input_shape} → "
                    f"{fhe_output_shape}"
                )
                return fhe_output_shape

        return node.fhe_input_shape

    def _sync_module_attributes(self, node, module):
        """Sync node statistics to Orion module."""
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
        """Update batch dimension in a shape structure."""
        if shape is None:
            return None
        if isinstance(shape, torch.Size):
            return torch.Size([self.batch_size] + list(shape[1:]))
        if isinstance(shape, (list, tuple)):
            updated = [self._update_shape_batch_size(s) for s in shape]
            return tuple(updated) if isinstance(shape, tuple) else updated
        return shape

    def update_batch_size(self):
        """Update batch size for all Orion modules."""
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
        """Run propagation with logging."""
        logger.debug(f"\n{'='*60}")
        logger.debug(f"Starting propagation with {len(args)} input(s)")
        logger.debug(f"{'='*60}")
        self.run(*args)
        logger.debug(f"\n{'='*60}")
        logger.debug("✓ Propagation completed successfully")
        logger.debug(f"{'='*60}")

    def _process_dataloader(self, dl):
        """Create a temporary DataLoader with larger batch size if needed."""
        if self.batch_size > dl.batch_size:
            from torch.utils.data.sampler import RandomSampler
            shuffle = dl.sampler is None or isinstance(dl.sampler, RandomSampler)
            
            logger.info(
                f"Temporarily increased batch size from {dl.batch_size} "
                f"to {self.batch_size} for faster statistics collection"
            )
            
            return DataLoader(
                dataset=dl.dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=dl.num_workers,
                pin_memory=dl.pin_memory,
                drop_last=dl.drop_last
            ), dl.batch_size
        return dl, dl.batch_size

    def _extract_batch_input(self, batch, device):
        """Extract and prepare input tensor(s) from a batch."""
        x = batch[0] if isinstance(batch, (list, tuple)) and len(batch) > 0 else batch
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return [t.to(device) for t in x]
        else:
            raise TypeError(f"Unsupported batch element type: {type(x).__name__}")

    def propagate_all(self, input_data, device='cpu', show_progress=True):
        """Run on tensors or DataLoader(s); then refresh batch size."""
        if not isinstance(input_data, list):
            input_data = [input_data]

        all_tensors = all(isinstance(x, (torch.Tensor, list)) for x in input_data)
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
            logger.info(f"Propagating through {len(input_data)} DataLoader(s)")
            
            # Process dataloaders with temporary batch size increase if needed
            temp_loaders = []
            user_batch_sizes = []
            for dl in input_data:
                temp_dl, orig_batch_size = self._process_dataloader(dl)
                temp_loaders.append(temp_dl)
                user_batch_sizes.append(orig_batch_size)
            
            # Iterate through batches
            iterator = zip(*temp_loaders)
            if show_progress:
                try:
                    total = min(len(dl) for dl in temp_loaders)
                except TypeError:
                    total = None
                iterator = tqdm(
                    iterator,
                    desc="Processing batches",
                    unit="batch",
                    total=total,
                    leave=True,
                )
            
            for batches in iterator:
                inputs = [self._extract_batch_input(b, device) for b in batches]
                self.propagate(*inputs)
            
            # Reset to original user batch size
            original_batch_size = user_batch_sizes[0]
            if len(set(user_batch_sizes)) > 1:
                logger.warning(
                    f"Multiple DataLoaders with different batch sizes detected: "
                    f"{user_batch_sizes}. Using first batch size: {original_batch_size}"
                )
            self.batch_size = original_batch_size

        else:
            types = [type(x).__name__ for x in input_data]
            raise ValueError(
                "All inputs must be either Tensors or DataLoaders, not mixed. "
                f"Got: {types}"
            )

        self.update_batch_size()