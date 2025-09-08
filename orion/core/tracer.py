from collections.abc import Mapping

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import nn as on
from .logger import logger, TracerDashboard



# ----------------------------- Utilities ----------------------------- #

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


# ----------------------------- Classes ----------------------------- #

class Node:
    """
    Statistics tracker for a single node in the computation graph.
    
    Independent node that can be instantiated with just an fx.Node and
    optionally a module. Validation and statistics tracking are self-contained.
    """
    def __init__(self, fx_node: fx.Node, module=None):
        self.fx_node = fx_node
        self.module = module 
        fx_node.stats = self
        
        # Min/max value tracking
        self.input_min = float("inf")
        self.input_max = float("-inf")
        self.output_min = float("inf")
        self.output_max = float("-inf")
        
        # Shape tracking
        self.input_shape = None
        self.output_shape = None
        self.fhe_input_shape = None
        self.fhe_output_shape = None
        
        # Gap tracking
        self.input_gap = 1
        self.output_gap = 1

    @property
    def name(self):
        return self.fx_node.name
    
    @property
    def op(self):
        return self.fx_node.op
    
    @property
    def target(self):
        return self.fx_node.target
    
    @property
    def all_input_nodes(self):
        return self.fx_node.all_input_nodes
    
    @property
    def args(self):
        return self.fx_node.args
    
    @property
    def kwargs(self):
        return self.fx_node.kwargs
    
    def validate(self) -> None:
        """Validate FHE constraints: parent gaps, stride, BN parent count."""
        parents = self.all_input_nodes

        # 1) All parent gaps must match
        if parents:
            gaps = {p.stats.output_gap for p in parents}
            if len(gaps) != 1:
                self._raise_validation_error(
                    f"Inconsistent input gaps for {self.name}: {gaps}"
                )

        # 2) Call function-level checks
        if parents and self.op == "call_function":
            if len(parents) > 1:  
                common_shape = parents[0].stats.fhe_output_shape  
                
                for p in parents[1:]:
                    if p.stats.fhe_output_shape != common_shape:
                        all_shapes = [p.stats.fhe_output_shape for p in parents]
                        self._raise_validation_error(
                            f"Inconsistent FHE input shapes for {self.name}: "
                            f"{all_shapes}"
                        )
            
        # 3) Module-level checks
        if not self.module:
            return

        # Stride must be uniform
        stride = getattr(self.module, "stride", None)
        if stride is not None:
            vals = (set(stride) if isinstance(stride, (tuple, list))
                    else {stride})
            if len(vals) != 1:
                self._raise_validation_error(
                    "Stride for {0} must be equal in all directions: {1}"
                    .format(self.name, stride)
                )

        # BatchNorm cannot have multiple parents
        if isinstance(self.module, on.BatchNormNd) and len(parents) > 1:
            self._raise_validation_error(
                f"BatchNorm node {self.name} has multiple parents which "
                "prevents fusion"
            )
    
    def _raise_validation_error(self, msg):
        """Log and raise a validation error."""
        logger.error(f"  ✗ {msg}")
        raise ValueError(msg)
    
    # ---- Statistics update methods ----
    def update_input_stats(self, args, kwargs):
        """Update node's input statistics."""
        tensors = list(iter_tensors((args, kwargs)))
        mn, mx = tensors_min_max(tensors)
        self.input_min = min(self.input_min, mn)
        self.input_max = max(self.input_max, mx)

        # Skip generating shapes if already done once 
        if self.input_shape is not None:
            return

        parents = self.all_input_nodes
        if parents:
            if len(parents) == 1:
                p = parents[0].stats
                self.input_shape = p.output_shape
                self.input_gap = p.output_gap
                self.fhe_input_shape = p.fhe_output_shape
            else:
                self.input_shape = [p.stats.output_shape for p in parents]
                self.input_gap = parents[0].stats.output_gap
                self.fhe_input_shape = [p.stats.fhe_output_shape for p in parents]
        else:
            self.input_shape = shape_tree(args)
            self.fhe_input_shape = self.input_shape
        
        self.log_input_stats()
    
    def update_output_stats(self, result):
        """Update node's output statistics."""
        tensors = list(iter_tensors(result))
        mn, mx = tensors_min_max(tensors)
        self.output_min = min(self.output_min, mn)
        self.output_max = max(self.output_max, mx)

        result_shapes = shape_tree(result)
        self.output_shape = self._compute_clear_output_shape(result_shapes)
        self.fhe_output_shape = self._compute_fhe_output_shape()
        self.output_gap = self._compute_fhe_output_gap()

        self._handle_getitem_fhe_shape()
        self.log_output_stats()
    
    def _compute_clear_output_shape(self, result_shapes):
        """Compute cleartext output shape."""
        if isinstance(self.module, on.Module):
            if self.module.preserve_input_shapes:
                return self.input_shape
            
        return result_shapes
        
    def _compute_fhe_output_gap(self):
        """Compute output gap."""
        if isinstance(self.module, on.Module):
            if self.module.preserve_input_shapes:
                return self.input_gap 
                
            new_gap = self.module.compute_fhe_output_gap(
                input_gap=self.input_gap,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                fhe_input_shape=self.fhe_input_shape,
                output_gap=self.output_gap,
                clear_output_shape=self.output_shape,
            )
            return new_gap
        return self.input_gap

    def _compute_fhe_output_shape(self):
        """Compute FHE output shape."""
        if not self.input_shape:
            return self.output_shape

        if isinstance(self.module, on.Module):
            if self.module.preserve_input_shapes:
                return self.fhe_input_shape
            
            fhe_output_shape = self.module.compute_fhe_output_shape(
                input_gap=self.input_gap,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                fhe_input_shape=self.fhe_input_shape,
                output_gap=self.output_gap,
                clear_output_shape=self.output_shape,
            )
            return fhe_output_shape
        
        # We only trace call functions (torch.mul, operator.add, etc.) that
        # preserve the input shape when accepting more than one argument.
        # (e.g. a + b = c), where a, b, and, c all have the same sizes.
        # Here fhe_input_shape is a list, and its FHE output shape is the
        # same as its inputs. Just grab the first shape.
        if (self.op == "call_function" and isinstance(self.fhe_input_shape, list)):
            return self.fhe_input_shape[0] 

        return self.fhe_input_shape

    def _handle_getitem_fhe_shape(self):
        """Handle getitem special case for FHE shapes."""
        if (self.op == "call_function"
                and getattr(self.target, "__name__", "") == "getitem"
                and self.all_input_nodes
                and len(self.args) > 1
                and isinstance(self.args[1], int)):
            idx = self.args[1]
            parent_stats = self.all_input_nodes[0].stats
            shapes = parent_stats.fhe_output_shape
            if isinstance(shapes, (list, tuple)) and 0 <= idx < len(shapes):
                self.fhe_output_shape = shapes[idx]
                logger.debug(
                    "  → getitem extracted FHE shape[%d]: %s",
                    idx, self.fhe_output_shape,
                )
    
    def sync_to_module(self):
        """Sync node statistics to associated Orion module."""
        if not isinstance(self.module, on.Module):
            return
            
        self.module.name = self.name
        
        self.module.input_min = self.input_min
        self.module.input_max = self.input_max
        self.module.output_min = self.output_min
        self.module.output_max = self.output_max
        
        self.module.input_shape = self.input_shape
        self.module.output_shape = self.output_shape
        self.module.fhe_input_shape = self.fhe_input_shape
        self.module.fhe_output_shape = self.fhe_output_shape
        
        self.module.input_gap = self.input_gap
        self.module.output_gap = self.output_gap
        
        logger.debug(
            f"Synced to Orion module: {self.name} "
            f"(type: {type(self.module).__name__})"
        )

    def update_module_batch_size(self, user_batch_size: int):
        """Replace batch size (first dim) with user_batch_size."""
        if not isinstance(self.module, on.Module):
            return

        def update_shape(shape):
            if shape is None:
                return None
            if isinstance(shape, torch.Size):
                remaining_dims = list(shape[1:])
                return torch.Size([user_batch_size] + remaining_dims)
            if isinstance(shape, (list, tuple)):
                updated_shapes = [update_shape(s) for s in shape]
                if isinstance(shape, tuple):
                    return tuple(updated_shapes)
                return updated_shapes
                
            return shape

        shape_attributes = [
            'input_shape', 
            'output_shape',
            'fhe_input_shape', 
            'fhe_output_shape'
        ]

        for attribute_name in shape_attributes:
            current_shape = getattr(self.module, attribute_name)
            updated_shape = update_shape(current_shape)
            setattr(self.module, attribute_name, updated_shape)

    # ---- Logging methods ----
    def log_execution(self):
        """Log the node being executed and its potential parents."""
        parents = [p.name for p in self.all_input_nodes]
        msg = f"Running {self.name} (op: {self.op})"
        if parents:
            msg += " [dim]with inputs from:[/] " + ", ".join(parents)

        logger.debug("")
        logger.debug(msg)

    def log_input_stats(self):
        """Log input statistics."""
        if self.all_input_nodes:
            logger.debug(f"Inherited clear input: shape: {self.input_shape}")
            logger.debug(f"Inherited FHE input: shape: {self.fhe_input_shape}, gap={self.input_gap}")
            logger.debug(f"Input stats (min, max): [{self.input_min:.4f}, {self.input_max:.4f}]")

    def log_output_stats(self):
        """Log output statistics."""
        logger.debug(f"Clear output: shape={self.output_shape}")
        logger.debug(f"FHE output: shape={self.fhe_output_shape}, gap={self.output_gap}")
        logger.debug(f"Output stats (min, max): [{self.output_min:.4f}, {self.output_max:.4f}]")


class OrionTracer(fx.Tracer):
    """Custom tracer with modified leaf module detection."""
    def is_leaf_module(self, m, module_path=""):
        if not isinstance(m, nn.Module):
            return False
        
        if isinstance(m, on.Module) and m.trace_internals:
            return False
       
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        
        return not any(m.children())

    def trace_model(self, model):
        model_name = model.__class__.__name__
        logger.info(f"Starting trace of model: {model_name}")

        if self.is_leaf_module(model):
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


    
class StatsTracker(fx.Interpreter):
    def __init__(self, module, temp_batch_size, user_batch_size):
        super().__init__(module)
        self.temp_batch_size = temp_batch_size
        self.user_batch_size = user_batch_size

        self.dashboard = TracerDashboard(title="Orion Nodes", user_batch_size=self.user_batch_size)
        self.dashboard.set_model(module)
        self._init_tracked_nodes()

    def _init_tracked_nodes(self):
        dash_nodes = []
        for fx_node in self.module.graph.nodes:
            module = None
            if fx_node.op == "call_module":
                module = self.module.get_submodule(fx_node.target)

            node = Node(fx_node, module)
            if module is not None:
                dash_nodes.append(node)

        self.dashboard.set_nodes(dash_nodes)

    def run_node(self, fx_node: fx.Node):
        """Execute a node and track its statistics."""
        node = fx_node.stats 
        
        # Log execution and validate
        node.log_execution()
        node.validate()

        # Map node arguments to their actual tensor values
        args = self.map_nodes_to_values(node.args, fx_node)
        kwargs = self.map_nodes_to_values(node.kwargs, fx_node)
        
        # 1. Update node's input statistics
        # 2. Run node on input
        # 3. Update node's output statistics
        if args or kwargs:
            node.update_input_stats(args, kwargs)
        
        result = super().run_node(fx_node)
        
        node.update_output_stats(result)
        self.dashboard.update(node)
        node.sync_to_module()
        
        return result

    def reset_batch_size(self):
        """Reset batch size for all Orion modules."""
        logger.debug("")
        logger.debug(f"Reverting to user batch size = {self.user_batch_size}.")
        
        updated = 0
        for fx_node in self.module.graph.nodes:
            node = fx_node.stats
            if not isinstance(node.module, on.Module):
                continue
            
            old_shape = node.module.input_shape
            node.update_module_batch_size(self.user_batch_size)
            
            logger.debug(f"{node.name}: {old_shape} → {node.module.input_shape}")
            updated += 1
            
        logger.debug(f"Updated batch size for {updated} Orion modules\n")

    def propagate(self, *args):
        self.run(*args)

    def _process_dataloader(self, dl):
        """Create a temporary DataLoader with larger batch size if needed."""
        if dl.batch_size != self.user_batch_size:
            raise ValueError(
                f"DataLoader batch size ({dl.batch_size}) must match "
                f"user_batch_size ({self.user_batch_size})"
            )

        if self.temp_batch_size > dl.batch_size:
            from torch.utils.data.sampler import RandomSampler
            shuffle = dl.sampler is None or isinstance(dl.sampler, RandomSampler)
            
            logger.debug(
                f"Temporarily increased batch size from {dl.batch_size} "
                f"to {self.temp_batch_size} for faster statistics collection"
            )
            
            return DataLoader(
                dataset=dl.dataset,
                batch_size=self.temp_batch_size,
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
            raise TypeError(
                f"Unsupported batch element type: {type(x).__name__}"
            )
        
    def propagate_all(self, input_data, device='cpu', show_progress=True):
        """Run on tensors or DataLoader(s); then refresh batch size."""
        if not isinstance(input_data, list):
            input_data = [input_data]

        all_tensors = all(isinstance(x, (torch.Tensor, list))
                        for x in input_data)
        all_loaders = all(isinstance(x, DataLoader) for x in input_data)

        self.dashboard.open()
        try:
            if all_tensors:
                types = self._format_input_types(input_data)
                logger.info(f"Propagating input data: {types}")

                inputs = []
                for x in input_data:
                    if isinstance(x, torch.Tensor):
                        inputs.append(x.to(device))
                    else:
                        inputs.append([t.to(device) for t in x])

                # one “batch” of tensors
                self.dashboard.start_run(total_batches=1)
                self.propagate(*inputs)
                self.dashboard.tick_batch()

            elif all_loaders:
                logger.info(
                    f"Propagating through {len(input_data)} DataLoader(s)"
                )
                temp_loaders = [self._process_dataloader(dl)[0]
                                for dl in input_data]

                total = None
                try:
                    total = min(len(dl) for dl in temp_loaders)
                except TypeError:
                    total = None

                self.dashboard.start_run(total_batches=total or 0)

                for batches in zip(*temp_loaders):
                    inputs = [self._extract_batch_input(b, device)
                            for b in batches]
                    self.propagate(*inputs)
                    if show_progress:
                        self.dashboard.tick_batch()
            else:
                types = self._format_input_types(input_data)
                raise ValueError(
                    "All inputs must be either Tensors or DataLoaders, not "
                    f"mixed. Got: {types}"
                )
        finally:
            self.dashboard.close()

        self.reset_batch_size()

    def _format_input_types(self, input_data):
        types = []
        for x in input_data:
            if isinstance(x, list):
                types.append(f"List[{len(x)}]")
            else:
                types.append(type(x).__name__)
        return types


