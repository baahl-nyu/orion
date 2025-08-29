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

logger = logging.getLogger("orion")


# ----------------------------- Utilities ----------------------------- #

def iter_tensors(obj):
    """Yield all tensors found recursively in obj.
    
    Examples:
        >>> # Single tensor
        >>> t = torch.randn(3, 4)
        >>> list(iter_tensors(t))
        [tensor([[...]])]
        
        >>> # Dictionary with nested tensors
        >>> obj = {'weights': torch.randn(2, 3), 
        ...        'biases': [torch.randn(3), torch.randn(3)]}
        >>> tensors = list(iter_tensors(obj))
        >>> len(tensors)
        3
    """
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
    """Return a shape-structured mirror of obj.
    
    Examples:
        >>> # List of tensors becomes list of shapes
        >>> tensors = [torch.randn(2, 3), torch.randn(2, 3)]
        >>> shape_tree(tensors)
        [torch.Size([2, 3]), torch.Size([2, 3])]
        
        >>> # Dictionary structure is preserved
        >>> obj = {'input': torch.randn(8, 16), 
        ...        'hidden': [torch.randn(8, 32), torch.randn(8, 32)]}
        >>> shape_tree(obj)
        {'input': torch.Size([8, 16]), 
         'hidden': [torch.Size([8, 32]), torch.Size([8, 32])]}
    """
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
    """Return (min, max) across tensors; inf/-inf if none.
    
    Examples:
        >>> # Multiple tensors with different ranges
        >>> t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t2 = torch.tensor([[-1.0, 5.0]])
        >>> t3 = torch.empty(0)
        >>> tensors = [t1, t2, t3]
        >>> tensors_min_max(tensors)
        (-1.0, 5.0)
    """
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
    """Statistics tracker for a single node in the computation graph.
    
    Wraps an fx.Node to track input/output statistics, shapes, and gaps
    during graph execution. 
    
    Args:
        fx_node: The fx.Node to track statistics for.
    
    Attributes:
        input_min, input_max: Min/max values seen at node inputs.
        output_min, output_max: Min/max values seen at node outputs.
        input_shape, output_shape: Tensor shapes in cleartext.
        fhe_input_shape, fhe_output_shape: (may differ due to packing).
        input_gap, output_gap: Multiplexed packing gap values.
    """
    def __init__(self, fx_node: fx.Node):
        self.fx_node = fx_node
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
    
    # ---- Logging methods ----
    def log_execution(self):
        """Log the node being executed and its potential parents."""
        parents = [p.name for p in self.all_input_nodes]
        msg = f"\n→ Running {self.name} (op: {self.op})"
        if parents:
            msg += " with inputs from: " + ", ".join(parents)

        logger.debug(msg)

    def log_input_stats(self):
        """Log input statistics."""
        logger.debug(f"  → Input: shape={self.input_shape}")
        logger.debug(f"  → FHE shape: {self.fhe_input_shape}, gap={self.input_gap}")
        logger.debug(f"  📊 Input stats: [{self.input_min:.4f}, {self.input_max:.4f}]")

    def log_output_stats(self):
        """Log output statistics."""
        logger.debug(f"  → Output: shape={self.output_shape}")
        logger.debug(f"  → FHE shape: {self.fhe_output_shape}, gap={self.output_gap}")
        logger.debug(f"  📊 Output stats: [{self.output_min:.4f}, {self.output_max:.4f}]")

    def log_shape_inheritance(self):
        """Log shape inheritance from parents."""
        if self.all_input_nodes:
            logger.debug(f"  ← Inherited shape: {self.input_shape}")
            logger.debug(f"  ← FHE shape: {self.fhe_input_shape}, gap={self.input_gap}")


class OrionTracer(fx.Tracer):
    """Custom tracer with modified leaf module detection.
    
    Treats non-container modules without children as leaves, providing finer
    granularity than PyTorch's default tracer. This ensures atomic operations
    (Linear, Conv2d, custom Orion layers) appear as single nodes in the graph
    while compound modules are decomposed.
    
    Methods:
        is_leaf_module: Determines if a module should be traced into.
        trace_model: Traces a model into an fx.GraphModule.
    """
    def is_leaf_module(self, m, module_path=""):
        # Non-modules (functions, methods) become call_function nodes
        if not isinstance(m, nn.Module):
            return False
        
        # User has modified the default on.Module behavior to no longer 
        # treat their module as a black box. Can now trace within this
        # module's forward pass, optimizing bootstrap placement there too. 
        if isinstance(m, Module) and m.trace_internals:
            return False  # Trace inside
       
        # Containers must be traced into to see their contents
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        
        # Modules with children are traced into; modules without children are leaves
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
    """Executes a traced graph to collect FHE statistics.
    
    Propagates sample data through the graph to record min/max values, ,
    tensor shapes, and some FHE-specific metadata (gaps) at each node. 
    Updates all Orion modules with these statistics for compilation.
    
    Args:
        module: The traced GraphModule to analyze.
        fit_batch_size: Batch size to use during statistics collection.
        user_batch_size: Target batch size used in packing (from config).
    """
    def __init__(self, module, temp_batch_size, user_batch_size):
        super().__init__(module)
        self.temp_batch_size = temp_batch_size
        self.user_batch_size = user_batch_size
        self._init_tracked_nodes()

    def _init_tracked_nodes(self):
        """Create Node wrappers for all fx nodes."""
        for fx_node in self.module.graph.nodes:
            Node(fx_node) 

    def _get_module(self, node):
        """Get the module for a call_module node, None otherwise."""
        if node.op != "call_module":
            return None
        return self.module.get_submodule(node.target)

    def run_node(self, fx_node: fx.Node):
        """Execute a node and track its statistics."""
        node = fx_node.stats 
        
        # Log execution and validate node works in Orion
        node.log_execution()
        self._validate_node(node)

        # Map node arguments to their actual tensor values
        args = self.map_nodes_to_values(node.args, fx_node)
        kwargs = self.map_nodes_to_values(node.kwargs, fx_node)
        
        # Update input statistics if we have actual data
        if args or kwargs:
            self._update_input_stats(node, args, kwargs)
        
        # Execute the actual node operation and track output statistics 
        # from the result
        result = super().run_node(fx_node)
        self._update_output_stats(node, result)
        
        # Sync statistics to Orion modules for later use
        mod = self._get_module(node)
        if isinstance(mod, Module):
            self._sync_module_attributes(node, mod)
        
        return result

    def _validate_node(self, node):
        """Validate FHE constraints: parent gaps, stride, BN parent count."""
        parents = node.all_input_nodes

        # 1) All parent gaps must match
        if parents:
            gaps = {p.stats.output_gap for p in parents}
            if len(gaps) != 1:
                self._raise_validation_error(
                    f"Inconsistent input gaps for {node.name}: {gaps}"
                )

        # 2) Call function level checks
        if parents and node.op == "call_function":
            if len(parents) > 1:  
                common_shape = parents[0].stats.fhe_output_shape  
                
                # Check all other parents FHE output shapes match
                for p in parents[1:]:
                    if p.stats.fhe_output_shape != common_shape:
                        all_shapes = [p.stats.fhe_output_shape for p in parents]
                        self._raise_validation_error(
                            f"Inconsistent FHE input shapes for {node.name}: {all_shapes}" 
                    )
            
        # 3) Module-level checks (only for call_module nodes)
        mod = self._get_module(node)
        if not mod:
            return

        # stride must be uniform (e.g., (s, s, ...))
        stride = getattr(mod, "stride", None)
        if stride is not None:
            vals = (set(stride) if isinstance(stride, (tuple, list)) else {stride})
            if len(vals) != 1:
                self._raise_validation_error(
                    "Stride for {0} must be equal in all directions: {1}"
                    .format(node.name, stride)
                )

        # BatchNorm cannot have multiple parents (prevents fusion)
        if isinstance(mod, BatchNormNd) and len(parents) > 1:
            self._raise_validation_error(
                f"BatchNorm node {node.name} has multiple parents which "
                "prevents fusion"
            )

    def _raise_validation_error(self, msg):
        """Log and raise a validation error."""
        logger.error(f"  ✗ {msg}")
        raise ValueError(msg)

    def _update_input_stats(self, node, args, kwargs):
        """Update node's input statistics."""
        tensors = list(iter_tensors((args, kwargs)))
        mn, mx = tensors_min_max(tensors)
        node.input_min = min(node.input_min, mn)
        node.input_max = max(node.input_max, mx)

        # Skip generating shapes if already done once 
        if node.input_shape is not None:
            return

        parents = node.all_input_nodes
        if parents:
            if len(parents) == 1:
                p = parents[0].stats
                node.input_shape = p.output_shape
                node.input_gap = p.output_gap
                node.fhe_input_shape = p.fhe_output_shape
            else:
                node.input_shape = [p.stats.output_shape for p in parents]
                node.input_gap = parents[0].stats.output_gap
                node.fhe_input_shape = [p.stats.fhe_output_shape for p in parents]
        else: # inputs to model won't have parents
            node.input_shape = shape_tree(args)
            node.fhe_input_shape = node.input_shape
        
        node.log_shape_inheritance()
        node.log_input_stats()

    def _update_output_stats(self, node, result):
        """Update node's output statistics."""
        tensors = list(iter_tensors(result))
        mn, mx = tensors_min_max(tensors)
        node.output_min = min(node.output_min, mn)
        node.output_max = max(node.output_max, mx)

        result_shapes = shape_tree(result)
        node.output_shape = self._compute_clear_output_shape(node, result_shapes)
        node.fhe_output_shape = self._compute_fhe_output_shape(node)
        node.output_gap = self._compute_output_gap(node)

        self._handle_getitem_fhe_shape(node)
        node.log_output_stats()

    def _handle_getitem_fhe_shape(self, node):
        """
        getitem: if parent FHE shape is a list, select the correct index
        to be extracted (multi-input).
        """
        if (node.op == "call_function"
                and getattr(node.target, "__name__", "") == "getitem"
                and node.all_input_nodes
                and len(node.args) > 1
                and isinstance(node.args[1], int)):
            idx = node.args[1]
            parent_stats = node.all_input_nodes[0].stats
            shapes = parent_stats.fhe_output_shape
            if isinstance(shapes, list) and 0 <= idx < len(shapes):
                node.fhe_output_shape = shapes[idx]
                logger.debug(
                    "  → getitem extracted FHE shape[%d]: %s",
                    idx, node.fhe_output_shape,
                )

    def _compute_clear_output_shape(self, node, result_shapes):
        """
        Compute cleartext output shape.
        
        If an Orion module has the attribute "preserve_input_shape = True"
        then just pass the input shape through. Use case (on.Flatten):
        Conv -> Flatten -> FC. To correctly pack Conv weights, we must
        know the prior linear transformation was a Conv.
        """
        mod = self._get_module(node)
        if isinstance(mod, Module) and mod.preserve_input_shape:
            return node.input_shape
        return result_shapes
        
    def _compute_output_gap(self, node):
        """Compute output gap."""
        mod = self._get_module(node)
        if isinstance(mod, LinearTransform):
            new_gap = mod.compute_fhe_output_gap(
                input_gap=node.input_gap,
                input_shape=node.input_shape,
                output_shape=node.output_shape,
            )
            return new_gap         
        return node.input_gap

    def _compute_fhe_output_shape(self, node):
        """Compute FHE output shape."""
        if not node.input_shape: # input to model
            return node.output_shape

        mod = self._get_module(node)
        if isinstance(mod, Module):
            fhe_output_shape = mod.compute_fhe_output_shape(
                input_gap=node.input_gap,
                input_shape=node.input_shape,
                output_shape=node.output_shape,
                fhe_input_shape=node.fhe_input_shape,
                output_gap=node.output_gap,
                clear_output_shape=node.output_shape,
            )
            return fhe_output_shape
        
        # Call functions like torch.add or operator.add should modify this
        if node.op == "call_function" and isinstance(node.fhe_input_shape, list):
            return node.fhe_input_shape[0]

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
        
        logger.debug(f"  ✓ Synced to Orion module: {node.name} (type: {type(module).__name__})")

    def reset_batch_size(self):
        """Reset batch size for all Orion modules."""
        logger.info(f"\nUpdating batch size to {self.user_batch_size} for all Orion modules...")
        
        updated = 0
        for fx_node in self.module.graph.nodes:  # Changed to fx_node
            mod = self._get_module(fx_node.stats)  # Pass the Node wrapper
            if not isinstance(mod, Module):
                continue
            
            old_shape = mod.input_shape
            self._update_module_shapes(mod)
            
            logger.debug(f"  {fx_node.name}: {old_shape} → {mod.input_shape}")
            updated += 1
            
        logger.info(f"✓ Updated batch size for {updated} Orion modules\n")

    def _update_module_shapes(self, module):
        """Update all shape attributes of a module with new batch size."""
        shape_attrs = [
            'input_shape', 
            'output_shape', 
            'fhe_input_shape', 
            'fhe_output_shape'
        ]
        for attr in shape_attrs:
            shape = getattr(module, attr)
            setattr(module, attr, self._update_shape_batch_size(shape))

    def _update_shape_batch_size(self, shape):
        """Update batch dimension in a shape structure."""
        if shape is None:
            return None
        if isinstance(shape, torch.Size):
            return torch.Size([self.user_batch_size] + list(shape[1:]))
        if isinstance(shape, (list, tuple)):
            updated = [self._update_shape_batch_size(s) for s in shape]
            return tuple(updated) if isinstance(shape, tuple) else updated
        return shape

    def propagate(self, *args):
        self.run(*args)

    def _process_dataloader(self, dl):
        """Create a temporary DataLoader with larger batch size if needed."""
       # Check that original batch size matches expected user batch size
        if dl.batch_size != self.user_batch_size:
            raise ValueError(
                f"DataLoader batch size ({dl.batch_size}) must match "
                f"user_batch_size ({self.user_batch_size})"
            )

        if self.temp_batch_size > dl.batch_size:
            from torch.utils.data.sampler import RandomSampler
            shuffle = dl.sampler is None or isinstance(dl.sampler, RandomSampler)
            
            logger.info(
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
            raise TypeError(f"Unsupported batch element type: {type(x).__name__}")
        
    def propagate_all(self, input_data, device='cpu', show_progress=True):
        """Run on tensors or DataLoader(s); then refresh batch size."""
        if not isinstance(input_data, list):
            input_data = [input_data]

        all_tensors = all(isinstance(x, (torch.Tensor, list)) for x in input_data)
        all_loaders = all(isinstance(x, DataLoader) for x in input_data)

        if all_tensors:
            types = self._format_input_types(input_data)
            logger.info(f"Propagating input data: {types}")
            
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
            temp_loaders = [self._process_dataloader(dl)[0] for dl in input_data]
            
            # Iterate through batches
            iterator = zip(*temp_loaders)
            if show_progress:
                try:
                    total = min(len(dl) for dl in temp_loaders)
                except TypeError:
                    total = None
                iterator = tqdm(
                    iterator, desc="Processing batches", unit="batch",
                    total=total, leave=True
                    )
            
            for batches in iterator:
                inputs = [self._extract_batch_input(b, device) for b in batches]
                self.propagate(*inputs)

        else:
            types = self._format_input_types(input_data)
            raise ValueError(
                f"All inputs must be either Tensors or DataLoaders, not mixed. Got: {types}"
            )
        
        self.reset_batch_size()

    def _format_input_types(self, input_data):
        types = []
        for x in input_data:
            if isinstance(x, list):
                types.append(f"List[{len(x)}]")
            else:
                types.append(type(x).__name__)
        return types