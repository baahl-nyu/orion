import sys
import logging
import warnings

import torch
import torch.nn as nn
import torch.fx as fx

import orion.nn as on
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.nn.normalization import BatchNormNd

logger = logging.getLogger("orion")


class OrionTracer(fx.Tracer):
    """
    Overrides the default fx.Tracer that does not recursively access all
    modules in the network. This is a deeper trace.
    """
    def is_leaf_module(self, m, _):
        if not isinstance(m, nn.Module):
            return False
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        return not any(True for _ in m.children())
    
    def trace_model(self, model):
        # Tracing outputs are slightly different when the user provides
        # a leaf module (e.g on.Conv2d) rather than a network. We'll wrap
        # it temporarily to consistently track FHE statistics.
        if self.is_leaf_module(model, ""):
            model = ModuleWrapper(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fx.GraphModule(model, super().trace(model))
        

class ModuleWrapper(on.Module):
    """Wrapper for leaf modules to make them traceable."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        return self.module(x)


class StatsTracker(fx.Interpreter):
    """Tracks important FHE statistics. """

    def __init__(self, module: fx.GraphModule) -> None:
        super().__init__(module)
        self._init_node_attributes()

    def _init_node_attributes(self):
        # Tracks min/max values and shapes for FHE-friendly inference
        for node in self.module.graph.nodes:
            node.input_min = float("inf")
            node.input_max = float("-inf")
            node.output_min = float("inf") 
            node.output_max = float("-inf")
            node.input_shape = None
            node.output_shape = None
            node.fhe_input_shape = None
            node.fhe_output_shape = None
            node.input_gap = 1
            node.output_gap = 1
        
    def run_node(self, node: fx.Node):
        # Run one node and track its input/output stats
        self._validate_node(node)
        
        inp = self.map_nodes_to_values(node.args, node)
        if inp: 
            self.update_input_stats(inp, node)

        result = super().run_node(node)  # Forward pass the node
        self.update_output_stats(result, node)
        
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, Module):
                self.sync_module_attributes(node)

        return result
    
    def _validate_node(self, node):
        # Validate that the layer works under FHE
        self._validate_shapes_and_gaps(node)

        if node.op == "call_module":
            self._validate_module_properties(node)
    
    def _validate_shapes_and_gaps(self, node):
        # Ensure consistent shapes and gaps across inputs
        parents = node.all_input_nodes
        if not parents:
            return
            
        # Helper function to check consistency
        def check_consistency(attr_name, label):
            values = [getattr(p, attr_name) for p in parents 
                     if getattr(p, attr_name) is not None]
            if len(set(values)) > 1:
                raise ValueError(
                    f"Inconsistent {label} for {node.name}: {set(values)}"
                )
        
        # Check all required consistencies
        check_consistency('output_shape', 'input shapes')
        check_consistency('fhe_output_shape', 'FHE shapes')
        check_consistency('output_gap', 'input gaps')
    
    def _validate_module_properties(self, node):
        # Check module-specific FHE compatibility requirements
        submodule = self.module.get_submodule(node.target)
        
        # Check stride equality in pooling layers
        stride = getattr(submodule, "stride", None)
        if stride and len(set(stride)) > 1:
            raise ValueError(
                f"Stride for {node.name} must be equal in all directions: {stride}"
            )
        
        # Check BatchNorm parent count
        is_batchnorm = isinstance(submodule, BatchNormNd)
        has_multiple_parents = len(node.all_input_nodes) > 1
        
        if is_batchnorm and has_multiple_parents:
            raise ValueError(
                f"BatchNorm node {node} has multiple parents which prevents fusion"
            )
    
    def update_input_stats(self, inp: tuple, node: fx.Node):
        # Update input statistics from actual tensor values
        min_values = []
        max_values = []
        
        for e in inp:
            if isinstance(e, torch.Tensor):
                min_values.append(e.detach().min())
                max_values.append(e.detach().max())
            else: # scalars
                scalar_tensor = torch.tensor(e)
                min_values.append(scalar_tensor)
                max_values.append(scalar_tensor)

        current_min = min(min_values)
        current_max = max(max_values)
        node.input_min = min(node.input_min, current_min)
        node.input_max = max(node.input_max, current_max)
        
        # Set input shape from parent's output shape for structure preservation
        if node.all_input_nodes:
            parent = node.all_input_nodes[0]
            node.input_shape = parent.output_shape
            node.input_gap = parent.output_gap
            node.fhe_input_shape = parent.fhe_output_shape
        else:
            # For input nodes with no parents, use actual tensor shape
            node.input_shape = inp[0].shape

    def update_output_stats(self, result: torch.Tensor, node: fx.Node):
        # Update output statistics based on actual result tensor
        node.output_min = min(node.output_min, result.min())
        node.output_max = max(node.output_max, result.max())
        
        # Determine appropriate output shape based on module type
        node.output_shape = self.compute_clear_output_shape(node, result)
        node.fhe_output_shape = self.compute_fhe_output_shape(node)
        node.output_gap = self.compute_fhe_output_gap(node)

    def compute_clear_output_shape(self, node: fx.Node, result):
        # Determine output shape, preserving structure except for transforming ops
        if not node.input_shape:
            return result.shape
            
        # Only LinearTransform modules change the output shape
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return result.shape
                
        # For all other modules, preserve the input shape
        return node.input_shape

    def compute_fhe_output_gap(self, node: fx.Node):
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return module.compute_fhe_output_gap(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                )
        return node.input_gap
        
    def compute_fhe_output_shape(self, node: fx.Node):
        if not node.input_shape:
            return node.output_shape

        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return module.compute_fhe_output_shape(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                    fhe_input_shape=node.fhe_input_shape,
                    output_gap=node.output_gap,
                    clear_output_shape=node.output_shape
                )
        return node.fhe_input_shape

    def sync_module_attributes(self, node: fx.Node):
        # Sync tracked node statistics to the corresponding module
        module = self.module.get_submodule(node.target)
        module.name = node.name

        # Min/max values
        module.input_min = node.input_min
        module.input_max = node.input_max
        module.output_min = node.output_min
        module.output_max = node.output_max
        
        # Shapes
        module.input_shape = node.input_shape 
        module.output_shape = node.output_shape 
        module.fhe_input_shape = node.fhe_input_shape
        module.fhe_output_shape = node.fhe_output_shape
        
        # Multiplexed aps
        module.input_gap = node.input_gap
        module.output_gap = node.output_gap

    def update_batch_size(self, batch_size):
        for node in self.module.graph.nodes:        
            if node.op == "call_module":
                module = self.module.get_submodule(node.target)

                shape_attrs = [
                    'input_shape', 
                    'output_shape', 
                    'fhe_input_shape', 
                    'fhe_output_shape'
                ]
                
                # Update only batch dimension
                for attr in shape_attrs:
                    current_shape = getattr(module, attr)
                    new_shape = torch.Size([batch_size] + list(current_shape[1:]))
                    setattr(module, attr, new_shape)

    def propagate(self, *args) -> None:
        # Run the graph with the provided inputs
        self.run(*args)