import torch
import logging
import networkx as nx

from orion.nn.activation import Chebyshev
from orion.nn.linear import Linear, Conv2d
from orion.nn.normalization import BatchNorm1d, BatchNorm2d

logger = logging.getLogger("orion")

class Fuser:
    def __init__(self, network_dag: nx.DiGraph):
        self.network_dag = network_dag

    def _fuse_linear_chebyshev(self, linear, cheb):
        linear.on_weight = linear.on_weight * cheb.prescale
        linear.on_bias = linear.on_bias * cheb.prescale + cheb.constant
        
        cheb.fused = True
        cheb.depth -= 1 # The prescale no longer consumes a level

    def _fuse_bn_chebyshev(self, bn, cheb):
        if bn.affine:
            bn.on_weight = bn.on_weight * cheb.prescale
            bn.on_bias = bn.on_bias * cheb.prescale + cheb.constant
        else:
            bn.affine = True 
            bn.on_weight = torch.ones(bn.num_features) * cheb.prescale
            bn.on_bias = torch.ones(bn.num_features) * cheb.constant

        cheb.fused = True
        cheb.depth -= 1 # The prescale no longer consumes a level

    def _fuse_linear_bn(self, linear, bn):
        on_inv_running_std = 1 / torch.sqrt(bn.on_running_var + bn.eps) 
        scale = bn.on_weight * on_inv_running_std

        if len(linear.on_weight.shape) == 2: # fc layer
            linear.on_weight *= scale.reshape(-1, 1)
        else: # conv2d layer
            linear.on_weight *= scale.reshape(-1, 1, 1, 1)

        linear.on_bias = scale * (linear.on_bias - bn.running_mean) + bn.on_bias

        bn.fused = True 
        bn.depth -= (2 if bn.affine else 1)

    def fuse_two_layers(self, parent_class, child_class, fusing_function):

        def get_parent_modules(node):
            parent_modules = []
            for parent in self.network_dag.predecessors(node):
                parent_module = self.network_dag.nodes[parent]["module"]
                if isinstance(parent_module, parent_class):
                    parent_modules.append(parent_module)
            
            return parent_modules

        # We'll iterate over all nodes in our network to determine if the
        # pattern ever goes parent_class -> child_class sequentially. If 
        # so, then we'll apply `fusing_function` to those two modules.
        for node in self.network_dag.nodes:
            child_module = self.network_dag.nodes[node]["module"]
            if isinstance(child_module, child_class):
                parent_modules = get_parent_modules(node)

                if len(parent_modules) == 0:
                    logger.info(
                        f"Skipping fusion of {node} since it has no " + \
                        f"fusable parent modules.")
                elif len(parent_modules) > 1:
                    logger.info(
                        f"Skipping fusion of {node} since it has more " + \
                        f"than one fusable parent modules.")
                else: # only a single parent, index directly
                    fusing_function(parent_modules[0], child_module)

    def fuse_linear_chebyshev(self):
        self.fuse_two_layers((Linear, Conv2d), Chebyshev, 
                             self._fuse_linear_chebyshev)  

    def fuse_bn_chebyshev(self):
        self.fuse_two_layers((BatchNorm1d, BatchNorm2d), Chebyshev, 
                             self._fuse_bn_chebyshev) 

    def fuse_linear_bn(self):
        self.fuse_two_layers(Linear, BatchNorm1d, self._fuse_linear_bn)
        self.fuse_two_layers(Conv2d, BatchNorm2d, self._fuse_linear_bn)

    def fuse_modules(self):
        # Order matters here. E.g., we first fuse Chebyshev prescal into
        # BatchNorm, which can then be further fused with linear/conv layers.
        self.fuse_linear_chebyshev()
        self.fuse_bn_chebyshev()
        self.fuse_linear_bn()

            