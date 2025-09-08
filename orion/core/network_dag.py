import torch
import torch.fx as fx
import networkx as nx
import matplotlib.pyplot as plt
import operator

from orion.nn.normalization import BatchNormNd


class NetworkDAG(nx.DiGraph):
    """
    Represents a neural network as a directed acyclic graph (DAG) using 
    NetworkX. This class builds a DAG from a PyTorch network, identifies
    residual connections, and provides several useful methods that we will 
    use in our automatic bootstrap placement algorithm. 
    """   
    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.residuals = {}  # Maps fork nodes to their corresponding join nodes
    
    def build_dag(self):
        """Extract computational graph from PyTorch model using torch.fx"""
        
        # Only keep nn.Modules, on.Modules, and basic arithmetic operations
        # in the final network graph we pass onward.
        keep_ops = {operator.add, operator.sub, operator.mul, 
                    torch.add, torch.sub, torch.mul}
        
        nodes_to_keep = set()
        for node in self.trace.graph.nodes:
            if node.op in ['call_module', 'placeholder']:
                nodes_to_keep.add(node.name)
            elif node.op == 'call_function' and node.target in keep_ops:
                nodes_to_keep.add(node.name)
        
        # Build parent relationships including nested args
        parent_map = {}
        for node in self.trace.graph.nodes:
            parent_map[node.name] = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    parent_map[node.name].append(arg.name)
                elif isinstance(arg, (tuple, list)):
                    for item in arg:
                        if isinstance(item, fx.Node):
                            parent_map[node.name].append(item.name)
        
        # Add nodes and edges, bridging over filtered nodes
        for node in self.trace.graph.nodes:
            if node.name not in nodes_to_keep:
                continue
            
            label = f"{node.name}\n({node.op})"
            
            # Get actual module reference if this is a call_module
            module = None
            if node.op == "call_module":
                module = self.trace.get_submodule(str(node.target))
            
            self.add_node(node.name, op=node.op, module=module, label=label)
            
            def get_sources(node_name):
                # Recursively find kept nodes by traversing filtered parents
                if node_name in nodes_to_keep:
                    return [node_name]
                sources = []
                for parent in parent_map.get(node_name, []):
                    sources.extend(get_sources(parent))
                return sources
            
            for parent in parent_map[node.name]:
                for source in get_sources(parent):
                    if source != node.name:
                        self.add_edge(source, node.name)

        # Verify network works with automatic bootstrap placement
        self._validate_single_output()
    
    def _validate_single_output(self):
        """
        Ensure the network has single entry and exit points by adding 
        placeholder nodes if necessary. This is required for automatic 
        bootstrap placement.
        """
        input_nodes = [n for n in self.nodes if self.in_degree(n) == 0]
        output_nodes = [n for n in self.nodes if self.out_degree(n) == 0]

        # Add a common placeholder parent for all input nodes
        if len(input_nodes) > 1:
            placeholder_input = "__placeholder_input__"
            self.add_node(placeholder_input, op="placeholder", 
                        module=None, label="placeholder_input")
            for input_node in input_nodes:
                self.add_edge(placeholder_input, input_node)
        elif len(input_nodes) == 0:
            # Handle edge case of empty graph
            raise ValueError("Graph has no input nodes")

        # Add a common placeholder child for all output nodes
        if len(output_nodes) > 1:
            placeholder_output = "__placeholder_output__"
            self.add_node(placeholder_output, op="placeholder", 
                        module=None, label="placeholder_output")
            for output_node in output_nodes:
                self.add_edge(output_node, placeholder_output)
        elif len(output_nodes) == 0:
            # Handle edge case of empty graph
            raise ValueError("Graph has no output nodes")
        
    def find_residuals(self):
        """Finds pairs of fork/join nodes representing residual connections. 
        We consider a fork (join) node to be any Orion module or arithmetic 
        operation in our computational graph that has two or more children 
        (parents). Each residual connection creates a pair of fork/join nodes 
        that become the start/end nodes of each subgraph that we will
        ultimately extract in our automatic bootstrap placement algorithm."""

        # Residual connections in FHE are particularly difficult to deal with. 
        # Each residual connection creates a pair of fork and join nodes in our 
        # graph. For every fork, there is a join somewhere later in the graph. 
        # Our automatic bootstrap placement algorithm relies on extracting the 
        # subgraphs between pairs of fork/join nodes. This function nicely finds 
        # fork/join pairs and stores them in the self.residuals dictionary so 
        # we can reference them later.
        changed = True
        while changed:
            changed = False

            # Deterministic topo order: by generations, then by name in each gen
            layers = list(nx.topological_generations(self))
            order = [n for layer in layers for n in sorted(layer)]

            for node in order:
                # Skip synthetic helpers
                if self.nodes[node].get('op') in ['fork', 'join']:
                    continue

                successors = list(self.successors(node))
                predecessors = list(self.predecessors(node))

                # Divergence -> try to bracket a residual
                if len(successors) >= 2:
                    convergence = self._find_convergence(node, successors)
                    if convergence:
                        self._process_residual(node, convergence)
                        changed = True
                        break
    
    def _find_convergence(self, node, successors):
        """Find where diverging paths reconverge (residual pattern)"""
        descendants = {s: {s} | nx.descendants(self, s) for s in successors}
        topo_order = list(nx.topological_sort(self))
        start_idx = topo_order.index(node)
        
        # Search forward in topological order
        for candidate in topo_order[start_idx + 1:]:
            if sum(candidate in desc for desc in descendants.values()) >= 2:
                return candidate
        return None
    
    def _process_residual(self, node, convergence):
        """Create fork/join pair for residual connection"""
        successors = list(self.successors(node))
        fork = self._unique_name(f"{node}_fork")
        join = self._unique_name(f"{convergence}_join")
        
        self.add_node(fork, op="fork", module=None, label=f"{fork}\n(fork)")
        self.add_node(join, op="join", module=None, label=f"{join}\n(join)")
        
        # Identify which branches actually converge
        converging = []
        for s in successors:
            if s == convergence or nx.has_path(self, s, convergence):
                converging.append(s)
        
        for s in successors:
            self.remove_edge(node, s)
        
        self.add_edge(node, fork)
        
        # Fork handles 2 branches (prefer ones that converge)
        branches_for_fork = converging[:2] if len(converging) >= 2 else successors[:2]
        for branch in branches_for_fork:
            self.add_edge(fork, branch)
        
        # Non-forked branches stay connected to original node
        remaining = [s for s in successors if s not in branches_for_fork]
        for branch in remaining:
            self.add_edge(node, branch)
        
        # Insert join before convergence point
        fork_reach = {fork} | nx.descendants(self, fork)
        for pred in list(self.predecessors(convergence)):
            if pred in fork_reach:
                self.remove_edge(pred, convergence)
                self.add_edge(pred, join)
        self.add_edge(join, convergence)
        
        self.residuals[fork] = join
    
    def _unique_name(self, base):
        """Generate unique node name with counter suffix if needed"""
        if base not in self.nodes:
            return base
        counter = 1
        while f"{base}_{counter}" in self.nodes:
            counter += 1
        return f"{base}_{counter}"
    
    def extract_residual_subgraph(self, fork):
        """Extract subgraph between a fork and its corresponding join"""
        if fork not in self.residuals:
            raise ValueError(f"{fork} is not a fork node")
        
        join = self.residuals[fork]
        
        # Collect all nodes/edges in paths from fork to join
        nodes_in_residual = set()
        edges_in_residual = set()
        
        for path in nx.all_simple_paths(self, fork, join):
            nodes_in_residual.update(path)
            edges_in_residual.update(zip(path[:-1], path[1:]))
        
        # Build subgraph
        subgraph = nx.DiGraph()
        for node in nodes_in_residual:
            subgraph.add_node(node, **self.nodes[node]) # Preserve node attributes
        subgraph.add_edges_from(edges_in_residual)
        
        return subgraph
    
    def remove_fused_batchnorms(self):
        """Removes BatchNorm nodes from the graph when it is known that they
        can be fused with preceding linear layers."""

        for node in list(self.nodes):
            node_module = self.nodes[node]["module"]
            
            if isinstance(node_module, BatchNormNd) and node_module.fused:
                # Get the parents and children of the batchnorm node
                parent_nodes = list(self.predecessors(node))
                child_nodes = list(self.successors(node))

                # Our tracer has already verified that the BN node only has
                # one parent.
                parent = parent_nodes[0]

                # Our fuser will have fused this BN node if it was possible,
                # and it's fused attribute will have been set. Remove this
                # BN node so that it isn't counted when we assign levels to
                # layers further into the compilation process.
                if node_module.fused:
                    for child in child_nodes:
                        self.add_edge(parent, child)
                    self.remove_node(node)
    
    def topological_sort(self):
        return nx.topological_sort(self)
    
    def plot(self, figsize=(15, 7.5), save_path=None):
        """Visualize the DAG with color-coded node types"""
        color_map = {
            'placeholder': 'lightgreen',
            'call_module': 'lightblue',
            'call_function': 'lightyellow',
            'fork': 'salmon',
            'join': 'salmon',
            'merge': 'orange'
        }
        
        node_colors = [color_map.get(self.nodes[node]['op'], 'white') 
                        for node in self.nodes()]
        labels = nx.get_node_attributes(self, 'label')
        
        try:
            pos = nx.nx_agraph.graphviz_layout(self, prog='dot')
        except:
            pos = nx.spring_layout(self, k=2, iterations=50)
        
        plt.figure(figsize=figsize)
        nx.draw(self, pos, labels=labels, node_color=node_colors,
                node_size=1000, font_size=8, arrows=True, 
                edge_color='gray', arrowsize=15)
                
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()