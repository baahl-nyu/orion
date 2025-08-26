import math 
 
import networkx as nx 
import torch.nn as nn
import matplotlib.pyplot as plt

from orion.nn.linear import LinearTransform

class LevelDAG(nx.DiGraph):
    """
    The level digraph implementation from Section 5.2 of Orion:
    https://arxiv.org/pdf/2311.03470. It your goal is to understand this
    code, it may be useful to have the paper beside you for reference.
    The goal of this class is to determine the levels of all layers
    and locations of bootstrap operations in a given neural network.
    """

    def __init__(self, l_eff, network_dag, path=None):
        super().__init__()
        self.l_eff = l_eff
        self.network_dag = network_dag
        self.path = path
        self.network_dag
        self.build_level_dag_from_path()

    def __add__(self, other):
        """
        Accepts two level DAGs and computes their aggregate level DAG
        by "adding" the two together. This addition is a bit abstract;
        see the paper for additional details.
        """

        if self.number_of_nodes() == 0:
            return other 
        elif other.number_of_nodes() == 0:
            return self

        fork, join = self.topo_path[0], self.topo_path[-1]
        if fork == join:
            raise ValueError(
                "Level DAGs can only be added if they represent paths in a " 
                "residual subgraph. Use append() if the desired behavior "
                "is to concatenate subgraphs."
            )
        
        # Get all start/end nodes that are shared between both DAGs
        source_nodes = [f"{fork}@l={i}" for i in range(self.l_eff+1)]
        target_nodes = [f"{join}@l={i}" for i in range(self.l_eff+1)]

        # Instantiate a new instance that black-boxes the intermediate
        # nodes of both level DAGs we're adding
        aggregate_level_dag = LevelDAG(self.l_eff, self.network_dag, path=None)
        aggregate_level_dag.add_nodes_from(
            (node, {"weight": 0}) for node in source_nodes)
        aggregate_level_dag.add_nodes_from(
            (node, {"weight": 0}) for node in target_nodes)   

        # Iterate over all pairs of start/end nodes in the level DAG 
        # and assign an edge weight between each pair equal to the sum
        # of the shortest paths between the pairs in each individual
        # level DAG.
        for source in source_nodes:
            for target in target_nodes:
                path1, length1 = self.shortest_path(source, target)
                path2, length2 = other.shortest_path(source, target)

                # We have to be careful here, because each path "could"
                # include a previously solved aggregate level digraph.
                # If so, then it will contain a "path" variable along the
                # edge that captures the data we black-boxed above. To not
                # lose this information, we'll track all nodes traversed
                # which also conveniently include the best levels as well.
                nodes_traversed = set()
                if length1 != float("inf") and length2 != float("inf"):
                    for path, dag in zip([path1, path2], [self, other]):
                        for u, v in zip(path[:-1], path[1:]): 
                            if "path" in dag.get_edge_data(u, v):
                                nodes_traversed.update(dag[u][v]["path"])
                            else:
                                nodes_traversed.update([u, v])
                
                total = length1 + length2
                aggregate_level_dag.add_edge(
                    source, target, weight=total, path=nodes_traversed)

        return aggregate_level_dag
    
    def append(self, other):
        """Append the LevelDAG "other" to the end of the LevelDAG "self"."""

        if self.number_of_nodes() == 0:
            self.add_nodes_from(other.nodes(data=True))
            self.add_edges_from(other.edges(data=True))
            return 
        elif other.number_of_nodes() == 0:
            return
        
        # We'll be connecting the head nodes of "other" to the tail nodes of
        # "self". Get these now before composing the graphs for correctness
        self_tail_nodes = self.tail()
        other_head_nodes = other.head()
    
        # Compose graphs directly into self
        self.add_nodes_from(other.nodes(data=True))
        self.add_edges_from(other.edges(data=True))
        
        for tail in self_tail_nodes:
            for head in other_head_nodes:
                weight, _ = self.estimate_bootstrap_latency(tail, head)
                self.add_edge(tail, head, weight=weight, path=[tail, head])
        
    def build_level_dag_from_path(self):
        """If a path parameter is provided, this method automatically 
           generates a level digraph for this path. If this path contains
           a previously solved SESE region, then we insert it before
           moving on."""

        if not self.path:
            return
        
        self.topo_path = list(nx.topological_sort(self.path))
        solved_dags = self.network_dag.solved_residual_level_dags

        prev_dag_nodes = None
        visited_nodes = set()
        for curr_path_node in self.topo_path:
            if curr_path_node not in visited_nodes:
                if curr_path_node in solved_dags.keys():
                    # Then there already exists a solved LevelDAG for this SESE
                    # region. We'll append it into the LevelDAG we're building.
                    solved_level_dag = solved_dags[curr_path_node]
                    self.append(solved_level_dag)

                    # Then update our list of visited nodes to skip over the
                    # nodes that were blackboxed by this region. 
                    tail = self.network_dag.residuals[curr_path_node]
                    start_idx = self.topo_path.index(curr_path_node)
                    end_idx = self.topo_path.index(tail)   
                    
                    for idx in range(start_idx, end_idx + 1):
                        visited_nodes.add(self.topo_path[idx])
                    
                    prev_dag_nodes = solved_level_dag.tail()
                else:
                    # Otherwise, just build the next layer and connect it.
                    curr_dag_nodes = self.build_layer(curr_path_node)
                    self.connect_layer_to_existing_dag(curr_dag_nodes, prev_dag_nodes)
                    prev_dag_nodes = curr_dag_nodes

    def build_layer(self, node: str):
        """Builds the next layer of nodes in the level DAG and estimates
           their latency, eventually used in shortest path."""

        level_dag_nodes = [f"{node}@l={i}" for i in range(self.l_eff+1)]
        self.add_nodes_from(level_dag_nodes)

        node_module = self.network_dag.nodes[node]["module"]
        for level, name in enumerate(level_dag_nodes):
            weight = self.estimate_layer_latency(node_module, level)
            self.nodes[name]["weight"] = weight

        return level_dag_nodes

    def estimate_layer_latency(self, module, level):
        """
        Analytical model for estimating linear layer latency. A more
        comprehensive profiler could provide more accurate estimates.
        One is in the works, however we feel this route is overkill.
        What really matters (e.g. 95% of inference) is the latencies 
        of bootstrapping. Even setting to zero every linear layer 
        latency here will not affect bootstrap counts.
        """

        if isinstance(module, nn.Identity):
            return 0
        elif module and not hasattr(module, "depth"):
            raise ValueError(
                f"The multiplicative depth of the Orion module {module} "
                f"cannot be automatically determined. Ensure it has a " 
                f"depth attribute.") 
        elif not module or not hasattr(module, "depth"):
            return 0
        elif module.level and module.level != level: # user-specified level
            return float("inf") # always use user-specified level
        elif level < module.depth:
            return float("inf")
        elif isinstance(module, LinearTransform):
            # Iterate over blocks, extract number of diagonals and estimate
            # linear transform latency analytically based on params.
            alpha = 0.001
            runtime = 0
            for diags in module.diagonals.values(): # iterate over blocks
                runtime += (alpha * len(diags) * level)
            return runtime
        else:
            return 0

    def connect_layer_to_existing_dag(self, curr_dag_nodes, prev_dag_nodes):
        """
        Connect the layer built in build_layer() to the existing level DAG
        through edges weighted by if a bootstrap is required. 
        """

        if not prev_dag_nodes:
            return
                
        for curr_node in curr_dag_nodes:
            for prev_node in prev_dag_nodes:
                weight, _ = self.estimate_bootstrap_latency(prev_node, curr_node)
                self.add_edge(
                    prev_node, curr_node, weight=weight, path=[curr_node, prev_node])
        
    def estimate_bootstrap_latency(self, prev_node: str, curr_node: str):
        """Estimate bootstrap latency between nodes in the network."""
        
        # Extract node information
        prev_path_node = prev_node.split("@")[0]
        prev_module = self.network_dag.nodes[prev_path_node]["module"]
        prev_level = int(prev_node.split("=")[-1])
        curr_level = int(curr_node.split("=")[-1])
        
        # Case 1: Previous module is None or Identity
        if prev_module is None or isinstance(prev_module, nn.Identity):
            if prev_level >= curr_level:
                return (0, 0)
            return (float("inf"), 0)
                
        # Case 2: Ensure module has depth information
        if not hasattr(prev_module, "depth"):
            raise ValueError(
                f"The multiplicative depth of the Orion module {prev_module} "
                "cannot be automatically determined. Ensure it has a depth attribute."
            )
        
        # Case 3: Bootstrap required
        if curr_level > prev_level - prev_module.depth:
            if prev_level - prev_module.depth <= 0:
                return (float("inf"), 0)
            
            # Analytical fit based on experiments. Once again could benefit
            # from a profiler, but the search space here is quite massive.
            a, b, c = 3.41, 0.18, 4.81
            t_boot = a * math.exp(b * self.l_eff) + c
            num_boots_required = self.get_num_output_cts(prev_module)
            
            return (t_boot * num_boots_required, num_boots_required)

        # Case 4: No bootstrap required
        return (0, 0)

    def get_num_output_cts(self, module):
        num_slots = module.scheme.params.get_slots()
        
        # This node may have multiple outgoing edges
        if isinstance(module.fhe_output_shape, list):
            num_elements = [shape.numel() for shape in module.fhe_output_shape]
            return sum([math.ceil(e / num_slots) for e in num_elements])
        
        num_elements = module.fhe_output_shape.numel()
        return math.ceil(num_elements / num_slots)

    def shortest_path(self, source, target):
        """Relaxation stage of topological sort."""

        # Initialize distances and predecessors
        distances = {node: float("inf") for node in self.nodes}
        distances[source] = self.nodes[source]["weight"]
        predecessors = {node: None for node in self.nodes}

        for node in nx.topological_sort(self):
            if distances[node] == float("inf"):
                continue 
            for neighbor in self.neighbors(node):
                edge_weight = self[node][neighbor]["weight"]
                neighbor_node_weight = self.nodes[neighbor]["weight"]
                dist = distances[node] + edge_weight + neighbor_node_weight

                if dist < distances[neighbor]:
                    distances[neighbor] = dist
                    predecessors[neighbor] = node
        
        # Reconstruct shortest path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        return path, distances[target]
    
    def head(self):
        head_nodes = []
        for node in self.nodes:
            if self.in_degree(node) == 0:
                head_nodes.append(node)
        return head_nodes
    
    def tail(self):
        tail_nodes = []
        for node in self.nodes:
            if self.out_degree(node) == 0:
                tail_nodes.append(node)
        return tail_nodes

    def plot(self, save_path="", figsize=(10,10)):
        """Plot the level digraph with edge colors based on 'weight'."""
        try:
            pos = nx.nx_agraph.graphviz_layout(self, prog='dot')
        except:
            print("Graphviz not installed. Defaulting to worse visualization.")
            pos = nx.kamada_kawai_layout(self)

        # Extract edge weights and map to specific colors
        edge_weights = nx.get_edge_attributes(self, 'weight')
        edge_colors = [
            "red" if weight == float("inf") else
            "yellow" if weight > 0 else
            "green"
            for weight in edge_weights.values()
        ]

        # Plot the graph with edge colors
        plt.figure(figsize=figsize)
        nx.draw(
            self,
            pos,
            with_labels=True,
            arrows=True,
            edge_color=edge_colors
        )
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()