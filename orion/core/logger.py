import logging
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.theme import Theme
from rich.tree import Tree
import torch

# -------------------- logger setup --------------------
THEME = Theme({
    "header": "bold cyan",
    "primary": "#94e2d5",
    "accent": "#a6e3a1",
    "muted": "grey54",
    "label": "white",
    "value": "#f9e2af",
    "good": "#b4befe",
    "warn": "yellow",
    "bad": "red",
    "frame": "grey70",
})

logger = logging.getLogger("orion")
logger.setLevel(logging.WARNING)
logger.propagate = False

if not any(isinstance(h, RichHandler) for h in logger.handlers):
    handler = RichHandler(
        console=Console(emoji=True, theme=THEME),
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
        omit_repeated_times=False
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def set_log_level(level):
    logger.setLevel(level)


class TracerDashboard:
    def __init__(self, title="Model", refresh_per_second=10, user_batch_size=None):
        self.console = Console(emoji=True, theme=THEME)
        self.title = title
        self.refresh_per_second = int(refresh_per_second)
        self.user_batch_size = user_batch_size

        # logging.INFO will print dashboard, DEBUG let's tracer handle logging.
        self.quiet = logging.getLogger("orion").getEffectiveLevel() <= logging.DEBUG
        
        self.live = None
        self.model = None
        self.nodes = {}  
        
        self.total_batches = 0
        self.done_batches = 0
    
    def open(self):
        if self.quiet or self.live:
            return
        self.live = Live(
            self._build_tree(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
        )
        self.live.start()
    
    def close(self):
        if self.quiet or not self.live:
            return
        self.live.update(self._build_tree())
        self.live.stop()
        self.live = None
    
    def set_nodes(self, nodes):
        """Store nodes for live updates"""
        self.nodes = {n.name: n for n in nodes}
    
    def set_model(self, model):
        """Store the model for tree building"""
        self.model = model
    
    def start_run(self, total_batches):
        self.total_batches = int(total_batches or 0)
        self.done_batches = 0
        if self.quiet or not self.live:
            return
        self.live.update(self._build_tree())

    def tick_batch(self):
        self.done_batches += 1
        if self.quiet or not self.live:
            return
        self.live.update(self._build_tree())

    def update(self, node):
        if node.name in self.nodes:
            self.nodes[node.name] = node
            if self.quiet or not self.live:
                return
            self.live.update(self._build_tree())
    
    def _build_tree(self):
        """Build tree with live data"""
        # Header with progress if running
        if self.total_batches:
            pct = min(1.0, self.done_batches / self.total_batches)
            bar_width = 20
            filled = int(pct * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            title = f"[header]{self.title}[/] {bar} {int(pct*100)}%"
        else:
            title = f"[header]{self.title}[/]"
        
        root = Tree(title)
        
        if self.model:
            self._add_modules_with_data(root, self.model)
        
        # Wrap in a Panel to ensure it displays properly
        return Panel(root, box=box.ROUNDED, border_style="frame", padding=(1, 2))
    
    def _add_modules_with_data(self, tree_node, module, prefix=""):
        """Add modules to tree with live data"""
        for name, child in module.named_children():
            # Build hierarchical path with dots (for display)
            full_name = f"{prefix}.{name}" if prefix else name
            class_name = child.__class__.__name__
            
            # Base label - just module name and class
            label = f"[label]{name}[/] [white]({class_name})[/]"
            
            # Add to tree first
            child_node = tree_node.add(label)
            
            # Convert dots to underscores for lookup
            lookup_name = full_name.replace('.', '_')
            
            # Add data as sub-items if available
            if lookup_name in self.nodes:
                node = self.nodes[lookup_name]
                
                # Format shapes
                in_shape = self._format_shape(node.input_shape)
                out_shape = self._format_shape(node.output_shape)
                
                fhe_in = self._format_shape(node.fhe_input_shape)
                fhe_out = self._format_shape(node.fhe_output_shape)

                in_gap = node.input_gap 
                out_gap = node.output_gap

                in_min = node.input_min 
                in_max = node.input_max 

                out_min = node.output_min 
                out_max = node.output_max
                
                # Add data as indented sub-lines
                child_node.add(f"[muted]input (min/max):  [/] [good]({in_min:.3f}[/], [good]{in_max:.3f}[/])")
                child_node.add(f"[muted]output (min/max): [/] [primary]({out_min:.3f}[/], [primary]{out_max:.3f})[/]")
                child_node.add(f"[muted]clear shapes:     [/] [value]{in_shape}[/] → [value]{out_shape}[/]")
                child_node.add(f"[muted]fhe shapes:       [/] [accent]{fhe_in}[/] → [accent]{fhe_out}[/]")
                child_node.add(f"[muted]fhe gap:          [/] [primary]{in_gap}[/] → [primary]{out_gap}[/]")
            
            # Recurse for children (keep using dots for path building)
            self._add_modules_with_data(child_node, child, full_name)
    
    def _format_shape(self, shape):
        """Format shape for display"""
        if shape is None:
            return "-"
        
        # Apply batch size replacement
        shape = self._replace_batch_size(shape)
        
        if isinstance(shape, (tuple, list, torch.Size)):
            return str(tuple(shape))
        return str(shape)
    
    def _replace_batch_size(self, obj):
        """Replace first dimension with user batch size"""
        if obj is None or self.user_batch_size is None:
            return obj
        
        if isinstance(obj, torch.Size):
            shape_list = list(obj)
            if shape_list:
                shape_list[0] = self.user_batch_size
            return tuple(shape_list)
        
        if isinstance(obj, (list, tuple)):
            items = [self._replace_batch_size(shape) for shape in obj]
            return tuple(items) if isinstance(obj, tuple) else items
        
        return obj
    
    def print_model_tree(self, model, title="Model", max_depth=None):
        """Static model tree print (no live data)"""
        root = Tree(f"[primary]{title}[/]")
        
        def add_modules(tree_node, module, depth=0):
            if max_depth and depth >= max_depth:
                return
            
            for name, child in module.named_children():
                class_name = child.__class__.__name__
                label = f"[label]{name}[/] ([muted]{class_name}[/])"
                
                child_node = tree_node.add(label)
                add_modules(child_node, child, depth + 1)
        
        add_modules(root, model)
        print()
        self.console.print(root)
        print()