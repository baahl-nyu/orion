# orion_rich_showcase.py
# A grab-bag of Rich patterns you can reuse in Orion logging.

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

try:
    import torch
    TorchSize = torch.Size
except Exception:  # pragma: no cover
    # Fallback so the demo still runs without torch
    class TorchSize(tuple):
        def __repr__(self):  # mimic torch.Size pretty repr
            return f"torch.Size({tuple(self)})"

from rich import pretty, inspect
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.json import JSON
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree
from rich.traceback import install as install_rich_tb
from rich.columns import Columns

# ---------- Console & logging setup ----------

EMOJI_CON = Console(
    emoji=True,
    theme=Theme({
        "orion.title": "bold bright_cyan",
        "orion.dim": "dim",
        "orion.good": "bold green",
        "orion.warn": "bold yellow",
        "orion.bad": "bold red",
        "shape": "bold bright_cyan",
        "gap": "bold magenta",
        "stat": "bold yellow",
    }),
)

class ShapeHighlighter(RegexHighlighter):
    """Color specific tokens inside log messages w/o changing the message text."""
    highlights = [
        r"(?P<shape>torch\.Size\([^\)]*\))",
        r"(?P<gap>gap=\d+)",
        r"(?P<stat>\[(?:-?\d+\.\d{2,}|-?inf),\s*(?:-?\d+\.\d{2,}|inf)\])",
    ]

pretty.install(indent_guides=True)           # rich reprs for dicts/lists
install_rich_tb(show_locals=True)            # pretty tracebacks

# NBSP after emoji keeps spacing consistent in terminals
EMO_LEFT  = "⬅️\u00A0"
EMO_RIGHT = "➡️\u00A0"
EMO_OK    = "✅\u00A0"
EMO_RUN   = "🚀\u00A0"
EMO_BAR   = "📊\u00A0"
EMO_PACK  = "📦\u00A0"
EMO_STEP  = "🪜\u00A0"
EMO_RULE  = "🧭\u00A0"
EMO_BOOT  = "🧰\u00A0"

# Rich logging handler
class LevelIconFilter(logging.Filter):
    ICONS = {
        "DEBUG": "🐞",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💥",
    }
    def filter(self, record: logging.LogRecord) -> bool:
        # Prefix the level icon, keep message untouched
        prefix = self.ICONS.get(record.levelname, "")
        if prefix and not str(record.msg).startswith(prefix):
            record.msg = f"{prefix} {record.msg}"
        return True

handler = logging.Handler()
# Use Rich’s handler so we get markup, colors, and wrapping
from rich.logging import RichHandler
handler = RichHandler(
    console=EMOJI_CON,
    rich_tracebacks=True,
    markup=True,
    show_time=False,
)
handler.addFilter(LevelIconFilter())

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[handler],
)
log = logging.getLogger("orion-demo")
log.addFilter(LevelIconFilter())

# Make our console highlight shapes/gaps/stat ranges
EMOJI_CON.highlighter = ShapeHighlighter()

# ---------- Demo data models ----------

@dataclass
class NodeStat:
    name: str
    kind: str
    parent: str | None
    in_shape: TorchSize
    out_shape: TorchSize
    gap: int
    vmin: float
    vmax: float

def sample_nodes() -> List[NodeStat]:
    return [
        NodeStat("flatten", "Flatten", "act1", TorchSize([1, 5, 14, 14]), TorchSize([1, 5, 14, 14]), 2, 0.0, 8.9634),
        NodeStat("fc1", "Linear", "flatten", TorchSize([1, 5, 14, 14]), TorchSize([1, 100]), 1, -2.1076, 1.7375),
        NodeStat("bn2", "BatchNorm1d", "fc1", TorchSize([1, 100]), TorchSize([1, 100]), 1, -2.1076, 1.7375),
        NodeStat("act2", "Quad", "bn2", TorchSize([1, 100]), TorchSize([1, 100]), 1, 0.0, 4.4420),
        NodeStat("fc2", "Linear", "act2", TorchSize([1, 100]), TorchSize([1, 10]), 1, -1.1325, 0.9340),
    ]

# ---------- Individual “recipes” you can borrow ----------

def recipe_rule_sections():
    EMOJI_CON.rule(f"{EMO_RULE} [orion.title]Forward Pass[/]")
    EMOJI_CON.rule(characters="·")

def recipe_node_logs(nodes: List[NodeStat]):
    for n in nodes:
        parent_str = f" with inputs from: [orion.dim]{n.parent}[/]" if n.parent else ""
        log.debug(f"{EMO_RUN}Running {n.name} (op: call_module){parent_str}")
        log.debug(f"{EMO_LEFT}Inherited clear input: shape: {n.in_shape}")
        log.debug(f"{EMO_LEFT}Inherited FHE input: shape: {n.in_shape}, gap={n.gap}")
        log.debug(f"{EMO_BAR}Input stats (min, max): [{n.vmin:.4f}, {n.vmax:.4f}]")
        log.debug(f"➠ Clear output: shape={n.out_shape}")
        log.debug(f"➠ FHE output: shape={n.out_shape}, gap={n.gap}")
        log.debug(f"{EMO_BAR}Output stats (min, max): [{min(0,n.vmin):.4f}, {max(0,n.vmax):.4f}]")
        log.debug(f"{EMO_OK}Synced to Orion module: {n.name} (type: {n.kind})\n")

def recipe_tree(nodes: List[NodeStat]):
    tree = Tree(f"[orion.title]Model[/]", guide_style="orion.dim")
    node_map: dict[str, Tree] = {"root": tree}
    for n in nodes:
        parent_key = n.parent or "root"
        parent_tree = node_map.get(parent_key, tree)
        node_map[n.name] = parent_tree.add(f"{n.name} [orion.dim]({n.kind})[/]  {n.in_shape} → {n.out_shape}")
    EMOJI_CON.print(tree)

def recipe_table(nodes: List[NodeStat]):
    table = Table(show_header=True, header_style="bold cyan", title="Pass 1 Stats")
    for col in ("layer","type","in_shape","out_shape","gap","min","max"):
        table.add_column(col)
    for n in nodes:
        table.add_row(n.name, n.kind, repr(n.in_shape), repr(n.out_shape), str(n.gap), f"{n.vmin:.4f}", f"{n.vmax:.4f}")
    EMOJI_CON.print(Panel(table, border_style="bright_blue", title=f"{EMO_BAR} Node Summary"))

def recipe_progress(total_batches=30):
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        console=EMOJI_CON,
    ) as prog:
        t = prog.add_task("Processing batches", total=total_batches)
        for _ in range(total_batches):
            time.sleep(0.03)
            prog.advance(t)

def recipe_columns_cards():
    cards = [
        Panel("embed: hybrid\nrotations: 2\nblocks: 1×1", title="conv1", border_style="cyan"),
        Panel("rotations: 6\nblocks: 1×1", title="fc1", border_style="magenta"),
        Panel("rotations: 0\nblocks: 1×1", title="fc2", border_style="green"),
    ]
    EMOJI_CON.print(Columns(cards))

def recipe_status():
    with EMOJI_CON.status("Packing matrices…"):
        time.sleep(0.2)
    with EMOJI_CON.status("Placing bootstraps…"):
        time.sleep(0.2)

def recipe_live_dashboard(nodes: List[NodeStat], steps=40):
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=2),
        Layout(name="bottom"),
    )
    def panel_top(i: int) -> Panel:
        n = nodes[i % len(nodes)]
        text = (
            f"[b]{n.name}[/] ({n.kind})\n"
            f"in  [shape]{n.in_shape}[/]\n"
            f"out [shape]{n.out_shape}[/] • [gap]gap={n.gap}[/]\n"
            f"[stat]range=[{n.vmin:.4f}, {n.vmax:.4f}][/]"
        )
        return Panel(text, title="Current Node", border_style="cyan")
    def panel_bottom(i: int) -> Panel:
        # Fake “metrics”
        util = 20 + 60 * (0.5 + 0.5 * math.sin(i / 7.0))
        mem = 2.5 + 0.7 * (0.5 + 0.5 * math.cos(i / 11.0))
        return Panel(f"GPU Util ~ {util:5.1f}%   •   Mem {mem:0.1f} GB / 12 GB", border_style="green")
    with Live(layout, refresh_per_second=12, console=EMOJI_CON):
        for i in range(steps):
            layout["top"].update(panel_top(i))
            layout["bottom"].update(panel_bottom(i))
            time.sleep(0.05)

def recipe_pretty_json_and_inspect(nodes: List[NodeStat]):
    sample = {
        "compile": {"conv1": {"level": 5}, "fc1": {"level": 3}, "fc2": {"level": 1}},
        "rotations": {"conv1": 2, "fc1": 6, "fc2": 0},
    }
    EMOJI_CON.print(JSON.from_data(sample))
    inspect(nodes[0], methods=False)

def recipe_bars():
    def bar(x, mx, width=24):
        filled = max(0, min(width, int(width * x / mx)))
        return "█" * filled + "·" * (width - filled)
    mx = 4.44
    x = 0.93
    EMOJI_CON.print(f"{EMO_BAR} Activation range: {bar(x, mx)}  {x:.2f}/{mx:.2f}")

def recipe_packing_block():
    blocks = [
        ("conv1", (1568, 784), (2048, 8192), 2, 13),
        ("fc1" , (100, 1568), (128, 8192), 6, 128),
        ("fc2" , (10, 100)  , (8192, 8192), 0, 109),
    ]
    for name, orig, resized, rots, diags in blocks:
        EMOJI_CON.print(f"\n{EMO_PACK} Packing [b]{name}[/]:")
        EMOJI_CON.print(f"├── embed method: hybrid")
        EMOJI_CON.print(f"├── original matrix shape: {orig}")
        EMOJI_CON.print(f"├── # blocks (rows, cols) = (1, 1)")
        EMOJI_CON.print(f"├── resized matrix shape: {resized}")
        EMOJI_CON.print(f"├── # output rotations: {rots}")
        EMOJI_CON.print(f"├── time to pack (s): {random.uniform(0.04, 0.32):0.2f}")
        EMOJI_CON.print(f"├── # diagonals = {diags}")

def recipe_bootstrap_and_compile():
    EMOJI_CON.print(f"\n{EMO_BOOT} Running bootstrap placement... done! [0.002 secs.]")
    EMOJI_CON.print("├── Network requires 0 bootstrap operations.")
    EMOJI_CON.print(f"\n{EMO_STEP} Compiling network layers...")
    EMOJI_CON.print("├── conv1 @ level=5")
    EMOJI_CON.print("├── act1 @ level=4")
    EMOJI_CON.print("├── flatten @ level=3")
    EMOJI_CON.print("├── fc1 @ level=3")
    EMOJI_CON.print("├── act2 @ level=2")
    EMOJI_CON.print("├── fc2 @ level=1")

def recipe_traceback_preview():
    # Show a pretty traceback with locals (try/except so the demo continues)
    try:
        def boom(x):  # noqa: ANN001
            y = x - 3
            return 1 / (y - y)  # Division by zero on purpose
        boom(10)
    except Exception as e:  # noqa: BLE001
        log.warning("Intentionally caught error to preview Rich tracebacks: %s", e)

# ---------- Main orchestration ----------

def main():
    nodes = sample_nodes()

    recipe_rule_sections()
    recipe_node_logs(nodes)
    recipe_tree(nodes)

    recipe_table(nodes)
    recipe_columns_cards()
    recipe_progress(total_batches=24)

    recipe_status()
    recipe_packing_block()
    recipe_bootstrap_and_compile()

    EMOJI_CON.rule(f"{EMO_RULE} [orion.title]Dashboard[/]")
    recipe_live_dashboard(nodes, steps=36)

    EMOJI_CON.rule(f"{EMO_RULE} [orion.title]Introspection[/]")
    recipe_pretty_json_and_inspect(nodes)
    recipe_bars()
    recipe_traceback_preview()

    EMOJI_CON.print(Panel.fit("[orion.good]Demo complete.[/] You can now copy the parts you like!"))

if __name__ == "__main__":
    main()