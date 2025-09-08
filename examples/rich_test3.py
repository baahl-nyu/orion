# orion_rich_playground.py
# A tiny playground to test Rich-based logging and visuals for Orion-style workflows.

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional

# ----- optional torch-ish shape repr (keeps script runnable w/o torch) -----
try:
    import torch
    TorchSize = torch.Size
except Exception:  # pragma: no cover
    class TorchSize(tuple):
        def __repr__(self):  # mimic torch.Size pretty repr
            return f"torch.Size({tuple(self)})"

# ----- Rich setup -----
from rich import pretty, inspect
from rich.columns import Columns
from rich.console import Console, Group
from rich.highlighter import RegexHighlighter
from rich.json import JSON
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree
from rich.traceback import install as install_rich_tb
from rich.logging import RichHandler

# Console + theme
CON = Console(
    emoji=True,
    theme=Theme(
        {
            "orion.title": "bold bright_cyan",
            "orion.dim": "dim",
            "orion.good": "bold green",
            "orion.warn": "bold yellow",
            "orion.bad": "bold red",
            "shape": "bold bright_cyan",
            "gap": "bold magenta",
            "stat": "bold yellow",
        }
    ),
)

# Pretty reprs + tracebacks
pretty.install(indent_guides=True)
install_rich_tb(show_locals=True, width=None)

# NBSP after emoji keeps spacing aligned
EMO_LEFT  = "⬅️\u00A0"
EMO_RIGHT = "➡️\u00A0"
EMO_OK    = "✅\u00A0"
EMO_RUN   = "🚀\u00A0"
EMO_BAR   = "📊\u00A0"

# Highlighter for shapes/gaps/stats in plain log messages
class ShapeHighlighter(RegexHighlighter):
    highlights = [
        r"(?P<shape>torch\.Size\([^\)]*\))",
        r"(?P<gap>gap=\d+)",
        r"(?P<stat>\[(?:-?\d+\.\d+|[-+]?inf),\s*(?:-?\d+\.\d+|[-+]?inf)\])",
    ]
CON.highlighter = ShapeHighlighter()

# Logging with Rich
class LevelIconFilter(logging.Filter):
    ICONS = {
        "DEBUG": "🐞",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💥",
    }
    def filter(self, record: logging.LogRecord) -> bool:
        icon = self.ICONS.get(record.levelname, "")
        if icon and not str(record.msg).startswith(icon):
            record.msg = f"{icon} {record.msg}"
        return True

handler = RichHandler(console=CON, rich_tracebacks=True, markup=True, show_time=False)
handler.addFilter(LevelIconFilter())
logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[handler])
log = logging.getLogger("orion-demo")

# ----- demo data -----
@dataclass
class NodeStat:
    name: str
    kind: str
    parent: Optional[str]
    in_shape: TorchSize
    out_shape: TorchSize
    gap: int
    vmin: float
    vmax: float

def sample_nodes() -> List[NodeStat]:
    return [
        NodeStat("flatten", "Flatten", "act1", TorchSize([1, 5, 14, 14]), TorchSize([1, 5, 14, 14]), 2, 0.0, 8.9634),
        NodeStat("fc1",     "Linear",  "flatten", TorchSize([1, 5, 14, 14]), TorchSize([1, 100]), 1, -2.1076, 1.7375),
        NodeStat("bn2",     "BatchNorm1d","fc1",  TorchSize([1, 100]), TorchSize([1, 100]), 1, -2.1076, 1.7375),
        NodeStat("act2",    "Quad",    "bn2",     TorchSize([1, 100]), TorchSize([1, 100]), 1, 0.0, 4.4420),
        NodeStat("fc2",     "Linear",  "act2",    TorchSize([1, 100]), TorchSize([1, 10]), 1, -1.1325, 0.9340),
    ]

# ----- recipes -----

def rule_sections():
    CON.rule("🧭 [orion.title]Forward Pass[/]")
    CON.rule(characters="·")

def node_logs(nodes: List[NodeStat]):
    for n in nodes:
        parent = f" with inputs from: [orion.dim]{n.parent}[/]" if n.parent else ""
        log.debug(f"{EMO_RUN}Running {n.name} (op: call_module){parent}")
        log.debug(f"{EMO_LEFT}Inherited clear input: shape: {n.in_shape}")
        log.debug(f"{EMO_LEFT}Inherited FHE input: shape: {n.in_shape}, gap={n.gap}")
        log.debug(f"{EMO_BAR}Input stats (min, max): [{n.vmin:.4f}, {n.vmax:.4f}]")
        log.debug(f"{EMO_RIGHT}Clear output: shape={n.out_shape}")
        log.debug(f"{EMO_RIGHT}FHE output: shape={n.out_shape}, gap={n.gap}")
        log.debug(f"{EMO_BAR}Output stats (min, max): [{min(0,n.vmin):.4f}, {max(0,n.vmax):.4f}]")
        log.debug(f"{EMO_OK}Synced to Orion module: {n.name} (type: {n.kind})\n")

def fx_tree_like(nodes: List[NodeStat]):
    tree = Tree("[orion.title]Model[/]", guide_style="orion.dim")
    lookup = {"root": tree}
    for n in nodes:
        parent = lookup.get(n.parent or "root", tree)
        lookup[n.name] = parent.add(f"[cyan]{n.name}[/] [orion.dim]({n.kind})[/]  {n.in_shape} → {n.out_shape}")
    CON.print(tree)

def table_summary(nodes: List[NodeStat]):
    t = Table(show_header=True, header_style="bold cyan", title="Pass 1 Stats")
    for col in ("layer","type","in_shape","out_shape","gap","min","max"):
        t.add_column(col)
    for n in nodes:
        t.add_row(n.name, n.kind, repr(n.in_shape), repr(n.out_shape),
                  str(n.gap), f"{n.vmin:.4f}", f"{n.vmax:.4f}")
    CON.print(Panel(t, border_style="bright_blue", title=f"{EMO_BAR} Node Summary"))

def progress_demo(total_batches=24):
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        console=CON,
        transient=True,
    ) as prog:
        task = prog.add_task("Processing batches", total=total_batches)
        for _ in range(total_batches):
            time.sleep(0.03)
            prog.advance(task)

def columns_cards():
    cards = [
        Panel("embed: hybrid\nrotations: 2\nblocks: 1×1", title="conv1", border_style="cyan"),
        Panel("rotations: 6\nblocks: 1×1", title="fc1", border_style="magenta"),
        Panel("rotations: 0\nblocks: 1×1", title="fc2", border_style="green"),
    ]
    CON.print(Columns(cards))

def status_demo():
    with CON.status("Packing matrices…"):
        time.sleep(0.2)
    with CON.status("Placing bootstraps…"):
        time.sleep(0.2)

def live_watch_dashboard(nodes: List[NodeStat], steps=36):
    layout = Layout()
    layout.split_column(Layout(name="top", ratio=2), Layout(name="bottom"))

    def panel_top(i: int) -> Panel:
        n = nodes[i % len(nodes)]
        txt = (
            f"[b]{n.name}[/] ({n.kind})\n"
            f"in  [shape]{n.in_shape}[/]\n"
            f"out [shape]{n.out_shape}[/] • [gap]gap={n.gap}[/]\n"
            f"[stat]range=[{n.vmin:.4f}, {n.vmax:.4f}][/]"
        )
        return Panel(txt, title="current", border_style="cyan")

    def panel_bottom(i: int) -> Panel:
        util = 20 + 60 * (0.5 + 0.5 * math.sin(i / 7.0))
        mem  = 2.5 + 0.7 * (0.5 + 0.5 * math.cos(i / 11.0))
        return Panel(f"GPU Util ~ {util:5.1f}%   •   Mem {mem:0.1f} GB / 12 GB", border_style="green")

    with Live(layout, refresh_per_second=12, console=CON):
        for i in range(steps):
            layout["top"].update(panel_top(i))
            layout["bottom"].update(panel_bottom(i))
            time.sleep(0.05)

def pretty_json_and_inspect(nodes: List[NodeStat]):
    sample = {"compile": {"conv1": {"level": 5}, "fc1": {"level": 3}, "fc2": {"level": 1}},
              "rotations": {"conv1": 2, "fc1": 6, "fc2": 0}}
    CON.print(JSON.from_data(sample))
    inspect(nodes[0], methods=False)

def bars_demo():
    def bar(x, lo, hi, width=24):
        frac = 0 if hi == lo else (x - lo) / (hi - lo)
        k = max(0, min(width, int(frac * width)))
        return "█" * k + "·" * (width - k)
    lo, hi = 0.0, 4.44
    x = 0.93
    CON.print(f"{EMO_BAR} Activation range: {bar(x, lo, hi)}  {x:.2f}/{hi:.2f}")

def packing_block_demo():
    blocks = [
        ("conv1", (1568, 784), (2048, 8192), 2, 13),
        ("fc1" , (100, 1568), (128, 8192), 6, 128),
        ("fc2" , (10, 100)  , (8192, 8192), 0, 109),
    ]
    for name, orig, resized, rots, diags in blocks:
        CON.print(f"\n📦 Packing [b]{name}[/]:")
        CON.print(f"├── embed method: hybrid")
        CON.print(f"├── original matrix shape: {orig}")
        CON.print(f"├── # blocks (rows, cols) = (1, 1)")
        CON.print(f"├── resized matrix shape: {resized}")
        CON.print(f"├── # output rotations: {rots}")
        CON.print(f"├── time to pack (s): {random.uniform(0.04, 0.32):0.2f}")
        CON.print(f"├── # diagonals = {diags}")

def bootstrap_and_compile_demo():
    CON.print(f"\n🧰 Running bootstrap placement... done! [0.002 secs.]")
    CON.print("├── Network requires 0 bootstrap operations.")
    CON.print(f"\n🪜 Compiling network layers...")
    CON.print("├── conv1 @ level=5")
    CON.print("├── act1 @ level=4")
    CON.print("├── flatten @ level=3")
    CON.print("├── fc1 @ level=3")
    CON.print("├── act2 @ level=2")
    CON.print("├── fc2 @ level=1")

def traceback_preview():
    try:
        def boom(x):  # noqa: ANN001
            y = x - 3
            return 1 / (y - y)  # division by zero on purpose
        boom(10)
    except Exception as e:  # noqa: BLE001
        log.warning("Intentionally caught error to preview Rich tracebacks: %s", e)

# ----- run all -----

def main():
    nodes = sample_nodes()

    rule_sections()
    node_logs(nodes)
    fx_tree_like(nodes)

    table_summary(nodes)
    columns_cards()
    progress_demo(total_batches=24)

    status_demo()
    packing_block_demo()
    bootstrap_and_compile_demo()

    CON.rule("🧭 [orion.title]Dashboard[/]")
    live_watch_dashboard(nodes, steps=36)

    CON.rule("🧭 [orion.title]Introspection[/]")
    pretty_json_and_inspect(nodes)
    bars_demo()
    traceback_preview()

    CON.print(Panel.fit("[orion.good]Demo complete.[/] Copy any parts you like."))

if __name__ == "__main__":
    main()