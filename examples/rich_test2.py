# orion_rich_styles.py
# Style-focused Rich demo with presets for colors, borders, rules, progress, and logging levels.

import argparse
import logging
import math
import time
from dataclasses import dataclass
from typing import List

try:
    import torch
    TorchSize = torch.Size
except Exception:
    class TorchSize(tuple):
        def __repr__(self): return f"torch.Size({tuple(self)})"

from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.rule import Rule
from rich.text import Text
from rich import box

# --------- Presets (colors & styles) ---------

PRESETS: dict[str, dict] = {
    # Palette names use Rich color names (supports HEX like '#282a36' too).
    "dracula": {
        "theme": Theme({
            "orion.title": "bold #8be9fd",
            "orion.dim": "dim #6272a4",
            "shape": "bold #50fa7b",
            "gap": "bold #bd93f9",
            "stat": "bold #f1fa8c",
            "accent": "#ff79c6",
            "panel_border": "#bd93f9",
            # RichHandler level styles:
            "logging.level.debug": "#8be9fd",
            "logging.level.info": "#50fa7b",
            "logging.level.warning": "bold #f1fa8c",
            "logging.level.error": "bold #ff5555",
            "logging.level.critical": "reverse bold #ff5555",
        }),
        "rule_char": "━",
        "bar_style": "#bd93f9",
        "bar_complete": "#50fa7b",
        "bar_finished": "#8be9fd",
    },
    "nord": {
        "theme": Theme({
            "orion.title": "bold #88c0d0",
            "orion.dim": "dim #4c566a",
            "shape": "bold #a3be8c",
            "gap": "bold #b48ead",
            "stat": "bold #ebcb8b",
            "accent": "#81a1c1",
            "panel_border": "#5e81ac",
            "logging.level.debug": "#88c0d0",
            "logging.level.info": "#a3be8c",
            "logging.level.warning": "bold #ebcb8b",
            "logging.level.error": "bold #bf616a",
            "logging.level.critical": "reverse bold #bf616a",
        }),
        "rule_char": "─",
        "bar_style": "#5e81ac",
        "bar_complete": "#a3be8c",
        "bar_finished": "#88c0d0",
    },
    "gruvbox": {
        "theme": Theme({
            "orion.title": "bold #fabd2f",
            "orion.dim": "dim #928374",
            "shape": "bold #b8bb26",
            "gap": "bold #d3869b",
            "stat": "bold #fe8019",
            "accent": "#83a598",
            "panel_border": "#fe8019",
            "logging.level.debug": "#83a598",
            "logging.level.info": "#b8bb26",
            "logging.level.warning": "bold #fabd2f",
            "logging.level.error": "bold #fb4934",
            "logging.level.critical": "reverse bold #fb4934",
        }),
        "rule_char": "═",
        "bar_style": "#fe8019",
        "bar_complete": "#b8bb26",
        "bar_finished": "#fabd2f",
    },
    "solarized": {
        "theme": Theme({
            "orion.title": "bold #268bd2",
            "orion.dim": "dim #586e75",
            "shape": "bold #2aa198",
            "gap": "bold #6c71c4",
            "stat": "bold #b58900",
            "accent": "#859900",
            "panel_border": "#268bd2",
            "logging.level.debug": "#268bd2",
            "logging.level.info": "#2aa198",
            "logging.level.warning": "bold #b58900",
            "logging.level.error": "bold #dc322f",
            "logging.level.critical": "reverse bold #dc322f",
        }),
        "rule_char": "·",
        "bar_style": "#268bd2",
        "bar_complete": "#2aa198",
        "bar_finished": "#b58900",
    },
    "mono": {
        "theme": Theme({
            "orion.title": "bold white",
            "orion.dim": "dim",
            "shape": "bold white",
            "gap": "bold white",
            "stat": "bold white",
            "accent": "white",
            "panel_border": "white",
            "logging.level.debug": "white",
            "logging.level.info": "white",
            "logging.level.warning": "bold white",
            "logging.level.error": "bold white",
            "logging.level.critical": "reverse bold white",
        }),
        "rule_char": "─",
        "bar_style": "white",
        "bar_complete": "white",
        "bar_finished": "white",
    },
}

BOX_MAP = {
    "rounded": box.ROUNDED,
    "square": box.SQUARE,
    "heavy": box.HEAVY,
    "double": box.DOUBLE,
    "minimal": box.MINIMAL,
    "ascii": box.ASCII,
}

SPINNERS = ["dots", "line", "bouncingBar", "aesthetic", "arc", "growHorizontal"]

# ---------- Sample data ----------
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
        NodeStat("flatten","Flatten","act1",TorchSize([1,5,14,14]),TorchSize([1,5,14,14]),2,0.0,8.9634),
        NodeStat("fc1","Linear","flatten",TorchSize([1,5,14,14]),TorchSize([1,100]),1,-2.1076,1.7375),
        NodeStat("bn2","BatchNorm1d","fc1",TorchSize([1,100]),TorchSize([1,100]),1,-2.1076,1.7375),
        NodeStat("act2","Quad","bn2",TorchSize([1,100]),TorchSize([1,100]),1,0.0,4.4420),
        NodeStat("fc2","Linear","act2",TorchSize([1,100]),TorchSize([1,10]),1,-1.1325,0.9340),
    ]

# ---------- Build console & logging with preset ----------
def build_console_and_logger(preset: str):
    p = PRESETS[preset]
    console = Console(emoji=True, theme=p["theme"])
    handler = RichHandler(console=console, rich_tracebacks=True, markup=True, show_time=False)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[handler])
    log = logging.getLogger("orion-style-demo")
    return console, log, p

# ---------- Demos that respond to preset ----------
def show_rules(console: Console, p: dict):
    console.print(Rule(Text(" Orion • Forward Pass ", style="orion.title"), characters=p["rule_char"]))
    console.print(Rule(characters=p["rule_char"]))

def show_panels_table(console: Console, p: dict, nodes: List[NodeStat], box_style):
    t = Table(title="Pass 1 Stats", box=box_style, border_style=p["theme"].styles.get("panel_border",""))
    for col in ("layer","type","in_shape","out_shape","gap","min","max"):
        t.add_column(col, style="orion.dim" if col in ("type","gap") else None)
    for n in nodes:
        t.add_row(n.name, n.kind, repr(n.in_shape), repr(n.out_shape), str(n.gap), f"{n.vmin:.4f}", f"{n.vmax:.4f}")
    console.print(Panel.fit(t, border_style=p["theme"].styles.get("panel_border",""), title="📊 Summary", box=box_style))

def show_tree(console: Console, nodes: List[NodeStat]):
    tree = Tree("Model", guide_style="orion.dim")
    node_map = {"root": tree}
    for n in nodes:
        parent = node_map.get(n.parent or "root", tree)
        node_map[n.name] = parent.add(f"[orion.title]{n.name}[/] ([orion.dim]{n.kind}[/])  [shape]{n.in_shape}[/] → [shape]{n.out_shape}[/]")
    console.print(Panel(tree, title="Topology"))

def show_progress(console: Console, p: dict, spinner_name: str):
    with Progress(
        SpinnerColumn(spinner_name=spinner_name),
        TextColumn("[progress.description]{task.description}", style="accent"),
        BarColumn(bar_width=None, style=p["bar_style"], complete_style=p["bar_complete"], finished_style=p["bar_finished"]),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Processing batches", total=40)
        for _ in range(40):
            time.sleep(0.025)
            prog.advance(task)

def show_live_dashboard(console: Console, nodes: List[NodeStat], box_style):
    layout = Layout()
    layout.split_row(Layout(name="left", ratio=2), Layout(name="right"))
    def node_panel(i: int):
        n = nodes[i % len(nodes)]
        txt = Text.assemble(
            (f"{n.name}", "orion.title"), ("  ("+n.kind+")\n", "orion.dim"),
            ("in  ", "orion.dim"), (repr(n.in_shape)+"\n", "shape"),
            ("out ", "orion.dim"), (repr(n.out_shape)+"  ", "shape"),
            ("gap=", "orion.dim"), (str(n.gap), "gap"), ("\n", ""),
            ("range=", "orion.dim"), (f"[{n.vmin:.4f}, {n.vmax:.4f}]", "stat"),
        )
        return Panel(txt, title="Current Node", box=box_style)
    def metrics_panel(i: int):
        util = 20 + 60 * (0.5 + 0.5 * math.sin(i/7))
        mem = 2.5 + 0.7 * (0.5 + 0.5 * math.cos(i/11))
        t = Text(f"GPU Util ~ {util:5.1f}%   •   Mem {mem:0.1f} GB / 12 GB")
        t.stylize("accent")
        return Panel(t, title="Runtime", box=box_style)
    with Live(layout, refresh_per_second=12, console=console):
        for i in range(36):
            layout["left"].update(node_panel(i))
            layout["right"].update(metrics_panel(i))
            time.sleep(0.05)

def log_like_orion(log: logging.Logger, nodes: List[NodeStat]):
    NBSP = "\u00A0"
    EMO_RUN, EMO_LEFT, EMO_RIGHT, EMO_OK, EMO_BAR = "🚀"+NBSP, "⬅️"+NBSP, "➡️"+NBSP, "✅"+NBSP, "📊"+NBSP
    for n in nodes:
        parent = f" with inputs from: {n.parent}" if n.parent else ""
        log.debug(f"{EMO_RUN}Running {n.name} (op: call_module){parent}")
        log.debug(f"{EMO_LEFT}Inherited clear input: shape: {n.in_shape}")
        log.debug(f"{EMO_LEFT}Inherited FHE input: shape: {n.in_shape}, gap={n.gap}")
        log.debug(f"{EMO_BAR}Input stats (min, max): [{n.vmin:.4f}, {n.vmax:.4f}]")
        log.debug(f"{EMO_RIGHT}Clear output: shape={n.out_shape}")
        log.debug(f"{EMO_RIGHT}FHE output: shape={n.out_shape}, gap={n.gap}")
        log.info (f"{EMO_OK}Synced to Orion module: {n.name} (type: {n.kind})\n")

# ---------- CLI & run ----------
def main():
    parser = argparse.ArgumentParser(description="Orion Rich style presets")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="dracula")
    parser.add_argument("--box", choices=list(BOX_MAP.keys()), default="rounded")
    parser.add_argument("--spinner", choices=SPINNERS, default="aesthetic")
    args = parser.parse_args()

    console, log, preset_data = build_console_and_logger(args.preset)
    nodes = sample_nodes()
    box_style = BOX_MAP[args.box]

    # Section 1: Rules & Orion-like logs in the preset theme
    show_rules(console, preset_data)
    log_like_orion(log, nodes)

    # Section 2: Panels, tables, trees styled by preset & box choice
    show_panels_table(console, preset_data, nodes, box_style)
    show_tree(console, nodes)

    # Section 3: Progress & live dashboard honoring bar/spinner styles
    show_progress(console, preset_data, spinner_name=args.spinner)
    console.print(Rule("Live Dashboard", characters=preset_data["rule_char"]))
    show_live_dashboard(console, nodes, box_style)

    console.print(Panel.fit(f"[orion.title]Preset:[/] {args.preset}   "
                            f"[orion.title]Box:[/] {args.box}   "
                            f"[orion.title]Spinner:[/] {args.spinner}",
                            border_style=preset_data["theme"].styles.get("panel_border",""),
                            box=box_style))

if __name__ == "__main__":
    main()