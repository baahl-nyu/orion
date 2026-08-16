import os
import math
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import matplotlib.pyplot as plt

# tqdm not used; Rich progress is used for packing visualization

# Rich debug UI
import logging
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress, BarColumn
from rich.live import Live
from rich.columns import Columns
from .logger import THEME, logger as ORION_LOGGER
from typing import Optional

# Module-level console consistent with project theme
_CONSOLE = Console(emoji=True, theme=THEME)

# Global live layout to render multiple cards side-by-side
_PACKING_LIVE: Optional[Live] = None
_PACKING_CARDS: list = []

def _update_live():
    global _PACKING_LIVE
    if _PACKING_LIVE is None:
        _PACKING_LIVE = Live(
            Columns(_PACKING_CARDS, equal=False, expand=False, padding=(0, 0)),
            console=_CONSOLE,
            refresh_per_second=12,
            transient=False,
        )
        _PACKING_LIVE.start()
    else:
        _PACKING_LIVE.update(
            Columns(_PACKING_CARDS, equal=False, expand=False, padding=(0, 0))
        )

# --------------------- Rich helpers --------------------- #
def _fmt(value):
    """Format a value for display, using '-' when missing."""
    if value is None:
        return "-"
    try:
        # torch.Size -> tuple string for readability
        if isinstance(value, torch.Size):
            return str(tuple(value))
        return str(value)
    except Exception:
        return "-"


def _compute_bar_width(embed_method, original_shape, num_blocks, block_height,
                       resized_shape, output_rotations, elapsed, total_diagonals):
    """Estimate a good bar width to match the table width.

    We compute width ≈ max_label + 1 + max_value to align with the 2-column grid.
    This keeps the progress bar visually the same width as the card content.
    """
    labels = [
        "embed method:",
        "original shape:",
        "# blocks (r,c):",
        "block height:",
        "resized shape:",
        "output rotations:",
        "time to pack (s):",
        "# diagonals:",
    ]
    values = [
        _fmt(embed_method),
        _fmt(original_shape),
        _fmt(num_blocks),
        _fmt(block_height),
        _fmt(resized_shape),
        _fmt(output_rotations),
        _fmt(elapsed),
        _fmt(total_diagonals),
    ]
    label_max = max(len(s) for s in labels)
    value_max = max(len(str(v)) for v in values)
    width = label_max + 1 + value_max
    # Clamp to reasonable min/max to avoid giant or tiny bars
    return max(18, min(64, width))


def _packing_panel(
    *,
    title: str,
    embed_method=None,
    original_shape=None,
    num_blocks=None,
    block_height=None,
    resized_shape=None,
    output_rotations=None,
    elapsed=None,
    total_diagonals=None,
    progress_renderable=None,
):
    """Build a Rich panel summarizing packing details for a single layer."""
    table = Table.grid(padding=(0, 0))
    table.expand = False
    table.add_row("[muted]embed method:[/]", f"[value]{_fmt(embed_method)}[/]")
    table.add_row("[muted]original shape:[/]", f"[value]{_fmt(original_shape)}[/]")
    table.add_row("[muted]# blocks (r,c):[/]", f"[value]{_fmt(num_blocks)}[/]")
    table.add_row("[muted]block height:[/]", f"[value]{_fmt(block_height)}[/]")
    table.add_row("[muted]resized shape:[/]", f"[value]{_fmt(resized_shape)}[/]")
    table.add_row("[muted]output rotations:[/]", f"[value]{_fmt(output_rotations)}[/]")
    table.add_row("[muted]time to pack (s):[/]", f"[value]{_fmt(elapsed)}[/]")
    table.add_row("[muted]# diagonals:[/]", f"[value]{_fmt(total_diagonals)}[/]")

    content = Group(progress_renderable, table) if progress_renderable is not None else table

    return Panel.fit(
        content,
        title=f"[primary]{title}[/]",
        border_style="frame",
        box=box.ROUNDED,
        padding=(0, 1),
    )

#-------------------#
#   Packing Logic   #
#-------------------#

def shape_numel(shape):
    """Total element count of a layer shape attribute (fhe_input_shape /
    fhe_output_shape), tolerating the occasional list-of-shapes case (a
    node with several distinct outgoing shapes) the same way
    core/level_dag.py::get_num_output_cts does."""
    if shape is None:
        return 0
    if isinstance(shape, (list, tuple)) and shape and not isinstance(shape[0], int):
        return max((s.numel() for s in shape), default=0)
    return shape.numel()

def round_up_slots(elements, max_slots):
    """Next power-of-two slot count that fits `elements`, floored at 256
    (Cheddar's EvalSpecialFFT hard minimum) and capped at the scheme's
    max slots."""
    if elements <= 0:
        return 0
    val = 1 << math.ceil(math.log2(elements))
    val = max(256, val)
    if max_slots is not None:
        val = min(max_slots, val)
    return val

def propagate_physical_slots(network_dag, topo_sort):
    """Compute each layer's true *physical* ciphertext slot width by
    walking the network in topological order, instead of letting every
    layer assume the scheme's fixed global max.

    Why this exists (cheddar-only concern): pack_conv2d/pack_linear used
    to always pack against `scheme.params.get_slots()`, so every Cheddar
    ciphertext ended up encoded at full width regardless of how few
    elements were meaningful -- confirmed to silently defeat Cheddar's
    own sparse-bootstrap circuit (see test_bootstrap_sparse[cheddar]).
    This is gated to the cheddar backend only: lattigo/desilo's rotation
    primitives were never verified to be safe with a narrower-than-global
    packing width (unlike Cheddar's HRot, which was directly confirmed to
    wrap modulo the ciphertext's own tracked width, not the full ring --
    see Context.cpp::HRot in the local cheddar clone). For every other
    backend this function is a pure no-op: every module gets `.slots`/
    `.physical_slots` set to the scheme's global max, identical to the
    pre-existing behavior, so lattigo/desilo are unaffected byte-for-byte.

    Naively computing each layer's width independently from its own
    shape is unsafe: the "hybrid" packing method's output-rotation fold
    (see LinearTransform.evaluate_transforms) only *logically* compresses
    a layer's output -- it never narrows the ciphertext's physical width
    -- so a downstream layer computing its own width in isolation could
    build diagonals for a different physical width than the ciphertext it
    actually receives, causing silently wrong rotation wraparound. This
    pass instead threads the *actual* physical width forward: a node's
    input width is the max over its predecessors' physical output width
    (this is what reconciles residual-add branches that could otherwise
    diverge), and non-LinearTransform ops just relay it unchanged. Width
    can only stay the same or grow here -- there is no resize primitive
    (yet) to narrow it back down mid-network.

    Sets `module.physical_slots` on every module in the DAG (so
    auto_bootstrap.py's get_bootstrap_slots can reuse the same
    authoritative value instead of recomputing its own), and additionally
    `module.slots` on every LinearTransform (read directly by
    pack_conv2d/pack_linear/evaluate_transforms).

    Returns the physical width the network's own input must be encoded
    at, i.e. whatever the first LinearTransform layer's packing requires.
    """
    max_slots = None
    backend = None
    embed_method = None
    output_width = {}
    network_input_slots = None

    # ORION_CHEDDAR_DISABLE_FOLD_RETAG=1: A/B escape hatch for the
    # fold-retag narrowing below -- restores the old monotone
    # width-only-grows behavior (pack AND runtime read the same flag,
    # so compile-time widths and runtime retags always stay consistent
    # within a process).
    fold_retag_off = bool(os.environ.get(
        "ORION_CHEDDAR_DISABLE_FOLD_RETAG", ""))

    # The last LinearTransform in topo order is packed with the square
    # method (orion.py forces last=True: no output fold), so it never
    # narrows -- mirror that exclusion here.
    last_lt_node = None
    for _n in reversed(topo_sort):
        _m = network_dag.nodes[_n]["module"]
        if _m is not None and hasattr(_m, "generate_diagonals"):
            last_lt_node = _n
            break

    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        preds = list(network_dag.predecessors(node))
        pred_width = max((output_width[p] for p in preds), default=0)

        if module is None:
            output_width[node] = pred_width
            continue

        if max_slots is None:
            max_slots = module.scheme.params.get_slots()
            backend = module.scheme.params.get_backend()
            embed_method = module.scheme.params.get_embedding_method()

        # ORION_CHEDDAR_DISABLE_NARROW_PACKING=1: diagnostic escape hatch
        # -- route cheddar through the same no-op path as every other
        # backend, restoring the pre-narrow-packing global-width
        # behavior (every ciphertext packed/booted at max_slots). Used
        # to A/B the HELRM threshold-500 precision regression.
        narrow_disabled = bool(os.environ.get(
            "ORION_CHEDDAR_DISABLE_NARROW_PACKING", ""))
        if backend != "cheddar" or narrow_disabled:
            # No-op path for every other backend: deliberately do NOT
            # touch .slots/.physical_slots here, leaving them at their
            # class-level None default. Every consumer already falls
            # back correctly when they're None -- pack_conv2d/
            # pack_linear/evaluate_transforms via `self.slots or
            # get_slots()`, auto_bootstrap.get_bootstrap_slots via an
            # explicit `is not None` check. Setting them to max_slots
            # here would look equivalent for packing (those consumers
            # already default to the global max) but would silently
            # override nn.operations.Bootstrap.compile()'s own
            # `self.physical_slots or <narrower elements-based calc>`
            # fallback -- permanently forcing full-width bootstraps for
            # lattigo/desilo instead of the narrower sparse bootstrap
            # they already correctly compute on their own from a
            # layer's logical shape. output_width still needs a value
            # for this loop's own bookkeeping, just not exposed here.
            output_width[node] = max_slots
            if network_input_slots is None and hasattr(module, "generate_diagonals"):
                network_input_slots = max_slots
            continue

        own_input_elements = shape_numel(getattr(module, "fhe_input_shape", None))
        input_width = pred_width
        if own_input_elements:
            input_width = max(input_width, round_up_slots(own_input_elements, max_slots))
        if input_width == 0:
            input_width = max_slots

        is_linear_transform = hasattr(module, "generate_diagonals")
        if is_linear_transform:
            own_output_elements = shape_numel(module.fhe_output_shape)
            width = max(input_width, round_up_slots(own_output_elements, max_slots))
            width = min(max_slots, width)
            module.slots = width
            if network_input_slots is None:
                # diagonalize() uses this same single `width` scalar as
                # the block edge length for BOTH matrix dimensions (see
                # packing.diagonalize's num_block_rows/num_block_cols),
                # so the input ciphertext this layer consumes must
                # already be encoded at `width`, not the narrower
                # pre-output-max `input_width` -- that mismatch silently
                # produced wrong rotation wraparound (caught by
                # test_cheddar_narrow_packing.py's MAE assertion).
                network_input_slots = width

            # Fold-retag width narrowing: the hybrid method's output
            # fold (LinearTransform.evaluate_transforms's rotate-and-add
            # over slots/2^i) leaves the output genuinely periodic with
            # period slots >> output_rotations ~= pow2ceil(out_elements)
            # -- verified numerically on HELRM@500k (sparse_to_top's
            # 32768-wide output matches itself mod 512 to the noise
            # floor, ~2.6e-7). A periodic dense ciphertext IS a valid
            # sparse ciphertext, so evaluate_transforms retags it down
            # to `folded_slots` (pure metadata via SetCiphertextSlots)
            # and everything downstream -- including bootstrap circuit
            # selection -- runs at the narrow width. This is the resize
            # primitive the module docstring said didn't exist (yet).
            # Restricted to plain Linear (Embedding/Conv2d have their
            # own fold/layout conventions), non-last (the last LT uses
            # the square method, no fold), hybrid embedding method only.
            # Runtime cross-checks the prediction against the actual
            # output_rotations and refuses to retag on mismatch.
            module.folded_slots = None
            out_width = width
            if (not fold_retag_off
                    and type(module).__name__ == "Linear"
                    and node != last_lt_node
                    and embed_method == "hybrid"):
                folded = round_up_slots(own_output_elements, max_slots)
                # Floor at 512, not round_up_slots' 256: a 256-wide
                # min-KS bootstrap on the word32 chain hits cheddar's
                # "Hoist: min_ks should be false for bs == 1" abort (its
                # stride decomposition degenerates to a single-rotation
                # hoist group). 512-wide boots are verified fine on both
                # word sizes, and boot cost is width-flat, so the extra
                # width is free.
                folded = max(folded, 512)
                if folded and folded < width:
                    module.folded_slots = folded
                    out_width = folded
        else:
            width = input_width
            out_width = width

        module.physical_slots = width
        output_width[node] = out_width

    # Per-input required widths. network_input_slots (the first LT's
    # width in topo order) is a single scalar and silently WRONG for
    # multi-input networks whose inputs feed differently-narrow paths:
    # HELRM's dense input feeds bottom_0 (width 512) while its sparse
    # input feeds the compression LT (width 2048) -- encoding the sparse
    # input against the dense path's 512 made Scheme.encode fall through
    # to the legacy full-width path (1,067 elements > 512), and a
    # 32768-wide ciphertext entering diagonals built mod 2048 loses
    # every wraparound contribution (the deterministic thr500 precision
    # regression: 5.98 vs 12.78 bits). For each DAG source, walk forward
    # to the first LinearTransform(s) it reaches and require that width
    # (max over branches). Scheme.encode then targets the smallest
    # per-input width that fits the tensor being encoded.
    # (Non-cheddar / narrow-disabled runs never set .slots, so the walk
    # below finds nothing and falls back to {max_slots} -- same single
    # global width as before.)
    input_widths = set()
    sources = [n for n in topo_sort
               if not list(network_dag.predecessors(n))]
    for s in sources:
        seen = set()
        frontier = [s]
        found = set()
        while frontier:
            nxt = []
            for n in frontier:
                if n in seen:
                    continue
                seen.add(n)
                mod = network_dag.nodes[n]["module"]
                if mod is not None and hasattr(mod, "generate_diagonals") \
                        and getattr(mod, "slots", None):
                    found.add(mod.slots)
                    continue  # stop at the first LT on this branch
                nxt.extend(network_dag.successors(n))
            frontier = nxt
        if os.environ.get("ORION_DEBUG_INPUT_WIDTHS"):
            print(f"[input-widths] source={s} found={sorted(found)}")
        # Orion's network_dag merges every network input into ONE
        # placeholder node, so a source may legitimately fan out to
        # several entry LTs of different widths (HELRM: dense->512,
        # sparse->2048) and per-source attribution is impossible here.
        # Expose every entry width; Scheme.encode disambiguates by
        # element count (smallest listed width that fits the tensor).
        input_widths.update(found)
    if not input_widths:
        input_widths = {network_input_slots or max_slots}

    return sorted(input_widths)

def pack_conv2d(conv_layer: nn.Module, last: bool):
    slots = conv_layer.slots or conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    weight = conv_layer.on_weight
    if conv_layer.groups > 1:
        weight = resolve_grouped_conv(conv_layer)

    toeplitz = construct_conv2d_toeplitz(conv_layer, weight)
    diagonals, output_rotations, slice_meta = diagonalize(
        toeplitz,
        slots,
        embed_method,
        last,
        debug=conv_layer.scheme.params.get_debug_status(),
        layer_name=getattr(conv_layer, "name", "Conv2d"),
    )

    return diagonals, output_rotations, slice_meta

def construct_conv2d_toeplitz(conv_layer, weight):
    N, on_Ci, on_Hi, on_Wi = conv_layer.fhe_input_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    Ho, Wo = conv_layer.output_shape[2:]
   
    P = conv_layer.padding[0]
    D = conv_layer.dilation[0]
    iG = conv_layer.input_gap 
    oG = conv_layer.output_gap
    kH, kW = weight.shape[2:]

    def compute_first_kernel_position():
        mpx_anchors = valid_image_indices[:, :iG, :iG].reshape(-1, 1)

        row_idxs = torch.arange(0, kH*D*iG, D*iG).reshape(-1, 1)
        col_idxs = torch.arange(0, kW*D*iG, D*iG)
        kernel_offsets = valid_image_indices[0, row_idxs, col_idxs].flatten()
        
        img_pixels_touched = mpx_anchors + kernel_offsets
        return img_pixels_touched.flatten()
    
    def compute_row_interchange_map():
        output_indices = torch.arange(on_Ho * on_Wo).reshape(on_Ho, on_Wo)
        
        start_indices = output_indices[:oG, :oG].flatten()
        corner_indices = output_indices[0:(Ho*oG):oG, 0:(Wo*oG):oG].reshape(-1, 1)
        return corner_indices + start_indices
    
    # Padded input dimensions with multiplexing
    Hi_pad = on_Hi + 2*P*iG 
    Wi_pad = on_Wi + 2*P*iG

    # Initialize our sparse Toeplitz matrix
    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * Hi_pad * Wi_pad
    toeplitz = sp.lil_matrix((n_rows, n_cols), dtype="f")

    # Create an index grid for the padded input image.
    valid_image_indices = torch.arange(n_cols).reshape(on_Ci, Hi_pad, Wi_pad)

    # Pad the kernel's input and output channels to the nearest multiple
    # of gap^2 to ensure that multiplexing works.
    kernel = torch.zeros(on_Co * oG**2, on_Ci * iG**2, kW, kH) 
    kernel[:weight.shape[0], :weight.shape[1], ...] = weight

    # All the indices the kernel initially touches
    initial_kernel_position = compute_first_kernel_position()

    # Create our row-interchange map that dictates how we permute rows for 
    # optimal packing. Also return all indices that the first top-left filter 
    # value touches throughout the convolution.
    row_map = compute_row_interchange_map()
    corner_indices = valid_image_indices[0, 0:(Ho*oG):oG, 0:(Wo*oG):oG].flatten() 

    # Create offsets for the multiplexed output channels.
    out_channels = (torch.arange(on_Co) * (on_Ho * on_Wo)).reshape(on_Co, 1)

    # Flattened kernel populates rows of our Toeplitz matrix
    kernel_flat = kernel.reshape(kernel.shape[0], -1)

    # Iterate over all positions that the top-left kernel element can touch 
    # populating the correct (permuted) rows of our Toeplitz matrix.
    for i, start_idx in enumerate(corner_indices):
        rows = (row_map[i] + out_channels).reshape(-1, 1)
        cols = initial_kernel_position + start_idx
        toeplitz[rows, cols] = kernel_flat

    # Keep only the columns corresponding to the non-padded input image.
    row_idxs = torch.arange(P*iG, P*iG + on_Hi).reshape(-1, 1)
    col_idxs = torch.arange(P*iG, P*iG + on_Wi)
    image_indices = valid_image_indices[:, row_idxs, col_idxs].flatten()
    toeplitz = toeplitz.tocsc()[:, image_indices]
    
    # Support batching
    toeplitz = sp.kron(sp.eye(N, dtype="f"), toeplitz, format="csr")
    return toeplitz
    
def construct_conv2d_bias(conv_layer):
    N, Co, Ho, Wo = conv_layer.output_shape 
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]

    bias = conv_layer.on_bias
    bias = bias.repeat_interleave(Ho*Wo)
    bias = bias.reshape(1, Co, Ho, Wo)
    bias_multiplexed = multiplex(bias, conv_layer.output_gap).squeeze(0)
    
    mC, mH, mW = bias_multiplexed.shape
    bias_vector = torch.zeros(on_Co, on_Ho, on_Wo)
    bias_vector[:mC, :mH, :mW] = bias_multiplexed
    bias_vector = bias_vector.flatten().repeat(N)

    return bias_vector

def pack_linear(linear_layer: nn.Module, last: bool):
    slots = linear_layer.slots or linear_layer.scheme.params.get_slots()
    embed_method = linear_layer.scheme.params.get_embedding_method()

    weight = construct_linear_matrix(linear_layer)
    diagonals, output_rotations, slice_meta = diagonalize(
        weight,
        slots,
        embed_method,
        last,
        debug=linear_layer.scheme.params.get_debug_status(),
        layer_name=getattr(linear_layer, "name", "Linear"),
    )
    return diagonals, output_rotations, slice_meta

def construct_linear_matrix(linear_layer):
    if len(linear_layer.input_shape) == 2:
        N = linear_layer.input_shape[0]
        matrix = linear_layer.on_weight 
    else: # Prior layer was not a linear layer
        out_features = linear_layer.out_features
        input_gap = linear_layer.input_gap 
        N, Ci, Hi, Wi = linear_layer.input_shape 
        on_Ci, on_Hi, on_Wi = linear_layer.fhe_input_shape[1:]
        
        reshaped = linear_layer.on_weight.reshape(out_features, Ci, Hi, Wi)
        reshaped = multiplex(reshaped, input_gap)

        matrix = torch.zeros(out_features, on_Ci, on_Hi, on_Wi)
        matrix[..., :Hi*input_gap, :Wi*input_gap] = reshaped 
        matrix = matrix.reshape(out_features, -1)
   
    matrix = torch.kron(torch.eye(N), matrix) 
    matrix_sparse = sp.csr_matrix(matrix.cpu().numpy())
    return matrix_sparse

def construct_linear_bias(linear_layer):
    N = linear_layer.input_shape[0]
    return linear_layer.on_bias.repeat(N)

#-----------------------------#
#       Helper Functions      #
#-----------------------------#

def multiplex(matrix, gap):
    N, Ci, Hi, Wi = matrix.shape
    Co = math.ceil(Ci / (gap**2))
    
    # Pad the tensor to have channels divisible by gap^2
    padded = torch.zeros(N, Co * gap**2, Hi, Wi)
    padded[:, :Ci, ...] = matrix
    return F.pixel_shuffle(padded, gap) # multiplexed

def resolve_grouped_conv(conv_layer):
    on_weight = conv_layer.on_weight.repeat(1, conv_layer.groups, 1, 1)

    # Zero out input channels to support arbitrary groups
    mask = torch.zeros_like(on_weight)        
    Ci_per_group = conv_layer.in_channels // conv_layer.groups
    Co_per_group = conv_layer.out_channels // conv_layer.groups
    
    for i in range(conv_layer.groups):
        mask[i*Co_per_group:(i+1)*Co_per_group, 
             i*Ci_per_group:(i+1)*Ci_per_group, ...] = 1

    return on_weight * mask

def _slice_heights(matrix, num_slots, num_block_cols, block_height):
    """Per-column-block ``{block_col: (row_offset, height, out_rotations)}``.

    A column block only needs to be as tall as the rows actually nonzero
    inside it. Blocks whose live rows already span the layer's padded
    height are left out, so a fully dense layer returns ``{}`` and
    nothing changes.

    Two constraints fix the window:

    * ``height`` is a power of two, because the fold halves the
      replication at each step.
    * ``offset`` is a MULTIPLE of ``height``. After the fold, slot k
      carries matrix row ``offset + (k mod height)``, so slot k is in its
      correct place exactly when ``offset`` is a multiple of ``height``
      and ``k`` falls in ``[offset, offset+height)``. Align the window
      and the right copy already sits at the right slots -- the only
      thing left to do is mask away the other copies, with no placement
      rotation. Grow the height until the aligned window covers the live
      rows.
    """
    csc = matrix.tocsc()
    meta = {}
    for bc in range(num_block_cols):
        sub = csc[:, bc * num_slots:(bc + 1) * num_slots]
        rows = sub.nonzero()[0]
        if rows.size == 0:
            # An all-zero block contributes nothing; diagonalize() emits
            # its usual single zero diagonal on the uniform path.
            continue
        lo, hi = int(rows.min()), int(rows.max())
        height = 1 << max(0, (hi - lo).bit_length())
        while height < block_height and (lo // height) * height + height <= hi:
            height <<= 1
        if height >= block_height:
            continue
        meta[bc] = ((lo // height) * height, height,
                    int(math.log2(num_slots // height)))

    # Shrinking costs one multiplicative level: each window's repeats
    # have to be masked away before the windows are summed. That is only
    # worth paying when the plaintexts saved are substantial. Measured on
    # HE-LRM at threshold 500, where the sparse layer is a single slice
    # spanning every output row and so is untouched, the only layer that
    # shrinks is a 16->432 placement whose 496 saved plaintexts bought a
    # level and 2 bits of precision -- a clear loss. Returning {} keeps
    # the layer on the uniform path at depth 1.
    saved = sum(block_height - h for _, h, _ in meta.values())
    floor = int(os.environ.get("ORION_SLICE_MIN_SAVING", "4096"))
    if os.environ.get("ORION_SLICE_DEBUG", "") == "1":
        spans = []
        for bc in range(min(num_block_cols, 6)):
            sub = csc[:, bc * num_slots:(bc + 1) * num_slots]
            r = sub.nonzero()[0]
            spans.append("empty" if r.size == 0
                         else f"{int(r.min())}-{int(r.max())}")
        print(f"[slice-heights] nnz={matrix.nnz} bh={block_height} "
              f"shrunk={len(meta)}/{num_block_cols} saved={saved} floor={floor} "
              f"first_spans={spans}", flush=True)
    if saved < floor:
        return {}
    return meta


def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    debug: bool = False,
    layer_name: Optional[str] = None,
):
    """
    For each (slots, slots) block of the input matrix, this function 
    extracts the generalized diagonals and stores them in a dictionary. 
    Each key ((i,j)) in the dictionary block_{i,j}, and the value is 
    another dictionary mapping diagonal indices to their values.

    Args:
        matrix (scipy.sparse.csr_matrix): A 4D tensor representing a weight matrix 
            for a fully-connected or convolutional layer. The shape must 
            conform to (num_blocks_y, num_blocks_x, slots, slots).
        slots (int): The number of SIMD plaintext slots, dictating the 
            block size.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) corresponding 
              to the (i, j)th (slots, slots) block of `matrix`. The value 
              for each key is another dictionary that maps diagonal indices 
              within the block to the diagonal's tensor values.

    Examples:
        >>> matrix = torch.tensor([[[[ 0,  1,  2,  3],
                                     [ 4,  5,  6,  7],
                                     [ 8,  9, 10, 11],
                                     [12, 13, 14, 15]]]])
        >>> # Example with slots=4, showing processing of a single block
        >>> print(diagonalize(matrix, slots=4)) 
        {(0, 0): {0: [0., 5., 10., 15.], 
                  1: [1., 6., 11., 12.], 
                  2: [2., 7., 8., 13.], 
                  3: [3., 4., 9., 14.]}}

        >>> # Example with slots=2, showing processing of four blocks or 
              sub-matrices
        >>> print(diagonalize(matrix, slots=2)) 
        {(0, 0): {0: [0., 5.], 
                  1: [1., 4.]}, 
         (0, 1): {0: [2., 7.], 
                  1: [3., 6.]}, 
         (1, 0): {0: [8., 13.], 
                  1: [9., 12.]}, 
         (1, 1): {0: [10., 15.], 
                  1: [11., 14.]}}
    """

    matrix_height, matrix_width = matrix.shape
    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)

    if num_block_rows == 1 and embed_method == "hybrid" and not is_last_layer:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    # ORION_SLICE_HEIGHT=1: size each column block by the rows actually
    # live in it instead of by the layer's padded output height.
    #
    # The number of diagonals a block yields IS its height, and every
    # diagonal is a full num_slots plaintext regardless of how much of it
    # is nonzero. For a block-diagonal weight matrix (26 independent
    # embedding tables concatenated into one transform, say) a given
    # column block touches only one table's rows -- 16 of 416 -- yet is
    # read at the padded full height 512, so it emits 512 diagonals that
    # are each 1/32 nonzero rather than 16 that are full. Measured on
    # HE-LRM@500,000: the sparse layer's 18 column blocks span 16 to 160
    # live rows each, 17 of them shrink, and peak device memory falls
    # 40.8 -> 28.8GB at matching MAE (0.000496 -> 0.000494).
    #
    # Self-disabling: a dense block's live rows span every output row, so
    # its height is unchanged and extraction is bit-identical. Only
    # block-structured layers (grouped conv, per-head projections, banked
    # embeddings) see any difference.
    #
    # Cost: a shorter block replicates the input further, so it needs
    # log2(num_slots/h) fold rotations rather than the layer-wide
    # log2(num_slots/block_height). That stays cheap because the fold is
    # amortised per distinct height, not per block: placement and fold
    # are both cyclic rotations, so they commute, letting the evaluator
    # place each block, accumulate everything sharing a height, and fold
    # each height group once -- a handful of folds for a layer with tens
    # of blocks.
    per_slice = (os.environ.get("ORION_SLICE_HEIGHT", "").strip().lower()
                 in ("1", "true", "yes", "on"))
    slice_meta = None
    eligible = (per_slice and num_block_rows == 1
                and embed_method == "hybrid" and not is_last_layer)
    if eligible:
        slice_meta = _slice_heights(
            matrix, num_slots, num_block_cols, block_height)
    if per_slice and os.environ.get("ORION_SLICE_DEBUG", "") == "1":
        print(f"[slice-pack] {layer_name}: shape={matrix.shape} "
              f"slots={num_slots} bh={block_height} "
              f"blocks=({num_block_rows},{num_block_cols}) "
              f"last={is_last_layer} embed={embed_method} "
              f"eligible={eligible} sliced={len(slice_meta) if slice_meta else 0}",
              flush=True)

    # Inflate dimensions of the sparse matrix. With per-block heights a
    # block reads rows [offset, offset+h), which can reach past the
    # layer's padded height, so make room for the deepest one.
    row_extent = num_block_rows * block_height
    if slice_meta:
        row_extent = max(row_extent,
                         max(m[0] + m[1] for m in slice_meta.values()))
        # The evaluator needs the layer's natural output period to build
        # each group's mask: a shrunk block's result repeats every
        # `height` slots, and the mask keeps the one copy per `period`
        # window that is in its correct place.
        slice_meta = {"period": block_height, "blocks": slice_meta}
    matrix.resize(row_extent, num_block_cols * num_slots)

    # Prepare indices for diagonal extraction
    row_idx = torch.arange(block_height).repeat(num_slots // block_height)
    col_idx = torch.arange(block_height)[:, None] + torch.arange(num_slots)[None, :]
    col_idx = torch.where(col_idx >= num_slots, col_idx - num_slots, col_idx)

    diagonals_by_block = {}
    total_diagonals = 0

    # Decide whether to render Rich cards/progress based on logger level
    show_cards = ORION_LOGGER.getEffectiveLevel() == logging.INFO

    start_time = time.time()

    # Optional Rich UI setup
    if show_cards:
        # Compute a bar width that matches the card content
        bar_width = _compute_bar_width(
            embed_method,
            (matrix_height, matrix_width),
            (num_block_rows, num_block_cols),
            block_height,
            matrix.shape,
            output_rotations,
            "-",
            "-",
        )
        progress = Progress(
            BarColumn(bar_width=bar_width),
            expand=False,
            console=_CONSOLE,
            transient=False,
        )
        task = progress.add_task("", total=num_block_rows * num_block_cols)

        # Initial panel with placeholders
        title = f"{layer_name or 'Layer'}"
        panel = _packing_panel(
            title=title,
            embed_method=embed_method,
            original_shape=(matrix_height, matrix_width),
            num_blocks=(num_block_rows, num_block_cols),
            block_height=block_height,
            resized_shape=matrix.shape,
            output_rotations=output_rotations,
            elapsed="-",
            total_diagonals="-",
            progress_renderable=progress,
        )

        # Append to global cards and render in columns
        idx = len(_PACKING_CARDS)
        _PACKING_CARDS.append(panel)
        _update_live()

    # Main compute loop (no tqdm fallback)
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            # Uniform path unless this block was shrunk; the index
            # tensors above are reused whenever the height matches so the
            # default route allocates nothing extra.
            bh, r_idx, c_idx = block_height, row_idx, col_idx
            row_start = num_slots * block_row
            if slice_meta and block_col in slice_meta["blocks"]:
                row_start, bh, _ = slice_meta["blocks"][block_col]
                r_idx = torch.arange(bh).repeat(num_slots // bh)
                c_idx = torch.arange(bh)[:, None] + torch.arange(num_slots)[None, :]
                c_idx = torch.where(c_idx >= num_slots, c_idx - num_slots, c_idx)

            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start: row_start + bh,
                col_start: col_start + num_slots,
            ]
            block_dense = torch.tensor(block_sparse.todense(), dtype=torch.float32)
            block_diagonals = block_dense[r_idx, c_idx]

            # Collect non-zero diagonals
            nonzero_diagonals = {}
            for i in range(bh):
                if torch.any(block_diagonals[i]):
                    nonzero_diagonals[i] = block_diagonals[i].tolist()

            total_diagonals += len(nonzero_diagonals)
            diagonals_by_block[(block_row, block_col)] = (
                nonzero_diagonals or {0: [0.0] * num_slots}
            )

            if show_cards:
                progress.advance(task, 1)
                _update_live()

    elapsed_time = time.time() - start_time

    if show_cards:
        # Final update with elapsed and totals
        final_panel = _packing_panel(
            title=title,
            embed_method=embed_method,
            original_shape=(matrix_height, matrix_width),
            num_blocks=(num_block_rows, num_block_cols),
            block_height=block_height,
            resized_shape=matrix.shape,
            output_rotations=output_rotations,
            elapsed=f"{elapsed_time:.2f}",
            total_diagonals=total_diagonals,
            progress_renderable=progress,
        )
        _PACKING_CARDS[idx] = final_panel
        _update_live()

        # If this is the last layer, close the live layout and reset
        if is_last_layer:
            global _PACKING_LIVE
            if _PACKING_LIVE is not None:
                _PACKING_LIVE.stop()
                _PACKING_LIVE = None
            _PACKING_CARDS.clear()

    return diagonals_by_block, output_rotations, slice_meta

def plot_toeplitz(matrix, save_path=""):
    if isinstance(matrix, sp.csr_matrix):
        matrix = matrix.todense()

    if matrix.ndim != 2:
        raise ValueError(f"Cannot plot matrix of dimension {matrix.ndim}")

    plt.imshow(matrix)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


#---------------------#
#   BatchNorm Logic   #
#---------------------#

def pack_bn1d(bn1d_layer):
    N = bn1d_layer.input_shape[0]
    on_running_mean = bn1d_layer.on_running_mean
    on_inv_running_std = 1 / torch.sqrt(bn1d_layer.running_var + bn1d_layer.eps)
    on_weight = bn1d_layer.on_weight if bn1d_layer.affine else None 
    on_bias = bn1d_layer.on_bias if bn1d_layer.affine else None 

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N),
        on_weight.flatten().repeat(N) if on_weight is not None else None, 
        on_bias.flatten().repeat(N) if on_bias is not None else None
    )

def pack_bn2d(bn2d_layer):
    N, Ci, Hi, Wi = bn2d_layer.input_shape
    on_Ci, on_Hi, on_Wi = bn2d_layer.fhe_input_shape[1:]

    on_running_mean = torch.zeros(on_Ci, on_Hi, on_Wi)
    on_inv_running_std = torch.zeros(on_Ci, on_Hi, on_Wi)

    mean = bn2d_layer.on_running_mean.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
    var = bn2d_layer.on_running_var.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

    mean_mpx = multiplex(mean, bn2d_layer.input_gap).squeeze(0)
    var_mpx = multiplex(var, bn2d_layer.input_gap).squeeze(0)

    mC, mH, mW = mean_mpx.shape
    on_running_mean[:mC, :mH, :mW] = mean_mpx 
    on_inv_running_std[:mC, :mH, :mW] = 1 / torch.sqrt(var_mpx + bn2d_layer.eps)

    on_weight = None 
    on_bias = None
    if bn2d_layer.affine:
        on_weight = torch.zeros(on_Ci, on_Hi, on_Wi)
        on_bias = torch.zeros(on_Ci, on_Hi, on_Wi)

        weight = bn2d_layer.on_weight.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
        bias = bn2d_layer.on_bias.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

        weight_mpx = multiplex(weight, bn2d_layer.input_gap).squeeze(0)
        bias_mpx = multiplex(bias, bn2d_layer.input_gap).squeeze(0)

        mC, mH, mW = weight_mpx.shape
        on_weight[:mC, :mH, :mW] = weight_mpx 
        on_bias[:mC, :mH, :mW] = bias_mpx 

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N), 
        on_weight.flatten().repeat(N) if on_weight is not None else None, 
        on_bias.flatten().repeat(N) if on_bias is not None else None
    )
