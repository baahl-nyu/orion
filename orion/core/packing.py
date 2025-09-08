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

def pack_conv2d(conv_layer: nn.Module, last: bool):
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    weight = conv_layer.on_weight
    if conv_layer.groups > 1:
        weight = resolve_grouped_conv(conv_layer)

    toeplitz = construct_conv2d_toeplitz(conv_layer, weight)
    diagonals, output_rotations = diagonalize(
        toeplitz,
        slots,
        embed_method,
        last,
        debug=conv_layer.scheme.params.get_debug_status(),
        layer_name=getattr(conv_layer, "name", "Conv2d"),
    )
    
    return diagonals, output_rotations

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
    slots = linear_layer.scheme.params.get_slots()
    embed_method = linear_layer.scheme.params.get_embedding_method()

    weight = construct_linear_matrix(linear_layer)
    diagonals, output_rotations = diagonalize(
        weight,
        slots,
        embed_method,
        last,
        debug=linear_layer.scheme.params.get_debug_status(),
        layer_name=getattr(linear_layer, "name", "Linear"),
    )
    return diagonals, output_rotations

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

    # Inflate dimensions of the sparse matrix
    matrix.resize(num_block_rows * block_height, num_block_cols * num_slots)

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
            row_start = num_slots * block_row
            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start: row_start + block_height,
                col_start: col_start + num_slots,
            ]
            block_dense = torch.tensor(block_sparse.todense(), dtype=torch.float32)
            block_diagonals = block_dense[row_idx, col_idx]

            # Collect non-zero diagonals
            nonzero_diagonals = {}
            for i in range(block_height):
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

    return diagonals_by_block, output_rotations

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
