import math
from os.path import getsize, join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import orion.nn as on


dense_features = 13
sparse_fields = 26
embedding_features = 16

criteo_train_parts = 6
criteo_total_parts = 7

bottom_widths = [dense_features, 512, 256, 64, embedding_features]
top_widths = [embedding_features * (sparse_fields + 1), 512, 256, 1]


def load_sparse_sizes(path):
    return [int(size) for size in np.fromfile(path, dtype=np.int32)]


class Criteo(Dataset):
    def __init__(self, data_dir, part):
        rows = getsize(join(data_dir, "dense.bin"))
        rows //= dense_features * np.dtype(np.float32).itemsize
        train_rows = rows * criteo_train_parts // criteo_total_parts

        self.start = 0 if part == "train" else train_rows
        self.stop = train_rows if part == "train" else rows
        self.dense = np.memmap(
            join(data_dir, "dense.bin"),
            dtype=np.float32,
            mode="r",
            shape=(rows, dense_features),
        )
        self.sparse = np.memmap(
            join(data_dir, "sparse.bin"),
            dtype=np.int32,
            mode="r",
            shape=(rows, sparse_fields),
        )
        self.labels = np.memmap(
            join(data_dir, "label.bin"),
            dtype=np.int32,
            mode="r",
            shape=(rows,),
        )

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        row = self.start + int(index)
        return {
            "dense": torch.tensor(self.dense[row], dtype=torch.float32),
            "sparse": torch.tensor(self.sparse[row], dtype=torch.long),
            "label": torch.tensor([self.labels[row]], dtype=torch.float32),
        }


def digits(index, base, width):
    index = torch.as_tensor(index, dtype=torch.long)
    out = []
    for _ in range(width):
        out.append(torch.remainder(index, base))
        index = torch.div(index, base, rounding_mode="floor")
    return torch.stack(out)


def radix_width(categories, base):
    return math.ceil(math.log(categories, base))


def expanded_width(categories, compress_threshold, base):
    if categories > compress_threshold:
        return base * radix_width(categories, base)
    return categories


class FullTable(nn.Module):
    def __init__(self, categories):
        super().__init__()
        self.table = nn.Embedding(categories, embedding_features)

        bound = 1 / math.sqrt(max(categories, 5))
        nn.init.uniform_(self.table.weight, -bound, bound)

    def forward(self, indices):
        return self.table(indices)

    def expanded_weight(self):
        return self.table.weight.data


class RadixTable(nn.Module):
    def __init__(self, categories, base):
        super().__init__()
        self.base = base
        self.width = radix_width(categories, base)
        self.tables = nn.ModuleList(
            nn.Embedding(base, embedding_features)
            for _ in range(self.width)
        )

        bound = 1 / math.sqrt(categories)
        for table in self.tables:
            nn.init.uniform_(table.weight, -bound, bound)

    def forward(self, indices):
        pieces = []
        index_digits = digits(indices, self.base, self.width)
        for table, digit in zip(self.tables, index_digits):
            digit = digit.to(indices.device)
            pieces.append(table(digit))

        return torch.stack(pieces).sum(dim=0)

    def expanded_weight(self):
        return torch.cat([table.weight.data for table in self.tables])


def categorical_table(categories, compress_threshold, base):
    if categories > compress_threshold:
        return RadixTable(categories, base)
    return FullTable(categories)


def mlp(widths, last_relu):
    layers = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(on.Linear(left, right))
        if last_relu or index < len(widths) - 2:
            layers.append(on.ReLU())
    return nn.Sequential(*layers)


def bottom_mlp():
    return mlp(bottom_widths, last_relu=True)


def top_mlp():
    return mlp(top_widths, last_relu=False)


class DLRM(nn.Module):
    def __init__(self, sparse_sizes, compress_threshold=20000, base=4):
        super().__init__()
        self.bottom = bottom_mlp()
        self.tables = nn.ModuleList(
            [
                categorical_table(size, compress_threshold, base)
                for size in sparse_sizes
            ]
        )
        self.top = top_mlp()

    def forward(self, dense, sparse):
        dense = dense.reshape(-1, dense_features)
        sparse = sparse.reshape(-1, sparse_fields)

        dense = self.bottom(dense)
        sparse = [
            table(sparse[:, field])
            for field, table in enumerate(self.tables)
        ]
        return self.top(torch.cat([dense, *sparse], dim=1))


def copy_into(width, output_width, start):
    layer = on.Linear(width, output_width, bias=False)
    layer.weight.data[:] = 0.0
    layer.weight.data[start : start + width, :width] = torch.eye(width)
    return layer


class HELRM(on.Module):
    def __init__(self, sparse_sizes, compress_threshold=20000, base=4):
        super().__init__()
        self.sparse_sizes = list(sparse_sizes)
        self.compress_threshold = compress_threshold
        self.base = base
        sparse_width = sum(
            expanded_width(size, compress_threshold, base)
            for size in self.sparse_sizes
        )

        top_input = top_widths[0]
        sparse_output = sparse_fields * embedding_features

        self.bottom = bottom_mlp()
        self.bottom_to_top = copy_into(embedding_features, top_input, 0)
        self.sparse = on.Embedding(sparse_width, sparse_output)
        self.sparse_to_top = copy_into(sparse_output, top_input, embedding_features)
        self.add = on.Add()
        self.top = top_mlp()

    def expanded_id(self, index, size):
        if size <= self.compress_threshold:
            out = torch.zeros(size)
            out[index] = 1.0
            return out

        out = []
        for digit in digits(index, self.base, radix_width(size, self.base)):
            piece = torch.zeros(self.base)
            piece[digit] = 1.0
            out.append(piece)
        return torch.cat(out)

    def expanded_sparse(self, sparse):
        sparse = sparse.reshape(sparse_fields)
        fields = [
            self.expanded_id(index, size)
            for index, size in zip(sparse, self.sparse_sizes)
        ]
        return torch.cat(fields).unsqueeze(0)

    def fhe_input(self, dense, sparse):
        return dense.reshape(1, dense_features), self.expanded_sparse(sparse)

    def forward(self, dense, expanded_sparse):
        dense = self.bottom_to_top(self.bottom(dense))
        sparse = self.sparse_to_top(self.sparse(expanded_sparse))
        return self.top(self.add(dense, sparse))


def copy_linears(source, target):
    source = [layer for layer in source if isinstance(layer, on.Linear)]
    target = [layer for layer in target if isinstance(layer, on.Linear)]

    for x, y in zip(source, target):
        y.weight.data.copy_(x.weight.data)
        y.bias.data.copy_(x.bias.data)


def export(dlrm, sparse_sizes, compress_threshold, base):
    helrm = HELRM(sparse_sizes, compress_threshold, base)
    with torch.no_grad():
        copy_linears(dlrm.bottom, helrm.bottom)
        copy_linears(dlrm.top, helrm.top)
        tables = [table.expanded_weight() for table in dlrm.tables]
        helrm.sparse.weight.data.copy_(torch.block_diag(*tables).T)
    return helrm
