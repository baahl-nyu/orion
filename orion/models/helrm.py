import torch
import torch.nn as nn
import orion.nn as on


class HELRM(on.Module):
    def __init__(self, num_dense=4, vocab_sizes=[4, 3, 2], hidden_dim=2):
        super(HELRM, self).__init__()

        self.dense_size = num_dense
        self.sparse_size = sum(vocab_sizes)
        self.hidden_dim = hidden_dim

        # For extracting and concatenating the dense and sparse features
        self.total_size = self.dense_size + self.sparse_size
        self.num_top_inp = self.hidden_dim * (1 + len(vocab_sizes))

        # Bottom MLP for the dense features
        self.bot_l = nn.Sequential(
            on.Extract(self.total_size, self.dense_size, 0),
            on.Linear(self.dense_size, 3),
            on.ReLU(),
            on.Linear(3, self.hidden_dim),
            on.ReLU(),
            self.generate_concat_transforms(
                in_features=self.hidden_dim, out_features=self.num_top_inp, offset=0
            ),
        )

        # Embedding for the sparse features
        self.embs = nn.Sequential(
            on.ExtractSparse(self.dense_size, self.sparse_size),
            self.generate_sparse_embedding(vocab_sizes, self.hidden_dim),
            self.generate_concat_transforms(
                in_features=self.hidden_dim * len(vocab_sizes),
                out_features=self.num_top_inp,
                offset=self.hidden_dim,
            ),
        )

        # Add (concatenate) the dense and sparse features
        self.add = on.Add()

        # Top MLP for the interaction between the dense and sparse features
        self.top_l = nn.Sequential(
            on.Linear(self.num_top_inp, 4),
            on.ReLU(),
            on.Linear(4, 2),
            on.ReLU(),
            on.Linear(2, 1),
            # nn.Sigmoid()
        )

    def generate_concat_transforms(self, in_features, out_features, offset):
        lin = on.Linear(in_features, out_features, bias=False)
        lin.weight.data[:] = 0.0
        lin.weight.data[offset : offset + in_features, :in_features] = torch.eye(
            in_features
        )
        return lin

    def generate_sparse_embedding(self, vocab_sizes, hidden_dim):
        emb = on.Embedding(sum(vocab_sizes), hidden_dim * len(vocab_sizes))
        emb.weight.data[:] = 0.0
        return emb

    def forward(self, x):
        dense_out = self.bot_l(x)
        sparse_out = self.embs(x)
        interact = self.add(dense_out, sparse_out)
        out = self.top_l(interact)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    inp = torch.tensor(
        [
            [
                0.1753,
                -0.9315,
                -1.5055,
                -0.6610,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ]
    )
    helrm = HELRM(num_dense=4, vocab_sizes=[4, 3, 2], hidden_dim=2)
    helrm.load_state_dict(torch.load("helrm.pth"))
    out = helrm(inp)
    print(out)
