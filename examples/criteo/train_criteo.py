import json
from argparse import ArgumentParser
from os import makedirs
from os.path import join

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dlrm import Criteo, DLRM, export, load_sparse_sizes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compress-threshold", type=int, default=20000)
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--out-dir", default="model")
    return parser.parse_args()


def move(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def train(model, loader, optimizer, device, epochs):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_total = 0.0
    last_loss = None
    batches = 0

    model.train()
    for epoch in range(epochs):
        progress = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            mininterval=5,
        )

        for batch in progress:
            batch = move(batch, device)
            optimizer.zero_grad()

            logit = model(batch["dense"], batch["sparse"])
            loss = loss_fn(logit, batch["label"])
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            loss_total += last_loss
            batches += 1
            progress.set_postfix(loss=f"{last_loss:.4f}", refresh=False)

    return {
        "train_loss_last": last_loss,
        "train_loss_mean": loss_total / batches,
        "train_batches": batches,
    }


def evaluate(model, loader, device):
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_total = 0.0
    accuracy_total = 0.0
    rows = 0
    batches = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            logit = model(batch["dense"], batch["sparse"])
            count = int(batch["label"].shape[0])
            prediction = logit >= 0
            label = batch["label"] > 0

            loss_total += loss_fn(logit, batch["label"]).item()
            accuracy_total += (prediction == label).sum().item()
            rows += count
            batches += 1

    return {
        "validation_bce": loss_total / rows,
        "validation_accuracy": accuracy_total / rows,
        "validation_batches": batches,
    }


def save_sample(model, dataset, out_dir):
    sample = dataset[0]
    dense = sample["dense"].unsqueeze(0)
    sparse = sample["sparse"]
    fhe_dense, fhe_sparse = model.fhe_input(dense, sparse)

    with torch.no_grad():
        clear = model(fhe_dense, fhe_sparse).detach().cpu()

    sample_dir = join(out_dir, "sample")
    makedirs(sample_dir, exist_ok=True)
    torch.save(dense, join(sample_dir, "dense.pt"))
    torch.save(sparse, join(sample_dir, "sparse.pt"))
    torch.save(clear, join(sample_dir, "clear.pt"))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    data_dir = "processed"
    out_dir = args.out_dir
    makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_sizes = load_sparse_sizes(join(data_dir, "counts.bin"))
    train_data = Criteo(data_dir, "train")
    validation_data = Criteo(data_dir, "validation")

    model = DLRM(sparse_sizes, args.compress_threshold, args.base).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=args.eval_batch_size)

    metrics = train(model, train_loader, optimizer, device, args.epochs)
    metrics.update(evaluate(model, validation_loader, device))
    model = model.cpu()
    helrm = export(model, sparse_sizes, args.compress_threshold, args.base)
    save_sample(helrm, validation_data, out_dir)

    checkpoint = {
        "sparse_sizes": sparse_sizes,
        "compress_threshold": args.compress_threshold,
        "base": args.base,
        "weights": helrm.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, join(out_dir, "model.pt"))
    with open(join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
