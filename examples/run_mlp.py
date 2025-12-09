import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import get_mnist_datasets, mae, train_on_mnist
from orion.core.mlflow_logger import MLflowLogger, set_logger

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize MLflow logger
logger = MLflowLogger(
    enabled=True,
    experiment_name="orion-mlp",
)
set_logger(logger)
logger.start_run(run_name="mlp_mnist", tags={"model": "MLP", "dataset": "MNIST"})

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/mlp.yml")
trainloader, testloader = get_mnist_datasets(data_dir="../data", batch_size=1)
net = models.MLP()

# Log scheme parameters
logger.log_scheme_params(scheme)

# Train model (optional)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_mnist(net, data_dir="../data", epochs=1, device=device, save_path="../data/models/mlp_mnist.pth")

# Get a test batch to pass through our network
inp, _ = next(iter(testloader))

# Run cleartext inference
net.eval()
out_clear = net(inp)

# Prepare for FHE inference.
# Certain polynomial activation functions require us to know the precise range
# of possible input values. We'll determine these ranges by aggregating
# statistics from the training set and applying a tolerance factor = margin.
with logger.timer("total_fit_time"):
    orion.fit(net, inp, batch_size=128)
with logger.timer("total_compile_time"):
    input_level = orion.compile(net)

# Encode and encrypt the input vector
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# Compare the cleartext and FHE results.
print()
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
precision = -math.log2(dist)
runtime = end - start

print(f"\nMAE: {dist:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Runtime: {runtime:.4f} secs.\n")

# Log inference metrics
logger.log_inference_metrics(
    mae=float(dist), precision=float(precision), runtime=runtime
)

# Log config file as artifact
logger.log_artifact("../configs/mlp.yml", "config")

# End MLflow run
logger.end_run()
