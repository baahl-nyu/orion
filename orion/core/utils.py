import os
import shutil
import zipfile

import ssl
import certifi
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm


def get_mnist_datasets(data_dir, batch_size=128, test_samples=10000, seed=None):
    """
    Loads MNIST datasets and returns training and test DataLoaders.
    
    Parameters:
        data_dir (str): Directory to store/download the MNIST data.
        batch_size (int): Batch size for the DataLoaders.
        test_samples (int): Number of samples to include in the test set.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: (train_loader, test_loader)
    """

    # Create a secure SSL context using certifi. Otherwise we'll get a
    # [SSL: CERTIFICATE_VERIFY_FAILED] error.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl_context

    try:
        if seed is not None:
            torch.manual_seed(seed)  # Set the global seed for reproducibility

        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Try to download and load the training and test datasets
        try:
            train_dataset = datasets.MNIST(
                data_dir, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                data_dir, train=False, download=True, transform=transform
            )
        except Exception as e: 
            raise RuntimeError(
                e + " Could not install MNIST dataset automatically " + 
                "You'll need to download it manually from torchvision.datasets.MNIST()"
            )
 
        # Limit the number of test samples if necessary
        if test_samples < len(test_dataset):
            test_dataset, _ = random_split(
                test_dataset,
                [test_samples, len(test_dataset) - test_samples]
            )

        # Create DataLoaders for training and test datasets
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size
        )

        return train_loader, test_loader
        
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = old_context


def get_cifar_datasets(data_dir, batch_size=128, test_samples=10000, seed=None):
    """
    Loads CIFAR-10 datasets and returns training and test DataLoaders.
    
    Parameters:
        data_dir (str): Directory to store/download the CIFAR-10 data.
        batch_size (int): Batch size for the DataLoaders.
        test_samples (int): Number of samples to include in the test set.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Create a secure SSL context using certifi. Otherwise we'll get a
    # [SSL: CERTIFICATE_VERIFY_FAILED] error.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl_context

    try:
        if seed is not None:
            torch.manual_seed(seed)  # Set the global seed for reproducibility
            
        # Define data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616)
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616)
            ),
        ])

        # Try to download and load the training and test datasets
        try:
            train_dataset = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test
            )
        except Exception as e:
            raise RuntimeError(
                str(e) + " Could not install CIFAR-10 dataset automatically " +
                "You'll need to download it manually from torchvision.datasets.CIFAR10()"
            )

        # Limit the number of test samples if necessary
        if test_samples < len(test_dataset):
            test_dataset, _ = random_split(
                test_dataset,
                [test_samples, len(test_dataset) - test_samples]
            )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size
        )

        return train_loader, test_loader
        
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = old_context


def download_and_prepare_tinyimagenet(data_dir='./data'):
    # Set paths
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_dir = os.path.join(data_dir, 'tiny-imagenet-200')
    zip_filename = os.path.join(data_dir, 'tiny-imagenet-200.zip')

    # Check if dataset already exists
    if not os.path.exists(dataset_dir):
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Create a secure SSL context using certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        old_context = ssl._create_default_https_context
        ssl._create_default_https_context = lambda: ssl_context

        try:
            # Download the dataset
            print("Downloading Tiny-ImageNet dataset...")
            try:
                urllib.request.urlretrieve(url, zip_filename)
            except Exception as e:
                raise RuntimeError(
                    f"{str(e)}\nCould not download Tiny-ImageNet dataset "
                    "automatically. You'll need to download it manually from "
                    "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )
            
            # Extract the dataset
            print("Extracting Tiny-ImageNet dataset...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print("Organizing Tiny-ImageNet dataset...")

            # Organize the training data
            train_dir = os.path.join(dataset_dir, 'train')
            for class_dir in os.listdir(train_dir):
                class_path = os.path.join(train_dir, class_dir)
                if os.path.isdir(class_path):
                    images_dir = os.path.join(class_path, 'images')
                    # Move images up one level
                    for image_file in os.listdir(images_dir):
                        shutil.move(os.path.join(images_dir, image_file), class_path)
                    # Remove the now-empty 'images' folder
                    os.rmdir(images_dir)
                    # Remove the .txt file in the class directory
                    for txt_file in os.listdir(class_path):
                        if txt_file.endswith('.txt'):
                            os.remove(os.path.join(class_path, txt_file))

            # Organize the validation data
            val_dir = os.path.join(dataset_dir, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

            with open(val_annotations_file, 'r') as f:
                annotations = f.readlines()

            for line in annotations:
                # Get the image filename and the corresponding class
                parts = line.split('\t')
                image_filename = parts[0]
                class_dir = parts[1]
                # Create the class directory if it doesn't exist
                class_path = os.path.join(val_dir, class_dir)
                os.makedirs(class_path, exist_ok=True)
                # Move the image to its class directory
                shutil.move(os.path.join(val_images_dir, image_filename), class_path)
            
            # Remove the now-empty 'images' folder and the annotations file
            os.rmdir(val_images_dir)
            os.remove(val_annotations_file)

            print("Tiny-ImageNet dataset preparation complete.")
        finally:
            # Restore the original SSL context
            ssl._create_default_https_context = old_context


def get_tiny_datasets(data_dir, batch_size, test_samples=10000, seed=None):
    """
    Return Tiny-ImageNet train/test DataLoaders.

    This is path-agnostic:
      - If `data_dir` is the parent (contains tiny-imagenet-200/), it works.
      - If `data_dir` is already the dataset root (has train/ and val/), it
        works.
      - If a nested root exists (tiny-imagenet-200/tiny-imagenet-200), it
        resolves it.

    Args:
        data_dir (str): Parent folder OR dataset root.
        batch_size (int): Batch size for DataLoaders.
        test_samples (int): Max number of validation samples to keep.
        seed (int|None): Seed for reproducibility.

    Returns:
        (DataLoader, DataLoader): (train_loader, test_loader)
    """
    def has_train_val(root):
        tdir = os.path.join(root, "train")
        vdir = os.path.join(root, "val")
        return os.path.isdir(tdir) and os.path.isdir(vdir)

    # Candidate roots, in order of likelihood.
    candidates = [
        data_dir,
        os.path.join(data_dir, "tiny-imagenet-200"),
        os.path.join(data_dir, "tiny-imagenet-200", "tiny-imagenet-200"),
    ]

    dataset_root = next((c for c in candidates if has_train_val(c)), None)

    # If not found, attempt download/prepare into the *parent* folder,
    # then re-resolve the root.
    if dataset_root is None:
        download_and_prepare_tinyimagenet(data_dir)
        dataset_root = next((c for c in candidates if has_train_val(c)), None)

    if dataset_root is None:
        tried = ", ".join(candidates)
        raise FileNotFoundError(
            "Could not locate Tiny-ImageNet 'train'/'val' folders. "
            f"Tried roots: {tried}. If you pass the parent folder, the "
            "dataset should appear at <parent>/tiny-imagenet-200/{train,val}. "
            "If you pass the root, it should directly contain {train,val}."
        )

    if seed is not None:
        torch.manual_seed(seed)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4802, 0.4481, 0.3975),
            (0.2302, 0.2265, 0.2262),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4802, 0.4481, 0.3975),
            (0.2302, 0.2265, 0.2262),
        ),
    ])

    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    if test_samples is not None and test_samples < len(test_dataset):
        test_dataset, _ = random_split(
            test_dataset,
            [test_samples, len(test_dataset) - test_samples]
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_on_mnist(
    model, data_dir="./data", epochs=200, batch_size=128, lr=0.1,
    momentum=0.9, weight_decay=5e-4, device="cpu", save_path=None
):
    """
    Train a model on the MNIST dataset.
    """
    train_loader, test_loader = get_mnist_datasets(data_dir, batch_size)
    train(
        model, train_loader, test_loader, epochs, lr, momentum,
        weight_decay, device, save_path
    )


def train_on_cifar(
    model, data_dir="./data", epochs=200, batch_size=128, lr=0.1,
    momentum=0.9, weight_decay=5e-4, device="cpu", save_path=None
):
    """
    Train a model on the CIFAR-10 dataset.
    """
    train_loader, test_loader = get_cifar_datasets(data_dir, batch_size)
    train(
        model, train_loader, test_loader, epochs, lr, momentum,
        weight_decay, device, save_path
    )


def train_on_tiny(
    model, data_dir="./data", epochs=200, batch_size=128, lr=0.1,
    momentum=0.9, weight_decay=5e-4, device="cpu", save_path=None
):
    """
    Train a model on the Tiny-ImageNet dataset.
    """
    train_loader, test_loader = get_tiny_datasets(data_dir, batch_size)
    train(
        model, train_loader, test_loader, epochs, lr, momentum,
        weight_decay, device, save_path
    )


def train(
    model, train_loader, test_loader, epochs, lr, momentum,
    weight_decay, device="cpu", save_path=None
):
    """
    Train the model on the given dataset using SGD and CosineAnnealingLR.
    """

    print(f"\nTraining on device = {device}!")
    device = torch.device(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_acc = 0.0
    
    for epoch in range(epochs):
        train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        test_acc = test_epoch(model, test_loader, criterion, device)

        # Save the model if the test accuracy improves
        if save_path and test_acc > best_acc:
            print(f"Saving model with accuracy: {test_acc:.3f}")
            best_acc = test_acc
            state = {
                "model_state_dict": model.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
            }
            torch.save(state, save_path)

        scheduler.step()

    model.to("cpu")


def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    Perform one training epoch.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(f"\nEpoch: {epoch + 1}")
    train_bar = tqdm(
        enumerate(train_loader), total=len(train_loader),
        desc="Training", leave=False
    )

    for batch_idx, (inputs, targets) in train_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        train_bar.set_postfix({
            "Loss": f"{train_loss / (batch_idx + 1):.3f}",
            "Acc": f"{100. * correct / total:.3f}% ({correct}/{total})"
        })


def test_epoch(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_bar = tqdm(
        enumerate(test_loader), total=len(test_loader),
        desc="Testing", leave=False
    )

    with torch.no_grad():
        for batch_idx, (inputs, targets) in test_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            test_bar.set_postfix({
                "Loss": f"{test_loss / (batch_idx + 1):.3f}",
                "Acc": f"{100. * correct / total:.3f}% ({correct}/{total})"
            })

    return 100. * correct / total

def mae(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have the same size, but got {tensor1.shape} "
            f"and {tensor2.shape}."
        )
    
    mse_value = F.l1_loss(tensor1, tensor2).item()
    return mse_value

def mse(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have the same size, but got {tensor1.shape} "
            f"and {tensor2.shape}."
        )
    
    mse_value = F.mse_loss(tensor1, tensor2).item()
    return mse_value

