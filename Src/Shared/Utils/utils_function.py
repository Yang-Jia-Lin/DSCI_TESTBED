"""
Src/Utils/utils_function.py
"""
import os
import platform
import subprocess
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

from Src.Deploy.Shared.dataloader import (
    CIFAR10TestDataset,
    build_cifar10_test_transform,
)


class NumpyEncoder(json.JSONEncoder):
    """
    用于解决 JSON 无法直接序列化 Numpy 数据类型的问题
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def open_file(file_path):
    """跨平台打开文件"""
    if platform.system() == "Windows":
        os.startfile(file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(file_path)])
    else:  # Linux
        subprocess.run(["xdg-open", str(file_path)])


def get_device(verbose: bool = True) -> torch.device:
    """
    检查 CUDA 是否可用，返回 torch.device，并可选地打印信息。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        if device.type == 'cuda':
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f'Using CUDA device: {name} \n')
        else:
            print('CUDA not available, using CPU. \n')
    return device


def get_data_loaders(
    root: str,
    batch_size: int,
    valid_size: float,
    random_seed: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    test_difficulty_table_path: str | None = None,
    test_difficulty: str | list[str] | None = None,
    include_difficulty_metadata: bool = False,
    include_image_id: bool = False,
    download: bool = True,
):
    """
    构造 CIFAR10 的 train/valid/test DataLoader。
    返回 tuple(train_loader, valid_loader, test_loader)。
    """
    # 1. 定义 transform
    transform_train = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010]),
    ])
    transform_test = build_cifar10_test_transform()

    # 2. 载入 Dataset
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    test_dataset = CIFAR10TestDataset(
        data_root=root,
        difficulty_table_path=test_difficulty_table_path,
        difficulty=test_difficulty,
        train=False,
        transform=transform_test,
        download=download,
        include_difficulty_metadata=include_difficulty_metadata,
        include_image_id=include_image_id,
    )

    # 3. 划分 train/valid
    num_train = len(train_dataset)
    split = int(valid_size * num_train)
    train_data, valid_data = random_split(
        train_dataset,
        [num_train - split, split],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # 4. DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_test_data_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    difficulty_table_path: str | None = None,
    difficulty: str | list[str] | None = None,
    include_difficulty_metadata: bool = False,
    include_image_id: bool = False,
    download: bool = False,
):
    """
    构造 CIFAR10 的 test DataLoader
    返回 test_loader
    """
    # 1. 定义 transform
    transform_test = build_cifar10_test_transform()

    # 2. 载入 Dataset
    test_dataset = CIFAR10TestDataset(
        data_root=root,
        difficulty_table_path=difficulty_table_path,
        difficulty=difficulty,
        train=False,
        transform=transform_test,
        download=download,
        include_difficulty_metadata=include_difficulty_metadata,
        include_image_id=include_image_id,
    )

    # 3. DataLoader
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return test_loader
