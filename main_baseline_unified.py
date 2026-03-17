"""
main_baseline_unified.py
========================
统一的 Baseline 方法运行入口 (Unified baseline method runner)

支持的方法 (Supported methods):
  - iafl       : Incentive-Aware Federated Learning (ICLR 2024)
  - cgsv       : Cosine Gradient Shapley Value (NeurIPS 2021)
  - rank       : Rank-based FL incentive (Kong et al., 2022)
  - fedavgft   : FedAvg + local fine-tuning
  - lgfedavg   : Local-Global FedAvg (Liang et al., 2020)

支持的数据分布 (Supported distributions):
  - iid            : 独立同分布 (homogeneous random split)
  - non-iid-dir    : Dirichlet 标签分布倾斜
  - non-iid-size   : 数据量倾斜
  - non-iid-class  : 类别数倾斜

输出指标 (Output metrics):
  - Global accuracy  : 全局/平均测试准确率
  - PCC              : 皮尔逊相关系数 (standalone acc vs. FL acc)
  - IPR              : 激励参与率 (% clients benefiting from FL)

注意 (Note):
  本脚本不修改各 baseline 方法的核心算法逻辑，仅统一实验框架。
  This script does NOT modify the core algorithm of any baseline method.
  It only unifies the experimental framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import copy
import math
import json
import random
import argparse
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.stats import pearsonr

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler


# ============================================================
# 第一部分：模型定义 (Part 1: Model Definitions)
# 复用自 IAFL 原始代码 utils/models.py，仅保留视觉模型
# ============================================================

class SimpleCNN(nn.Module):
    """用于 CIFAR-10 等 3 通道 32x32 图像"""
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


class SimpleCNNMNIST(nn.Module):
    """用于 MNIST / Fashion-MNIST 等 1 通道 28x28 图像"""
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


# ============================================================
# 第二部分：数据集加载 (Part 2: Dataset Loading)
# 不依赖 torchtext，仅支持视觉数据集
# ============================================================

class SimpleDataset(Dataset):
    """简单的 Tensor 数据集"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_dataset(name, data_dir='.data'):
    """
    加载并预处理数据集，返回 (train_data, train_targets, test_data, test_targets, num_classes)
    数据已归一化为 float32 张量，通道在前 (NCHW)。
    """
    if name == 'mnist':
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        train_data = train.data.unsqueeze(1).float().div(255)
        test_data = test.data.unsqueeze(1).float().div(255)
        # 全局归一化
        mean, std = train_data.mean(), train_data.std()
        train_data = train_data.sub_(mean).div_(std)
        test_data = test_data.sub_(mean).div_(std)
        train_targets = train.targets.long()
        test_targets = test.targets.long()
        return train_data, train_targets, test_data, test_targets, 10

    elif name in ['fashion-mnist', 'fmnist']:
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)
        train_data = train.data.unsqueeze(1).float().div(255)
        test_data = test.data.unsqueeze(1).float().div(255)
        mean, std = train_data.mean(), train_data.std()
        train_data = train_data.sub_(mean).div_(std)
        test_data = test_data.sub_(mean).div_(std)
        train_targets = train.targets.long()
        test_targets = test.targets.long()
        return train_data, train_targets, test_data, test_targets, 10

    elif name == 'cifar10':
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
        train_data = torch.from_numpy(train.data).float().div(255).permute(0, 3, 1, 2)
        test_data = torch.from_numpy(test.data).float().div(255).permute(0, 3, 1, 2)
        # Per-channel 归一化
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2470, 0.2435, 0.2616)
        for i, (m, s) in enumerate(zip(means, stds)):
            train_data[:, i].sub_(m).div_(s)
            test_data[:, i].sub_(m).div_(s)
        train_targets = torch.tensor(train.targets).long()
        test_targets = torch.tensor(test.targets).long()
        return train_data, train_targets, test_data, test_targets, 10

    else:
        raise NotImplementedError(f"Dataset '{name}' not supported. "
                                  f"Choose from: mnist, fashion-mnist, cifar10")


# ============================================================
# 第三部分：数据分布划分 (Part 3: Data Partitioning)
# 支持 iid / non-iid-dir / non-iid-size / non-iid-class
# ============================================================

def partition_iid(targets, n_clients):
    """IID: 随机均匀划分"""
    n = len(targets)
    idxs = np.random.permutation(n)
    splits = np.array_split(idxs, n_clients)
    return {i: splits[i].tolist() for i in range(n_clients)}


def partition_noniid_dir(targets, n_clients, alpha=0.5):
    """
    Non-IID Dirichlet 标签分布倾斜
    复用 IAFL 原始 partition.py 的 noniid-labeldir 逻辑
    """
    targets_np = np.array(targets)
    n_train = len(targets_np)
    num_classes = int(targets_np.max()) + 1
    min_size = 0
    min_require_size = 10

    net_dataidx_map = {}
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets_np == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            # 平衡：防止某些客户端分到过多样本
            proportions = np.array(
                [p * (len(idx_j) < n_train / n_clients)
                 for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / (proportions.sum() + 1e-12)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist()
                         for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def partition_noniid_size(targets, n_clients, size_imbalance_ratio=5.0):
    """
    Non-IID Size 数据量倾斜：
    各客户端数据的类别分布近似 IID，但数据量按指数分布不均衡。
    size_imbalance_ratio = max_size / min_size
    """
    n = len(targets)
    targets_np = np.array(targets)

    # 用指数分布生成不均衡比例
    # 使得 max/min ≈ size_imbalance_ratio
    raw = np.exp(np.linspace(0, np.log(size_imbalance_ratio), n_clients))
    proportions = raw / raw.sum()

    # 全局随机打乱
    idxs = np.random.permutation(n)
    split_points = (np.cumsum(proportions) * n).astype(int)[:-1]
    splits = np.split(idxs, split_points)

    # 随机打乱客户端分配顺序，避免编号与大小的系统性对应
    perm = np.random.permutation(n_clients)
    net_dataidx_map = {}
    for i in range(n_clients):
        net_dataidx_map[int(perm[i])] = splits[i].tolist()

    return net_dataidx_map


def partition_noniid_class(targets, n_clients,
                           min_classes_per_client=2,
                           max_classes_per_client=5):
    """
    Non-IID Class 类别数倾斜：
    每个客户端随机分配 [min, max] 范围内数量的类别。
    """
    targets_np = np.array(targets)
    num_classes = int(targets_np.max()) + 1
    max_classes_per_client = min(max_classes_per_client, num_classes)
    min_classes_per_client = min(min_classes_per_client, max_classes_per_client)

    # 为每个客户端分配类别
    client_classes = []
    for i in range(n_clients):
        n_cls = np.random.randint(min_classes_per_client, max_classes_per_client + 1)
        selected = np.random.choice(num_classes, n_cls, replace=False)
        client_classes.append(selected)

    # 统计每个类别被多少客户端选中
    class_to_clients = defaultdict(list)
    for i, classes in enumerate(client_classes):
        for c in classes:
            class_to_clients[c].append(i)

    net_dataidx_map = {i: [] for i in range(n_clients)}

    for k in range(num_classes):
        idx_k = np.where(targets_np == k)[0]
        np.random.shuffle(idx_k)
        clients_for_k = class_to_clients[k]
        if len(clients_for_k) == 0:
            # 该类别没有客户端选中，随机分配给一个
            lucky = np.random.randint(0, n_clients)
            net_dataidx_map[lucky].extend(idx_k.tolist())
        else:
            splits = np.array_split(idx_k, len(clients_for_k))
            for c_idx, split in zip(clients_for_k, splits):
                net_dataidx_map[c_idx].extend(split.tolist())

    # 打乱每个客户端的数据索引
    for i in range(n_clients):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def create_partition(targets, n_clients, distribution, **kwargs):
    """统一分发接口"""
    if distribution == 'iid':
        return partition_iid(targets, n_clients)
    elif distribution == 'non-iid-dir':
        return partition_noniid_dir(targets, n_clients,
                                    alpha=kwargs.get('alpha', 0.5))
    elif distribution == 'non-iid-size':
        return partition_noniid_size(targets, n_clients,
                                     size_imbalance_ratio=kwargs.get('size_imbalance_ratio', 5.0))
    elif distribution == 'non-iid-class':
        return partition_noniid_class(
            targets, n_clients,
            min_classes_per_client=kwargs.get('min_classes_per_client', 2),
            max_classes_per_client=kwargs.get('max_classes_per_client', 5))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ============================================================
# 第四部分：训练工具函数 (Part 4: Training Utilities)
# 复用自 IAFL 原始代码 utils/utils.py，移除 torchtext 依赖
# ============================================================

def train_model(model, loader, loss_fn, optimizer, device,
                local_epochs=1, scheduler=None):
    """本地训练"""
    model.train()
    total_loss = 0
    n_batches = 0
    for _ in range(local_epochs):
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    if scheduler:
        scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    return model, avg_loss


def evaluate(model, loader, loss_fn, device):
    """评估模型，返回 (loss, accuracy)"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            total_loss += loss_fn(outputs, label).item() * len(label)
            correct += (outputs.argmax(1) == label).sum().item()
            total += len(label)
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return avg_loss, accuracy


def compute_grad_update(old_model, new_model, device):
    """计算梯度更新 = new_state - old_state"""
    grad = {}
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    for k in old_sd.keys():
        grad[k] = (new_sd[k].float() - old_sd[k].float()).to(device)
    return grad


def add_gradient_updates(grad1, grad2, weight=1.0):
    """grad1 += weight * grad2 (in-place)"""
    if weight == 0:
        return
    for k in grad1:
        grad1[k].data += grad2[k].data * weight


def add_update_to_model(model, update, weight=1.0):
    """model_params += weight * update"""
    if not update:
        return
    sd = model.state_dict()
    for k in sd.keys():
        sd[k] = sd[k].float() + weight * update[k].float()
    model.load_state_dict(sd)


def add_gradients_to_model(model, gradients, weights):
    """model_params += sum_i(weights[i] * gradients[i])"""
    sd = model.state_dict()
    for k in sd.keys():
        sd[k] = sd[k].float()
        for i in range(len(gradients)):
            sd[k] += gradients[i][k] * weights[i]
    model.load_state_dict(sd)


def add_gradients_to_model_batch(models, gradients, weights_list):
    """
    批量更新多个模型
    models[j] += sum_i(weights_list[j][i] * gradients[i])
    """
    state_dicts = [m.state_dict() for m in models]
    for i, gradient in enumerate(gradients):
        for j, sd in enumerate(state_dicts):
            for k in sd.keys():
                sd[k] = sd[k].float()
                sd[k] += gradient[k] * weights_list[j][i]
    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])


def flatten_grad(grad_dict):
    """将梯度字典展平为一维向量"""
    return torch.cat([grad_dict[k].data.view(-1) for k in grad_dict])


def unflatten_grad(flattened, template):
    """将一维向量还原为梯度字典"""
    grad = {}
    offset = 0
    for k in template:
        n = template[k].numel()
        grad[k] = flattened[offset:offset + n].reshape(template[k].shape)
        offset += n
    return grad


def zero_gradient(model, device):
    """创建全零梯度"""
    grad = {}
    sd = model.state_dict()
    for k in sd.keys():
        grad[k] = torch.zeros_like(sd[k], device=device).float()
    return grad


# ============================================================
# 第五部分：独立训练 (Part 5: Standalone Training)
# 用于计算 PCC 和 IPR 的基准准确率
# ============================================================

def run_standalone_training(create_model_fn, train_loaders, test_loader,
                            loss_fn, optimizer_fn, lr, device,
                            standalone_epochs, logger):
    """
    独立训练每个客户端，返回各客户端的 standalone accuracy 列表
    """
    n_clients = len(train_loaders)
    standalone_accs = []
    logger.info(f"[Standalone] Training {n_clients} clients for {standalone_epochs} epochs each...")

    for i in range(n_clients):
        model = create_model_fn().to(device)
        opt = optimizer_fn(model.parameters(), lr=lr, weight_decay=1e-5)
        loader = train_loaders[i]
        model, _ = train_model(model, loader, loss_fn, opt, device,
                               local_epochs=standalone_epochs)
        _, acc = evaluate(model, test_loader, loss_fn, device)
        standalone_accs.append(acc)
        if (i + 1) % max(1, n_clients // 5) == 0 or i == n_clients - 1:
            logger.info(f"  Client {i+1}/{n_clients}: standalone_acc={acc:.4f}")

    return standalone_accs


# ============================================================
# 第六部分：Baseline 方法实现 (Part 6: Baseline Method Implementations)
# 核心算法逻辑完全保留自 IAFL 原始代码
# ============================================================

def _get_dataset_config(dataset_name):
    """获取数据集相关的默认配置"""
    configs = {
        'mnist':         {'lr': 0.01,  'lr_decay': 0.977, 'batch_size': 64,
                          'cgsv_gamma': 0.5},
        'fashion-mnist': {'lr': 0.01,  'lr_decay': 0.977, 'batch_size': 64,
                          'cgsv_gamma': 0.5},
        'cifar10':       {'lr': 0.001, 'lr_decay': 0.977, 'batch_size': 64,
                          'cgsv_gamma': 0.15},
    }
    return configs.get(dataset_name, configs['cifar10'])


# ---------- FedAvg + Fine-tune ----------

def run_fedavgft(server_model, agent_models, train_loaders, test_loader,
                 loss_fn, optimizer_fn, device, args, logger):
    """
    FedAvg + Fine-tune：标准 FedAvg 训练，结束后每个客户端本地 fine-tune 1 epoch。
    返回各客户端最终 FL 准确率列表。

    核心逻辑复用自 IAFL 原始代码 main_fedavgft.py
    """
    n_clients = args['num_clients']
    num_rounds = args['num_rounds']
    local_epochs = args['local_epochs']
    cfg = _get_dataset_config(args['dataset'])

    # 初始化优化器和调度器
    schedulers = []
    for model in agent_models:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_fn(model.parameters(), lr=cfg['lr'], weight_decay=1e-5),
            gamma=cfg['lr_decay'])
        schedulers.append(scheduler)

    for rnd in range(num_rounds):
        gradients = []
        join_indicator = torch.ones(n_clients).int()  # 全参与

        for i in range(n_clients):
            model = agent_models[i]
            scheduler = schedulers[i]
            opt = optimizer_fn(model.parameters(), lr=scheduler.get_last_lr()[-1])

            model.train()
            backup = copy.deepcopy(model)
            model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                                   device, local_epochs=local_epochs,
                                   scheduler=scheduler)
            gradient = compute_grad_update(backup, model, device)
            gradients.append(gradient)

        # --- Server aggregate: FedAvg ---
        weights = torch.div(join_indicator.float(), join_indicator.float().sum())
        add_gradients_to_model(server_model, gradients, weights)

        # --- Broadcast to all clients ---
        server_sd = server_model.state_dict()
        for model in agent_models:
            model.load_state_dict(server_sd)

        if (rnd + 1) % max(1, num_rounds // 10) == 0 or rnd == num_rounds - 1:
            _, acc = evaluate(server_model, test_loader, loss_fn, device)
            logger.info(f"  [FedAvgFT] Round {rnd+1}/{num_rounds}: global_acc={acc:.4f}")

    # --- Fine-tune ---
    fl_accs = []
    for i in range(n_clients):
        model = agent_models[i]
        opt = optimizer_fn(model.parameters(), lr=cfg['lr'])
        model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                               device, local_epochs=1)
        _, acc = evaluate(model, test_loader, loss_fn, device)
        fl_accs.append(acc)

    return fl_accs


# ---------- LG-FedAvg ----------

def run_lgfedavg(server_model, agent_models, train_loaders, test_loader,
                 loss_fn, optimizer_fn, device, args, logger):
    """
    LG-FedAvg：共享全连接层 (fc1, fc2, classifier)，卷积层保持本地。
    返回各客户端最终 FL 准确率列表。

    核心逻辑复用自 IAFL 原始代码 main_lgfedavg.py
    """
    n_clients = args['num_clients']
    num_rounds = args['num_rounds']
    local_epochs = args['local_epochs']
    cfg = _get_dataset_config(args['dataset'])
    shared_layers = ['fc1', 'fc2', 'classifier']

    schedulers = []
    for model in agent_models:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_fn(model.parameters(), lr=cfg['lr'], weight_decay=1e-5),
            gamma=cfg['lr_decay'])
        schedulers.append(scheduler)

    for rnd in range(num_rounds):
        gradients = []
        join_indicator = torch.ones(n_clients).int()

        for i in range(n_clients):
            model = agent_models[i]
            scheduler = schedulers[i]
            opt = optimizer_fn(model.parameters(), lr=scheduler.get_last_lr()[-1])

            model.train()
            backup = copy.deepcopy(model)
            model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                                   device, local_epochs=local_epochs,
                                   scheduler=scheduler)
            gradient = compute_grad_update(backup, model, device)
            gradients.append(gradient)

        # --- Server aggregate ---
        weights = torch.div(join_indicator.float(), join_indicator.float().sum())
        add_gradients_to_model(server_model, gradients, weights)

        # --- 仅共享指定层 ---
        server_sd = server_model.state_dict()
        for model in agent_models:
            local_sd = model.state_dict()
            for key in local_sd.keys():
                is_shared = any((sl + '.') in key for sl in shared_layers)
                if is_shared:
                    local_sd[key] = server_sd[key]
            model.load_state_dict(local_sd)

        if (rnd + 1) % max(1, num_rounds // 10) == 0 or rnd == num_rounds - 1:
            _, acc = evaluate(server_model, test_loader, loss_fn, device)
            logger.info(f"  [LG-FedAvg] Round {rnd+1}/{num_rounds}: server_acc={acc:.4f}")

    # 评估各客户端
    fl_accs = []
    for i in range(n_clients):
        _, acc = evaluate(agent_models[i], test_loader, loss_fn, device)
        fl_accs.append(acc)

    return fl_accs


# ---------- CGSV ----------

def mask_grad_update_by_order_layer(grad_update, mask_percentile):
    """
    CGSV 的梯度 mask 操作：按层对梯度幅度排序，保留 top mask_percentile 比例。
    复用自 IAFL 原始代码 main_cgsv.py
    """
    grad_update = copy.deepcopy(grad_update)
    mask_percentile = max(0.0, mask_percentile)
    for layer in grad_update:
        layer_mod = grad_update[layer].data.view(-1).abs()
        mask_order = math.ceil(len(layer_mod) * mask_percentile)
        if mask_order == 0:
            grad_update[layer].data = torch.zeros_like(grad_update[layer].data)
        else:
            topk, _ = torch.topk(layer_mod, min(mask_order, len(layer_mod) - 1))
            grad_update[layer].data[grad_update[layer].data.abs() < topk[-1]] = 0
    return grad_update


def run_cgsv(server_model, agent_models, train_loaders, test_loader,
             loss_fn, optimizer_fn, device, args, logger):
    """
    CGSV：基于余弦梯度 Shapley 值的贡献评估 + 梯度 mask 奖励。
    返回各客户端最终 FL 准确率列表。

    核心逻辑复用自 IAFL 原始代码 main_cgsv.py
    """
    n_clients = args['num_clients']
    num_rounds = args['num_rounds']
    local_epochs = args['local_epochs']
    cfg = _get_dataset_config(args['dataset'])

    # CGSV 专属超参数（与原始代码一致）
    Gamma = cfg['cgsv_gamma']
    cgsv_alpha = 0.95
    cgsv_beta = 1.0

    shard_sizes = torch.tensor(args['shard_sizes']).float()

    schedulers = []
    for model in agent_models:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_fn(model.parameters(), lr=cfg['lr'], weight_decay=1e-5),
            gamma=cfg['lr_decay'])
        schedulers.append(scheduler)

    rs = torch.zeros(n_clients, device=device)

    for rnd in range(num_rounds):
        gradients = []
        for i in range(n_clients):
            model = agent_models[i]
            scheduler = schedulers[i]
            opt = optimizer_fn(model.parameters(), lr=scheduler.get_last_lr()[-1])

            model.train()
            backup = copy.deepcopy(model)
            model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                                   device, local_epochs=local_epochs,
                                   scheduler=scheduler)
            gradient = compute_grad_update(backup, model, device)

            # CGSV 梯度归一化
            flat = flatten_grad(gradient)
            norm_val = torch.linalg.norm(flat) + 1e-7
            gradient = unflatten_grad(Gamma * flat / norm_val, gradient)
            gradients.append(gradient)

            model.load_state_dict(backup.state_dict())

        # --- Server aggregate ---
        agg_grad = zero_gradient(server_model, device)
        if rnd == 0:
            weights = shard_sizes / shard_sizes.sum()
        else:
            weights = rs

        for grad, w in zip(gradients, weights):
            add_gradient_updates(agg_grad, grad, weight=w.item())
        add_update_to_model(server_model, agg_grad)

        # --- 更新声誉分数 ---
        flat_agg = flatten_grad(agg_grad)
        phis = torch.tensor(
            [F.cosine_similarity(flatten_grad(g), flat_agg, dim=0, eps=1e-10)
             for g in gradients], device=device)

        rs = cgsv_alpha * rs + (1 - cgsv_alpha) * phis
        rs = torch.clamp(rs, min=1e-3)
        rs = rs / rs.sum()

        q_ratios = torch.tanh(cgsv_beta * rs)
        q_ratios = q_ratios / torch.max(q_ratios)

        # --- Client rewards: 梯度 mask ---
        for i in range(n_clients):
            reward_grad = mask_grad_update_by_order_layer(agg_grad, q_ratios[i].item())
            add_update_to_model(agent_models[i], reward_grad)

        if (rnd + 1) % max(1, num_rounds // 10) == 0 or rnd == num_rounds - 1:
            _, acc = evaluate(server_model, test_loader, loss_fn, device)
            logger.info(f"  [CGSV] Round {rnd+1}/{num_rounds}: server_acc={acc:.4f}")

    fl_accs = []
    for i in range(n_clients):
        _, acc = evaluate(agent_models[i], test_loader, loss_fn, device)
        fl_accs.append(acc)

    return fl_accs


# ---------- Rank ----------

def run_rank(server_model, agent_models, train_loaders, test_loader,
             loss_fn, optimizer_fn, device, args, logger):
    """
    Rank：按验证准确率对客户端排名，高排名客户端聚合更多更新。
    返回各客户端最终 FL 准确率列表。

    核心逻辑复用自 IAFL 原始代码 main_rank.py
    """
    n_clients = args['num_clients']
    num_rounds = args['num_rounds']
    local_epochs = args['local_epochs']
    cfg = _get_dataset_config(args['dataset'])

    schedulers = []
    for model in agent_models:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_fn(model.parameters(), lr=cfg['lr'], weight_decay=1e-5),
            gamma=cfg['lr_decay'])
        schedulers.append(scheduler)

    for rnd in range(num_rounds):
        gradients = []
        local_accs = []

        for i in range(n_clients):
            model = agent_models[i]
            scheduler = schedulers[i]
            opt = optimizer_fn(model.parameters(), lr=scheduler.get_last_lr()[-1])

            model.train()
            backup = copy.deepcopy(model)
            model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                                   device, local_epochs=local_epochs,
                                   scheduler=scheduler)
            gradient = compute_grad_update(backup, model, device)
            gradients.append(gradient)

            # Rank 使用验证准确率排序
            _, acc = evaluate(model, test_loader, loss_fn, device)
            local_accs.append(acc)

            model.load_state_dict(backup.state_dict())

        # --- Client rewards: 按排名分配 ---
        order = np.argsort(local_accs)  # 从低到高
        r_weights = [None] * n_clients
        for rank_pos, client_idx in enumerate(order):
            inclusion = torch.zeros(n_clients)
            inclusion[order[:rank_pos + 1]] = 1
            r_weights[client_idx] = inclusion / inclusion.sum()

        add_gradients_to_model_batch(agent_models, gradients, r_weights)

        if (rnd + 1) % max(1, num_rounds // 10) == 0 or rnd == num_rounds - 1:
            # 取最高 reward 客户端作为参考
            best_idx = order[-1]
            _, acc = evaluate(agent_models[best_idx], test_loader, loss_fn, device)
            logger.info(f"  [Rank] Round {rnd+1}/{num_rounds}: best_client_acc={acc:.4f}")

    fl_accs = []
    for i in range(n_clients):
        _, acc = evaluate(agent_models[i], test_loader, loss_fn, device)
        fl_accs.append(acc)

    return fl_accs


# ---------- IAFL ----------

def run_iafl(server_model, agent_models, train_loaders, test_loader,
             loss_fn, optimizer_fn, device, args, logger,
             standalone_accs=None):
    """
    IAFL：基于贡献度的比例聚合 + 随机恢复策略。
    返回各客户端最终 FL 准确率列表。

    核心逻辑复用自 IAFL 原始代码 main_IAFL.py
    默认使用 standalone_accs 作为贡献度量，kappa=0, q=0（与原论文基准设置一致）。
    """
    n_clients = args['num_clients']
    num_rounds = args['num_rounds']
    local_epochs = args['local_epochs']
    cfg = _get_dataset_config(args['dataset'])
    kappa = 0.0
    q = 0.0

    # 贡献度量：使用 standalone accuracies 或 shard sizes
    if standalone_accs is not None:
        p_i = torch.tensor(standalone_accs).float()
    else:
        p_i = torch.tensor(args['shard_sizes']).float()

    p_ceil = torch.max(p_i)
    reward_weights_gamma = torch.clip(p_i / p_ceil, max=1.0)

    schedulers = []
    for model in agent_models:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_fn(model.parameters(), lr=cfg['lr'], weight_decay=1e-5),
            gamma=cfg['lr_decay'])
        schedulers.append(scheduler)

    for rnd in range(num_rounds):
        # --- Stochastic recovery ---
        for i in range(n_clients):
            if np.random.rand() < q:
                agent_models[i].load_state_dict(server_model.state_dict())

        join_indicator = torch.ones(n_clients).int()

        gradients = []
        for i in range(n_clients):
            model = agent_models[i]
            scheduler = schedulers[i]
            opt = optimizer_fn(model.parameters(), lr=scheduler.get_last_lr()[-1])

            model.train()
            backup = copy.deepcopy(model)
            model, _ = train_model(model, train_loaders[i], loss_fn, opt,
                                   device, local_epochs=local_epochs,
                                   scheduler=scheduler)
            gradient = compute_grad_update(backup, model, device)
            gradients.append(gradient)

            model.load_state_dict(backup.state_dict())

        # --- Client rewards ---
        reward_proportions = reward_weights_gamma ** (1 - kappa)
        reward_size = torch.ceil(reward_proportions * (join_indicator.sum() - 1))

        r_weights = []
        for i in range(n_clients):
            tmp_indicator = join_indicator.clone()
            tmp_indicator[i] = 0
            available = torch.where(tmp_indicator == 1)[0]
            n_include = min(int(reward_size[i].item()), len(available))

            if n_include > 0:
                included = np.random.choice(available.numpy(), n_include, replace=False)
            else:
                included = []

            inclusion = torch.zeros(n_clients).int()
            for idx in included:
                inclusion[idx] = 1
            if join_indicator[i] == 1:
                inclusion[i] = 1

            if inclusion.sum() == 0:
                rw = inclusion.float()
            else:
                rw = inclusion.float() / inclusion.float().sum()
            r_weights.append(rw)

        add_gradients_to_model_batch(agent_models, gradients, r_weights)

        # --- Server update (max reference) ---
        max_idx = torch.argmax(reward_proportions).item()
        server_rw = r_weights[max_idx]
        add_gradients_to_model(server_model, gradients, server_rw)

        if (rnd + 1) % max(1, num_rounds // 10) == 0 or rnd == num_rounds - 1:
            _, acc = evaluate(server_model, test_loader, loss_fn, device)
            logger.info(f"  [IAFL] Round {rnd+1}/{num_rounds}: server_acc={acc:.4f}")

    fl_accs = []
    for i in range(n_clients):
        _, acc = evaluate(agent_models[i], test_loader, loss_fn, device)
        fl_accs.append(acc)

    return fl_accs


# ============================================================
# 第七部分：指标计算 (Part 7: Metrics Computation)
# ============================================================

def compute_metrics(standalone_accs, fl_accs):
    """
    计算三个评估指标:
      - global_accuracy: 所有客户端 FL 准确率的均值
      - PCC: standalone_accs 与 fl_accs 的皮尔逊相关系数
      - IPR: fl_acc >= standalone_acc 的客户端比例
    """
    n = len(fl_accs)
    global_accuracy = np.mean(fl_accs)

    # PCC
    if np.std(standalone_accs) < 1e-9 or np.std(fl_accs) < 1e-9:
        pcc = 0.0
        p_value = 1.0
    else:
        pcc, p_value = pearsonr(standalone_accs, fl_accs)

    # IPR
    ipr = np.mean([1.0 if fl_accs[i] >= standalone_accs[i] else 0.0
                    for i in range(n)])

    return {
        'global_accuracy': float(global_accuracy),
        'PCC': float(pcc),
        'PCC_pvalue': float(p_value),
        'IPR': float(ipr),
    }


# ============================================================
# 第八部分：参数解析与主函数 (Part 8: Argument Parsing & Main)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified Baseline Runner for FL Incentive Mechanisms')

    # --- 方法选择 ---
    parser.add_argument('--method', type=str, required=True,
                        choices=['iafl', 'cgsv', 'rank', 'fedavgft', 'lgfedavg'],
                        help='Baseline method to run')

    # --- 数据集 ---
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='Dataset name')

    # --- 数据分布 ---
    parser.add_argument('--distribution', type=str, default='iid',
                        choices=['iid', 'non-iid-dir', 'non-iid-size', 'non-iid-class'],
                        help='Data distribution type')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (only for non-iid-dir)')
    parser.add_argument('--size_imbalance_ratio', type=float, default=5.0,
                        help='Max/min data size ratio (only for non-iid-size)')
    parser.add_argument('--min_classes_per_client', type=int, default=2,
                        help='Min classes per client (only for non-iid-class)')
    parser.add_argument('--max_classes_per_client', type=int, default=5,
                        help='Max classes per client (only for non-iid-class)')

    # --- 客户端 ---
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of federated clients')

    # --- 训练 ---
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of FL communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local training epochs per round')
    parser.add_argument('--standalone_epochs', type=int, default=50,
                        help='Standalone training epochs for PCC/IPR baseline')

    # --- 模型 ---
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn'],
                        help='Model architecture')

    # --- 其他 ---
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output root directory')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_fn(dataset_name, model_type='cnn'):
    """返回一个可调用的模型创建函数"""
    def _create():
        if dataset_name in ['mnist', 'fashion-mnist', 'fmnist']:
            return SimpleCNNMNIST(input_dim=16 * 4 * 4,
                                  hidden_dims=[120, 84], output_dim=10)
        elif dataset_name == 'cifar10':
            return SimpleCNN(input_dim=16 * 5 * 5,
                             hidden_dims=[120, 84], output_dim=10)
        else:
            raise NotImplementedError(f"No model for dataset: {dataset_name}")
    return _create


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- 设备 ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device('cpu')

    # --- 实验名称和目录 ---
    # 命名规则与 run_experiments_baselines.py 中 collect_metrics() 保持一致
    exp_name = (f"{args.method}_{args.dataset}_{args.distribution}_"
                f"N{args.num_clients}_R{args.num_rounds}_seed{args.seed}")

    exp_dir = os.path.join(args.output_dir, args.method, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # --- 日志 ---
    log_file = os.path.join(exp_dir, 'experiment.log')
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    # 避免重复 handler
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Dataset: {args.dataset}, Distribution: {args.distribution}")
    logger.info(f"Clients: {args.num_clients}, Rounds: {args.num_rounds}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {exp_dir}")
    logger.info("=" * 60)

    # --- 数据加载 ---
    logger.info("[Data] Loading dataset...")
    train_data, train_targets, test_data, test_targets, num_classes = \
        load_dataset(args.dataset)

    # --- 数据划分 ---
    logger.info(f"[Data] Partitioning: {args.distribution}")
    partition_kwargs = {
        'alpha': args.alpha,
        'size_imbalance_ratio': args.size_imbalance_ratio,
        'min_classes_per_client': args.min_classes_per_client,
        'max_classes_per_client': args.max_classes_per_client,
    }
    net_dataidx_map = create_partition(
        train_targets.numpy(), args.num_clients,
        args.distribution, **partition_kwargs)

    shard_sizes = [len(net_dataidx_map[i]) for i in range(args.num_clients)]
    logger.info(f"[Data] Shard sizes: min={min(shard_sizes)}, max={max(shard_sizes)}, "
                f"mean={np.mean(shard_sizes):.0f}")

    # 保存分区信息
    partition_info = {
        'distribution': args.distribution,
        'shard_sizes': shard_sizes,
        'partition_kwargs': {k: v for k, v in partition_kwargs.items()},
    }
    with open(os.path.join(exp_dir, 'partition_info.json'), 'w') as f:
        json.dump(partition_info, f, indent=2)

    # --- 创建数据加载器 ---
    cfg = _get_dataset_config(args.dataset)
    batch_size = cfg['batch_size']

    # 完整训练集（用 SubsetRandomSampler 按 partition 索引取子集）
    full_train_dataset = SimpleDataset(train_data, train_targets)
    train_loaders = []
    for i in range(args.num_clients):
        indices = net_dataidx_map[i]
        sampler = SubsetRandomSampler(indices)
        loader = DataLoader(full_train_dataset, batch_size=batch_size,
                            sampler=sampler, drop_last=False)
        train_loaders.append(loader)

    test_dataset = SimpleDataset(test_data, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # --- 模型创建 ---
    model_fn = create_model_fn(args.dataset, args.model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = optim.Adam

    # --- 独立训练 (for PCC / IPR) ---
    logger.info(f"[Standalone] Starting standalone training ({args.standalone_epochs} epochs)...")
    standalone_accs = run_standalone_training(
        model_fn, train_loaders, test_loader,
        loss_fn, optimizer_fn, cfg['lr'], device,
        args.standalone_epochs, logger)

    logger.info(f"[Standalone] Avg standalone acc: {np.mean(standalone_accs):.4f}")

    # --- 初始化模型 ---
    server_model = model_fn().to(device)
    agent_models = [copy.deepcopy(server_model) for _ in range(args.num_clients)]

    # --- 运行 Baseline ---
    logger.info(f"[FL] Starting {args.method} training ({args.num_rounds} rounds)...")
    run_args = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'dataset': args.dataset,
        'shard_sizes': shard_sizes,
    }

    if args.method == 'fedavgft':
        fl_accs = run_fedavgft(server_model, agent_models, train_loaders,
                               test_loader, loss_fn, optimizer_fn,
                               device, run_args, logger)
    elif args.method == 'lgfedavg':
        fl_accs = run_lgfedavg(server_model, agent_models, train_loaders,
                               test_loader, loss_fn, optimizer_fn,
                               device, run_args, logger)
    elif args.method == 'cgsv':
        fl_accs = run_cgsv(server_model, agent_models, train_loaders,
                           test_loader, loss_fn, optimizer_fn,
                           device, run_args, logger)
    elif args.method == 'rank':
        fl_accs = run_rank(server_model, agent_models, train_loaders,
                           test_loader, loss_fn, optimizer_fn,
                           device, run_args, logger)
    elif args.method == 'iafl':
        fl_accs = run_iafl(server_model, agent_models, train_loaders,
                           test_loader, loss_fn, optimizer_fn,
                           device, run_args, logger,
                           standalone_accs=standalone_accs)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- 计算指标 ---
    logger.info("[Metrics] Computing evaluation metrics...")
    metrics = compute_metrics(standalone_accs, fl_accs)

    # 添加实验元信息
    metrics.update({
        'method': args.method,
        'dataset': args.dataset,
        'distribution': args.distribution,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'standalone_epochs': args.standalone_epochs,
        'seed': args.seed,
        'exp_name': exp_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'standalone_accs': standalone_accs,
        'fl_accs': fl_accs,
    })

    # 添加分布专属参数
    if args.distribution == 'non-iid-dir':
        metrics['alpha'] = args.alpha
    elif args.distribution == 'non-iid-size':
        metrics['size_imbalance_ratio'] = args.size_imbalance_ratio
    elif args.distribution == 'non-iid-class':
        metrics['min_classes_per_client'] = args.min_classes_per_client
        metrics['max_classes_per_client'] = args.max_classes_per_client

    # --- 保存结果 ---
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"[Result] Global Accuracy = {metrics['global_accuracy']:.4f}")
    logger.info(f"[Result] PCC            = {metrics['PCC']:.4f} (p={metrics['PCC_pvalue']:.4e})")
    logger.info(f"[Result] IPR            = {metrics['IPR']:.4f}")
    logger.info(f"[Result] Saved to: {metrics_path}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())