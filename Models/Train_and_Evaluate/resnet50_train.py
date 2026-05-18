"""
Src/Exp5_EE_Model/Resnet_Train_and_Evaluate/resnet50_train.py
"""
import torch
import torch.nn as nn
from pathlib import Path
import warnings
from Scripts.Exp5_EE_Model.Models.Resnet50 import MultiEEResNet50, Bottleneck, freeze_layers
from Src.Utils.log_function import save_model_weights, save_train_log
from Src.Utils.utils_function import get_device, get_data_loaders
from Src.paras import DATA_ROOT, WEIGHTS_DIR, RESULT_EE_MODEL_PATH


def make_optimizer(model, base_lr=0.001):
    """只把 requires_grad=True 的参数交给优化器。"""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=base_lr)


def train_one_epoch(model, loader, optimizer, stage_tag, device, criterion):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, stage=stage_tag)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100.0 * total_correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, stage_tag, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x, stage=stage_tag)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100.0 * total_correct / total


def run_stage(model, train_loader, valid_loader, device, criterion, log, lr,
              stage_name, freeze_kwargs, stage_tag, num_epochs=50):
    # 冻结 / 解冻
    freeze_layers(model, **freeze_kwargs)
    optimizer = make_optimizer(model, base_lr=lr)
    print(f"\n=== {stage_name} ===")
    for ep in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, stage_tag, device, criterion)
        vl_loss, vl_acc = validate_one_epoch(model, valid_loader, stage_tag, device, criterion)

        print(f"Epoch {ep:3d}/{num_epochs} | "
              f"train_loss {tr_loss:.4f} | train_acc {tr_acc:6.2f}% | "
              f"val_loss {vl_loss:.4f} | val_acc {vl_acc:6.2f}%")

        # 记录日志
        log['epoch'].append(ep if stage_name == 'Stage‑1' else log['epoch'][-1] + 1)
        log['stage'].append(stage_name)
        log['train_loss'].append(tr_loss)
        log['train_acc'].append(tr_acc)
        log['val_loss'].append(vl_loss)
        log['val_acc'].append(vl_acc)


if __name__ == '__main__':
    # 1) device ---------------------------------------------------------------
    device = get_device()
    if device.type == 'cpu':
        warnings.filterwarnings("ignore", message=".*pin_memory.*")

    # 2) Hyper‑parameters ------------------------------------------------------
    batch_size = 64
    valid_size = 0.1
    random_seed = 42
    epochs_stage = 50  # 每个阶段跑多少 epoch
    lr = 0.001
    blocks_num = [3, 4, 6, 3]  # ResNet‑50
    num_classes = 10  # CIFAR‑10
    include_top = True

    # 3) data ------------------------------------------------------------------
    train_loader, valid_loader, test_loader = get_data_loaders(
        root=DATA_ROOT,
        batch_size=batch_size,
        valid_size=valid_size,
        random_seed=random_seed
    )

    # 4) model -----------------------------------------------------------------
    model = MultiEEResNet50(
        block=Bottleneck,
        blocks_num=blocks_num,
        num_classes=num_classes,
        include_top=include_top
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # -------------------------- 训练三阶段 -------------------------------------
    log = {
        'epoch': [],
        'stage': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # ---- Stage‑1：训练主干 + final classifier ---------------------------------
    run_stage(
        model, train_loader, valid_loader, device, criterion, log, lr,
        stage_name='Stage‑1',
        freeze_kwargs=dict(freeze_x2_fc=True, freeze_x3_fc=True),  # 只锁早退头
        stage_tag='final',
        num_epochs=epochs_stage
    )

    # ---- Stage‑2：只训练 x2_fc -------------------------------------------------
    run_stage(
        model, train_loader, valid_loader, device, criterion, log, lr,
        stage_name='Stage‑2',
        freeze_kwargs=dict(freeze_backbone=True, freeze_x3_fc=True),  # backbone & x3_fc 锁
        stage_tag='x2_fc',
        num_epochs=epochs_stage
    )

    # ---- Stage‑3：只训练 x3_fc -------------------------------------------------
    run_stage(
        model, train_loader, valid_loader, device, criterion, log, lr,
        stage_name='Stage‑3',
        freeze_kwargs=dict(freeze_backbone=True, freeze_x2_fc=True),  # backbone & x2_fc 锁
        stage_tag='x3_fc',
        num_epochs=epochs_stage
    )
    print('\nFinished Training!')

    # 6) 保存模型 ---------------------------------------------------------------
    save_path = save_model_weights(model, "ResNet50_multi_EE", Path(WEIGHTS_DIR))
    print(f"model saved → {save_path}")

    # 7) 保存日志 ---------------------------------------------------------------
    log_path = save_train_log(log, "ResNet50_multi_EE", Path(RESULT_EE_MODEL_PATH))
    print(f"log saved   → {log_path}")