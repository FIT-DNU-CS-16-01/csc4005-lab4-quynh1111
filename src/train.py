from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import (
    EarlyStopping,
    classification_report_dict,
    compute_accuracy,
    count_parameters,
    ensure_dir,
    plot_curves,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    set_seed,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, remaining = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description='Train CRNN for UrbanSound8K environmental sound classification')
    parser.add_argument('--config', type=str, default=pre_args.config)
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới UrbanSound8K.zip hoặc thư mục UrbanSound8K đã giải nén')
    parser.add_argument('--project', type=str, default=cfg.get('project', 'csc4005-lab4-urbansound8k-crnn'))
    parser.add_argument('--run_name', type=str, default=cfg.get('run_name', 'debug_crnn'))
    parser.add_argument('--feature_type', type=str, choices=['logmel', 'mfcc'], default=cfg.get('feature_type', 'logmel'))
    parser.add_argument('--model_name', type=str, choices=['crnn_tiny', 'crnn_small', 'crnn_medium'], default=cfg.get('model_name', 'crnn_small'))
    parser.add_argument('--rnn_type', type=str, choices=['gru', 'lstm'], default=cfg.get('rnn_type', 'gru'))
    parser.add_argument('--bidirectional', action='store_true', default=bool(cfg.get('bidirectional', False)))
    parser.add_argument('--target_sr', type=int, default=int(cfg.get('target_sr', 16000)))
    parser.add_argument('--duration', type=float, default=float(cfg.get('duration', 4.0)))
    parser.add_argument('--n_mels', type=int, default=int(cfg.get('n_mels', 64)))
    parser.add_argument('--n_mfcc', type=int, default=int(cfg.get('n_mfcc', 40)))
    parser.add_argument('--n_fft', type=int, default=int(cfg.get('n_fft', 1024)))
    parser.add_argument('--hop_length', type=int, default=int(cfg.get('hop_length', 512)))
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default=cfg.get('optimizer', 'adamw'))
    parser.add_argument('--scheduler', type=str, choices=['none', 'plateau'], default=cfg.get('scheduler', 'plateau'))
    parser.add_argument('--lr', type=float, default=float(cfg.get('lr', 1e-3)))
    parser.add_argument('--weight_decay', type=float, default=float(cfg.get('weight_decay', 1e-4)))
    parser.add_argument('--dropout', type=float, default=float(cfg.get('dropout', 0.3)))
    parser.add_argument('--epochs', type=int, default=int(cfg.get('epochs', 25)))
    parser.add_argument('--batch_size', type=int, default=int(cfg.get('batch_size', 32)))
    parser.add_argument('--patience', type=int, default=int(cfg.get('patience', 6)))
    parser.add_argument('--seed', type=int, default=int(cfg.get('seed', 42)))
    parser.add_argument('--train_folds', type=str, default=cfg.get('train_folds', '1,2,3,4,5,6,7,8'))
    parser.add_argument('--val_folds', type=str, default=cfg.get('val_folds', '9'))
    parser.add_argument('--test_folds', type=str, default=cfg.get('test_folds', '10'))
    parser.add_argument('--val_size', type=float, default=float(cfg.get('val_size', 0.15)))
    parser.add_argument('--test_size', type=float, default=float(cfg.get('test_size', 0.15)))
    parser.add_argument('--num_workers', type=int, default=int(cfg.get('num_workers', 0)))
    parser.add_argument('--augment', action='store_true', default=bool(cfg.get('augment', False)))
    parser.add_argument('--no_augment', action='store_true', help='Tắt augmentation dù config bật augment')
    parser.add_argument('--use_wandb', action='store_true', default=bool(cfg.get('use_wandb', False)))
    parser.add_argument('--no_wandb', action='store_true', help='Tắt W&B trong trường hợp debug offline')
    args = parser.parse_args(remaining)
    if args.no_augment:
        args.augment = False
    if args.no_wandb:
        args.use_wandb = False
    return args


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = ensure_dir(Path('outputs') / args.run_name)

    data = create_dataloaders(
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        target_sr=args.target_sr,
        duration=args.duration,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        test_folds=args.test_folds,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment=args.augment,
        num_workers=args.num_workers,
    )
    print(f'Resolved data directory: {data.resolved_data_dir}')
    print(f'Classes: {data.class_names}')
    print(f'Feature shape: {data.feature_shape}')

    model = build_model(
        model_name=args.model_name,
        num_classes=len(data.class_names),
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if args.scheduler == 'plateau' else None
    total_params, trainable_params = count_parameters(model)

    use_wandb = args.use_wandb and wandb is not None
    if args.use_wandb and wandb is None:
        print('Cảnh báo: chưa import được wandb. Chạy tiếp ở chế độ không log online.')
    if use_wandb:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))
        wandb.config.update({
            'num_classes': len(data.class_names),
            'class_names': data.class_names,
            'device': str(device),
            'resolved_data_dir': data.resolved_data_dir,
            'feature_shape': data.feature_shape,
            'total_params': total_params,
            'trainable_params': trainable_params,
        })

    history: list[dict[str, float]] = []
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_time = time.perf_counter() - start
        lr_current = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': lr_current,
            'epoch_time_sec': round(epoch_time, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_current:.6f} | sec={epoch_time:.2f}"
        )
        if use_wandb:
            wandb.log(row)
        if early_stopper.step(val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        if early_stopper.should_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    if not (output_dir / 'best_model.pt').exists():
        torch.save(model.state_dict(), output_dir / 'best_model.pt')

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, data.test_loader, criterion, device)
    report = classification_report_dict(y_true, y_pred, data.class_names)
    cm = save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / 'confusion_matrix.png')
    plot_curves(history, output_dir / 'curves.png')
    save_history_csv(history, output_dir / 'history.csv')
    avg_epoch_time = sum(row['epoch_time_sec'] for row in history) / max(len(history), 1)
    metrics = {
        'model_name': args.model_name,
        'feature_type': args.feature_type,
        'rnn_type': args.rnn_type,
        'bidirectional': args.bidirectional,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'avg_epoch_time_sec': avg_epoch_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'class_names': data.class_names,
        'feature_shape': data.feature_shape,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'resolved_data_dir': data.resolved_data_dir,
    }
    save_json(metrics, output_dir / 'metrics.json')
    print(f'Best val acc: {best_val_acc:.4f}')
    print(f'Test acc: {test_acc:.4f}')
    print(f'Average epoch time: {avg_epoch_time:.2f} sec')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Saved outputs to: {output_dir}')

    if use_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'avg_epoch_time_sec': avg_epoch_time,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'confusion_matrix_image': wandb.Image(str(output_dir / 'confusion_matrix.png')),
            'curves_image': wandb.Image(str(output_dir / 'curves.png')),
        })
        wandb.finish()


if __name__ == '__main__':
    main()
