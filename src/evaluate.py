import os
import argparse
import yaml
import torch
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import validate
from src.utils import get_device, load_checkpoint


def evaluate_checkpoint(checkpoint_path, data_root, batch_size=128, num_workers=4):
    device = get_device()
    print(f"Device: {device}")

    _, val_loader, num_classes = get_dataloaders(data_root, batch_size, num_workers)

    model = build_model(num_classes=num_classes).to(device)
    epoch, best_val_acc = load_checkpoint(checkpoint_path, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch} (best val acc: {best_val_acc:.2f}%)")

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)

    print(f"Val Loss:  {val_loss:.4f}")
    print(f"Val Top-1: {val_top1:.2f}%")
    print(f"Val Top-5: {val_top5:.2f}%")
    return val_top1, val_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
