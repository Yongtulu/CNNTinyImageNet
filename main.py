import os
import argparse
import yaml
import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import train_one_epoch, validate, build_optimizer, build_scheduler
from src.utils import get_device, save_checkpoint, load_checkpoint, MetricsLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints/last.pt')
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    print("Loading dataset...")
    train_loader, val_loader, num_classes = get_dataloaders(
        config['data_root'],
        config['batch_size'],
        config['num_workers'],
    )
    print(f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images | Classes: {num_classes}")

    model = build_model(num_classes=num_classes, dropout_fc=config['dropout_fc']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

    logger = MetricsLogger()
    best_val_acc = 0.0
    start_epoch = 0
    warmup_epochs = config.get('warmup_epochs', 5)

    if args.resume:
        ckpt_path = os.path.join(config['checkpoint_dir'], 'last.pt')
        start_epoch, best_val_acc = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        # Advance the cosine scheduler to the correct position
        for _ in range(max(0, start_epoch - warmup_epochs)):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best val acc so far: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, config['epochs']):
        # Linear LR warmup (only applies when training from scratch)
        if epoch < warmup_epochs and not args.resume:
            warmup_lr = config['lr'] * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        logger.record(epoch, train_loss, train_acc, val_loss, val_top1, val_top5)
        logger.print_epoch(epoch, config['epochs'])

        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_top1,
        }

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            checkpoint_state['best_val_acc'] = best_val_acc
            save_checkpoint(checkpoint_state, os.path.join(config['checkpoint_dir'], 'best.pt'))
            print(f"  --> New best: {best_val_acc:.2f}%")

        save_checkpoint(checkpoint_state, os.path.join(config['checkpoint_dir'], 'last.pt'))

    logger.save(config['log_dir'])
    print(f"\nTraining complete. Best val top-1: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")
    print(f"Training curves saved to: {config['log_dir']}/training_curves.png")


if __name__ == '__main__':
    main()
