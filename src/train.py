import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import mps_empty_cache


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, config):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += batch_size

        if (i + 1) % config['log_interval'] == 0:
            running_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}", acc=f"{running_acc:.2f}%")

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

    mps_empty_cache(device)

    avg_loss = total_loss / total_samples
    avg_top1 = total_top1 / total_samples
    avg_top5 = total_top5 / total_samples
    return avg_loss, avg_top1, avg_top5


def build_optimizer(model, config):
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            nesterov=config['nesterov'],
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )


def build_scheduler(optimizer, config):
    if config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['T_max'],
            eta_min=config['eta_min'],
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('step_size', 30),
        gamma=config.get('gamma', 0.1),
    )
