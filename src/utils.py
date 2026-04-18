import os
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def mps_empty_cache(device):
    if device.type == 'mps':
        torch.mps.empty_cache()


def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=None):
    map_location = device if device else 'cpu'
    ckpt = torch.load(filepath, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_acc', 0.0)


class MetricsLogger:
    def __init__(self):
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_top1': [], 'val_top5': [],
        }

    def record(self, epoch, train_loss, train_acc, val_loss, val_top1, val_top5):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_top1'].append(val_top1)
        self.history['val_top5'].append(val_top5)

    def print_epoch(self, epoch, total_epochs):
        print(
            f"Epoch [{epoch+1:3d}/{total_epochs}] "
            f"train_loss={self.history['train_loss'][-1]:.4f} "
            f"train_acc={self.history['train_acc'][-1]:.2f}% "
            f"val_loss={self.history['val_loss'][-1]:.4f} "
            f"val_top1={self.history['val_top1'][-1]:.2f}% "
            f"val_top5={self.history['val_top5'][-1]:.2f}%"
        )

    def save(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        axes[1].plot(epochs, self.history['train_acc'], label='Train Top-1')
        axes[1].plot(epochs, self.history['val_top1'], label='Val Top-1')
        axes[1].plot(epochs, self.history['val_top5'], label='Val Top-5')
        axes[1].set_title('Accuracy (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=120)
        plt.close()
