import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.transforms import get_train_transforms, get_val_transforms


class TinyImageNetTrain(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        wnids = sorted(os.listdir(root))
        wnids = [w for w in wnids if os.path.isdir(os.path.join(root, w))]
        self.classes = wnids
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        for wnid in wnids:
            img_dir = os.path.join(root, wnid, 'images')
            for img_path in glob.glob(os.path.join(img_dir, '*.JPEG')):
                self.samples.append((img_path, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        annotations_file = os.path.join(root, 'val_annotations.txt')
        img_dir = os.path.join(root, 'images')

        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                filename, wnid = parts[0], parts[1]
                if wnid in class_to_idx:
                    img_path = os.path.join(img_dir, filename)
                    self.samples.append((img_path, class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_root, batch_size, num_workers):
    train_dataset = TinyImageNetTrain(
        root=os.path.join(data_root, 'train'),
        transform=get_train_transforms(),
    )
    val_dataset = TinyImageNetVal(
        root=os.path.join(data_root, 'val'),
        class_to_idx=train_dataset.class_to_idx,
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, len(train_dataset.classes)
