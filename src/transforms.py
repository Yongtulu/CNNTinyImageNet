import random
import numpy as np
from PIL import Image, ImageEnhance
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomCrop:
    def __init__(self, size, padding=8):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        w, h = img.size
        padded_w = w + 2 * self.padding
        padded_h = h + 2 * self.padding
        padded = Image.new(img.mode, (padded_w, padded_h))
        padded.paste(img, (self.padding, self.padding))

        left = random.randint(0, padded_w - self.size)
        top = random.randint(0, padded_h - self.size)
        return padded.crop((left, top, left + self.size, top + self.size))


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        transforms = []
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda x, f=factor: ImageEnhance.Brightness(x).enhance(f))
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda x, f=factor: ImageEnhance.Contrast(x).enhance(f))
        if self.saturation > 0:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda x, f=factor: ImageEnhance.Color(x).enhance(f))

        random.shuffle(transforms)
        for t in transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor


class Normalize:
    # Tiny ImageNet statistics
    def __init__(self, mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821)):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def get_train_transforms():
    return Compose([
        RandomCrop(64, padding=8),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms():
    return Compose([
        ToTensor(),
        Normalize(),
    ])
