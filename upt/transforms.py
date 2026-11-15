"""Image transforms that keep the human and object boxes of each pair in step with the image.

Same augmentation recipe as DETR, which UPT uses unchanged. Boxes stay in absolute pixels.
"""
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
BOX_KEYS = ("boxes_h", "boxes_o")
PAIR_KEYS = BOX_KEYS + ("object", "labels", "hoi")

_normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_jitter = T.ColorJitter(0.4, 0.4, 0.4)


def hflip(image, target):
    target = dict(target)
    for k in BOX_KEYS:
        b = target[k]
        target[k] = torch.stack([image.width - b[:, 2], b[:, 1], image.width - b[:, 0], b[:, 3]], 1)
    return F.hflip(image), target


def resize(image, target, size, max_size=None):
    """Resize so the short side is `size`, shrinking further if the long side would exceed `max_size`."""
    w, h = image.size
    if max_size is not None and max(w, h) / min(w, h) * size > max_size:
        size = int(round(max_size * min(w, h) / max(w, h)))
    new_w, new_h = (size, int(size * h / w)) if w < h else (int(size * w / h), size)
    scale = torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])
    target = dict(target)
    for k in BOX_KEYS:
        target[k] = target[k] * scale
    return F.resize(image, [new_h, new_w]), target


def random_crop(image, target, min_size=384, max_size=600):
    """Crop a random region and keep the pairs whose two boxes are both still visible."""
    w = random.randint(min_size, min(image.width, max_size))
    h = random.randint(min_size, min(image.height, max_size))
    top, left, h, w = T.RandomCrop.get_params(image, [h, w])
    target = dict(target)
    limit = torch.tensor([w, h, w, h], dtype=torch.float32)
    for k in BOX_KEYS:
        target[k] = torch.min((target[k] - torch.tensor([left, top, left, top])).clamp(min=0), limit)
    keep = torch.stack([(target[k][:, 2:] > target[k][:, :2]).all(1) for k in BOX_KEYS]).all(0)
    for k in PAIR_KEYS:
        if k in target:
            target[k] = target[k][keep]
    return F.crop(image, top, left, h, w), target


def train_transform(image, target):
    if random.random() < 0.5:
        image, target = hflip(image, target)
    image = _jitter(image)
    if random.random() < 0.5:
        image, target = resize(image, target, random.choice([400, 500, 600]))
        image, target = random_crop(image, target)
    image, target = resize(image, target, random.choice(SCALES), max_size=1333)
    return _normalize(image), target


def eval_transform(image, target):
    original = (image.height, image.width)
    image, target = resize(image, target, 800, max_size=1333)
    target["orig_size"] = original
    return _normalize(image), target
