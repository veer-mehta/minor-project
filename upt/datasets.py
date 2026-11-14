"""HICO-DET and V-COCO, read from the JSON annotation files published by the UPT authors."""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

# (annotation file, image folder), relative to the data root
_PATHS = {
    ("hicodet", "train2015"): ("hicodet/instances_train2015.json", "hicodet/hico_20160224_det/images/train2015"),
    ("hicodet", "test2015"): ("hicodet/instances_test2015.json", "hicodet/hico_20160224_det/images/test2015"),
    ("vcoco", "train"): ("vcoco/instances_vcoco_train.json", "coco/train2014"),
    ("vcoco", "val"): ("vcoco/instances_vcoco_val.json", "coco/train2014"),
    ("vcoco", "trainval"): ("vcoco/instances_vcoco_trainval.json", "coco/train2014"),
    ("vcoco", "test"): ("vcoco/instances_vcoco_test.json", "coco/val2014"),
}


class HOIDataset(Dataset):
    """Images that have at least one annotated human-object pair.

    Each item is (PIL image, target). The target holds one row per annotated pair:
    boxes_h, boxes_o (x1, y1, x2, y2 in pixels), object (class index) and labels
    (verb index for HICO-DET, action index for V-COCO). HICO-DET also has hoi.
    """

    def __init__(self, name, partition, data_root="data", transforms=None):
        anno_file, image_dir = _PATHS[(name, partition)]
        with open(os.path.join(data_root, anno_file)) as f:
            anno = json.load(f)
        self.name = name
        self.image_dir = os.path.join(data_root, image_dir)
        self.transforms = transforms

        if name == "hicodet":
            records = anno.pop("annotation")
            filenames = anno.pop("filenames")
            empty = set(anno.pop("empty"))
            self.items = [(filenames[i], r) for i, r in enumerate(records) if i not in empty]
            self.num_classes = len(anno["verbs"])
        else:
            records = anno.pop("annotations")
            self.items = [(r["file_name"], r) for r in records if len(r["actions"]) > 0]
            self.num_classes = len(anno["classes"])
        # Class names and the verb/object lookup tables, used by the model and the evaluators
        self.meta = anno

    def __len__(self):
        return len(self.items)

    def target(self, i):
        filename, r = self.items[i]
        target = {
            "boxes_h": torch.as_tensor(r["boxes_h"], dtype=torch.float32).reshape(-1, 4),
            "boxes_o": torch.as_tensor(r["boxes_o"], dtype=torch.float32).reshape(-1, 4),
            "filename": filename,
        }
        if self.name == "hicodet":
            # HICO-DET boxes are 1-based pixel indices; shift the top-left corner to 0-based coordinates
            target["boxes_h"][:, :2] -= 1
            target["boxes_o"][:, :2] -= 1
            target["object"] = torch.as_tensor(r["object"])
            target["labels"] = torch.as_tensor(r["verb"])
            target["hoi"] = torch.as_tensor(r["hoi"])
        else:
            target["object"] = torch.as_tensor(r["objects"])
            target["labels"] = torch.as_tensor(r["actions"])
        return target

    def __getitem__(self, i):
        target = self.target(i)
        image = Image.open(os.path.join(self.image_dir, target["filename"])).convert("RGB")
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
