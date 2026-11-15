"""HICO-DET mAP, and export of V-COCO predictions for Gupta's role evaluator."""
import pickle
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .datasets import HOIDataset, collate
from .model import to_detections


def match_pairs(gt_h, gt_o, det_h, det_o, scores, min_iou=0.5):
    """Label detections as true (1) or false (0) positives.

    A detected pair overlaps a ground-truth pair by the smaller of its human and object
    IoUs. Each detection is assigned to the ground-truth pair it overlaps most; of the
    detections assigned to one ground-truth pair, only the highest-scoring is correct.
    """
    iou = torch.min(box_iou(gt_h, det_h), box_iou(gt_o, det_o))  # (G, M)
    best_iou, best_gt = iou.max(0)
    labels = torch.zeros_like(scores)
    for g in range(len(gt_h)):
        candidates = torch.nonzero((best_gt == g) & (best_iou > min_iou)).squeeze(1)
        if len(candidates):
            labels[candidates[scores[candidates].argmax()]] = 1
    return labels


def average_precision_11pt(scores, labels, num_gt):
    """11-point interpolated AP: mean of the best precision at recall >= 0, 0.1, ..., 1."""
    if len(scores) == 0 or num_gt == 0:
        return 0.0
    tp = labels[scores.argsort(descending=True)].double().cumsum(0)
    precision = tp / torch.arange(1, len(tp) + 1)
    recall = tp / num_gt
    return sum(
        precision[recall >= t].max().item() for t in torch.linspace(0, 1, 11, dtype=torch.float64) if (recall >= t).any()
    ) / 11


def _loader(dataset, workers):
    return DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=workers)


@torch.no_grad()
def hico_map(model, dataset, device, workers=4):
    """Full / rare / non-rare mAP over the 600 HICO-DET interaction classes.

    Rare classes have fewer than 10 training instances. `dataset` must use eval_transform.
    """
    model.eval()
    to_hoi = torch.full((80, 117), -1)
    for hoi, obj, verb in dataset.meta["correspondence"]:
        to_hoi[obj, verb] = hoi

    scores, hois, labels = [], [], []
    for images, targets in _loader(dataset, workers):
        target = targets[0]
        det = {k: v.cpu() for k, v in to_detections(model([images[0].to(device)])[0]).items()}
        hoi = to_hoi[det["objects"], det["labels"]]
        label = torch.zeros_like(det["scores"])
        for c in hoi.unique():
            gt, d = target["hoi"] == c, hoi == c
            if gt.any():
                label[d] = match_pairs(
                    target["boxes_h"][gt], target["boxes_o"][gt], det["boxes_h"][d], det["boxes_o"][d], det["scores"][d]
                )
        scores.append(det["scores"])
        hois.append(hoi)
        labels.append(label)
    scores, hois, labels = torch.cat(scores), torch.cat(hois), torch.cat(labels)

    num_gt = Counter(h for _, r in dataset.items for h in r["hoi"])
    ap = torch.tensor([average_precision_11pt(scores[hois == c], labels[hois == c], num_gt[c]) for c in range(600)])
    train_count = Counter(h for _, r in HOIDataset("hicodet", "train2015").items for h in r["hoi"])
    rare = torch.tensor([train_count[c] < 10 for c in range(600)])
    return dict(full=ap.mean().item() * 100, rare=ap[rare].mean().item() * 100, non_rare=ap[~rare].mean().item() * 100)


class VCOCOResult(defaultdict):
    """One detection in the format Gupta's evaluator reads; absent actions score zero."""

    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)

    def __missing__(self, key):
        return 0.0 if key.endswith("_agent") else [0.0, 0.0, 0.1, 0.1, 0.0]


@torch.no_grad()
def dump_vcoco(model, dataset, device, path, workers=4):
    """Write test predictions, in original image coordinates, to a pickle file. `dataset` must use eval_transform."""
    model.eval()
    actions = [a.split() for a in dataset.meta["classes"]]  # e.g. ["hold", "obj"]
    results = []
    for images, targets in _loader(dataset, workers):
        image, target = images[0], targets[0]
        det = {k: v.cpu() for k, v in to_detections(model([image.to(device)])[0]).items()}
        (oh, ow), (h, w) = target["orig_size"], image.shape[-2:]
        scale = torch.tensor([ow / w, oh / h, ow / w, oh / h])
        image_id = int(target["filename"].split("_")[-1].split(".")[0])
        for bh, bo, s, a in zip(det["boxes_h"] * scale, det["boxes_o"] * scale, det["scores"].tolist(), det["labels"]):
            verb, role = actions[a]
            results.append(VCOCOResult(**{
                "image_id": image_id, "person_box": bh.tolist(),
                f"{verb}_agent": s, f"{verb}_{role}": bo.tolist() + [s],
            }))
    with open(path, "wb") as f:
        pickle.dump(results, f)
