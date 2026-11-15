"""Training loss for the interaction head."""
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def pair_labels(output, target, num_classes, min_iou=0.5):
    """(M, K) 0/1 labels: a pair is positive for an interaction when both of its boxes
    overlap a ground-truth pair of that interaction by at least `min_iou`."""
    labels = output["logits"].new_zeros(len(output["h_idx"]), num_classes)
    if len(target["labels"]):
        iou = torch.min(
            box_iou(output["boxes"][output["h_idx"]], target["boxes_h"]),
            box_iou(output["boxes"][output["o_idx"]], target["boxes_o"]),
        )
        pair, gt = torch.nonzero(iou >= min_iou).unbind(1)
        labels[pair, target["labels"][gt]] = 1
    return labels


def interaction_loss(outputs, targets, alpha=0.5, gamma=0.2, eps=1e-6):
    """Focal loss on the final scores, summed over the batch.

    Returns (loss sum, number of positive labels). The caller divides by the positive
    count, so the normaliser can cover several accumulated batches.
    """
    logits = torch.cat([o["logits"] for o in outputs])
    prior = torch.cat([o["prior"] for o in outputs])
    labels = torch.cat([pair_labels(o, t, logits.shape[1]) for o, t in zip(outputs, targets)])
    # Interactions the detected object class cannot take part in have zero prior and are never predicted
    keep = prior > 0
    logits, prior, labels = logits[keep], prior[keep], labels[keep]

    # The model's score is prior * sigmoid(logit); x is that score written as a logit
    x = torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-8)
    weight = (1 - labels - alpha).abs() * ((labels - torch.sigmoid(x)).abs() + eps) ** gamma
    loss = weight * F.binary_cross_entropy_with_logits(x, labels, reduction="none")
    return loss.sum(), int(labels.sum())
