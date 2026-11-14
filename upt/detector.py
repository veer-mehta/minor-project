"""Frozen DETR-R50 detector and the selection of its boxes for pairing.

DETR itself is the official implementation, loaded through torch.hub at a fixed commit.
"""
from types import SimpleNamespace

import torch
from torchvision.ops import batched_nms, box_convert

DETR_REPO = "facebookresearch/detr:29901c51d7fe8712168b8d0d64351170bc0f83e0"


def build_detr(num_classes):
    """DETR-R50 with untrained transformer weights and all parameters frozen."""
    detr = torch.hub.load(DETR_REPO, "detr_resnet50", pretrained=False, num_classes=num_classes, trust_repo=True)
    for p in detr.parameters():
        p.requires_grad_(False)
    return detr


def load_detr(path):
    """DETR-R50 with the fine-tuned detector weights released by the UPT authors."""
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    detr = build_detr(state["class_embed.weight"].shape[0] - 1)  # last output is "no object"
    detr.load_state_dict(state)
    return detr


def batch_images(images):
    """Pad (3, H, W) images to a common size. The mask is True on padding."""
    h = max(im.shape[1] for im in images)
    w = max(im.shape[2] for im in images)
    batch = images[0].new_zeros(len(images), 3, h, w)
    mask = torch.ones(len(images), h, w, dtype=torch.bool, device=batch.device)
    for i, im in enumerate(images):
        batch[i, :, : im.shape[1], : im.shape[2]] = im
        mask[i, : im.shape[1], : im.shape[2]] = False
    return batch, mask


@torch.no_grad()
def detect(detr, images):
    """Run DETR on a list of normalised (3, H, W) images.

    Returns the backbone feature map (B, 2048, h, w), the image sizes (B, 2) as (H, W),
    and per image a dict of boxes (xyxy, pixels), scores, labels and hidden_states
    for all 100 queries.
    """
    sizes = torch.as_tensor([im.shape[-2:] for im in images], device=images[0].device)
    batch, mask = batch_images(images)
    # DETR's backbone only reads .tensors and .mask from its input
    features, pos = detr.backbone(SimpleNamespace(tensors=batch, mask=mask))
    src, mask = features[-1].decompose()
    hs = detr.transformer(detr.input_proj(src), mask, detr.query_embed.weight, pos[-1])[0][-1]

    scores, labels = detr.class_embed(hs).softmax(-1)[..., :-1].max(-1)  # drop "no object"
    boxes = box_convert(detr.bbox_embed(hs).sigmoid(), "cxcywh", "xyxy")
    boxes = boxes * sizes.flip(1).repeat(1, 2)[:, None]  # relative -> pixels, (W, H, W, H)
    detections = [
        dict(boxes=b, scores=s, labels=l, hidden_states=h) for b, s, l, h in zip(boxes, scores, labels, hs)
    ]
    return src, sizes, detections


def select_proposals(detection, human_idx=0, score_thresh=0.2, min_instances=3, max_instances=15, nms_thresh=0.5):
    """Keep the boxes worth pairing: humans first, then objects, each sorted by score.

    After per-class NMS, each group keeps its boxes scoring at least `score_thresh`,
    but never fewer than `min_instances` (if that many exist) or more than `max_instances`.
    """
    # batched_nms returns indices in decreasing order of score
    keep = batched_nms(detection["boxes"], detection["scores"], detection["labels"], nms_thresh)
    scores, labels = detection["scores"][keep], detection["labels"][keep]
    chosen = []
    for group in (labels == human_idx, labels != human_idx):
        idx = group.nonzero().squeeze(1)
        n = int((scores[idx] >= score_thresh).sum().clamp(min_instances, max_instances))
        chosen.append(idx[:n])
    keep = keep[torch.cat(chosen)]
    return {k: v[keep] for k, v in detection.items()}
