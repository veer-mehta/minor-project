import os

import pytest
import torch

from upt.detector import batch_images, detect, load_detr, select_proposals


def fake_detection(human_scores, object_scores):
    """Well-separated boxes so NMS removes nothing; label 0 is human."""
    scores = torch.tensor(human_scores + object_scores)
    n = len(scores)
    x = torch.arange(n, dtype=torch.float32) * 100
    return dict(
        boxes=torch.stack([x, x, x + 50, x + 50], 1),
        scores=scores,
        labels=torch.tensor([0] * len(human_scores) + [5] * len(object_scores)),
        hidden_states=torch.randn(n, 256),
    )


@pytest.mark.parametrize(
    "human_scores,object_scores,n_h,n_o",
    [
        ([0.9, 0.1, 0.05, 0.01], [0.8, 0.7, 0.6, 0.5], 3, 4),  # too few confident humans: padded to 3
        ([0.9] * 20, [0.3], 15, 1),  # too many humans: capped at 15; only one object exists
        ([0.5, 0.4, 0.3, 0.25, 0.1], [0.1, 0.1, 0.1, 0.1], 4, 3),
    ],
)
def test_select_proposals_counts(human_scores, object_scores, n_h, n_o):
    p = select_proposals(fake_detection(human_scores, object_scores))
    is_human = p["labels"] == 0
    assert is_human.sum() == n_h and (~is_human).sum() == n_o
    assert is_human[:n_h].all(), "humans come first"
    assert (p["scores"][:n_h].diff() <= 0).all() and (p["scores"][n_h:].diff() <= 0).all()
    assert p["hidden_states"].shape == (n_h + n_o, 256)


def test_select_proposals_applies_nms():
    d = fake_detection([0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3])
    d["boxes"][1] = d["boxes"][0] + 1  # near-duplicate of the best human
    p = select_proposals(d)
    assert (p["labels"] == 0).sum() == 3 and 0.8 not in p["scores"].tolist()


def test_batch_images_pads_and_masks():
    batch, mask = batch_images([torch.ones(3, 4, 6), torch.ones(3, 5, 3)])
    assert batch.shape == (2, 3, 5, 6)
    assert not mask[0, :4, :6].any() and mask[0, 4:].all()
    assert batch[1, :, :, 3:].sum() == 0 and mask[1, :, 3:].all()


@pytest.mark.parametrize("dataset", ["hicodet", "vcoco"])
def test_detect_with_released_weights(dataset, num_classes=80):
    path = f"weights/detr-r50-{dataset}.pth"
    if not os.path.exists(path):
        pytest.skip("weights not downloaded")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detr = load_detr(path).to(device).eval()
    assert detr.class_embed.out_features == num_classes + 1
    assert not any(p.requires_grad for p in detr.parameters())

    images = [torch.randn(3, 480, 640, device=device), torch.randn(3, 512, 384, device=device)]
    features, sizes, detections = detect(detr, images)
    assert features.shape[:2] == (2, 2048) and sizes.tolist() == [[480, 640], [512, 384]]
    for (h, w), d in zip(sizes.tolist(), detections):
        assert d["boxes"].shape == (100, 4) and d["hidden_states"].shape == (100, 256)
        # DETR does not clip, so boxes may poke slightly outside the image
        assert d["boxes"].isfinite().all() and (d["boxes"][:, 2:] > d["boxes"][:, :2]).all()
        assert ((d["scores"] >= 0) & (d["scores"] <= 1)).all() and d["labels"].max() < num_classes
        p = select_proposals(d)
        assert 3 <= len(p["boxes"]) <= 30
