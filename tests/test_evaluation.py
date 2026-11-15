import pytest
import torch
from PIL import Image

from upt.evaluation import average_precision_11pt, match_pairs
from upt.transforms import hflip, random_crop, resize, eval_transform, train_transform


def boxes(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_match_pairs_one_true_positive_per_ground_truth():
    gt_h, gt_o = boxes([0, 0, 10, 10]), boxes([20, 20, 30, 30])
    det_h = boxes([0, 0, 10, 10], [0, 0, 10, 9], [0, 0, 10, 10])
    det_o = boxes([20, 20, 30, 30], [20, 20, 30, 30], [50, 50, 60, 60])  # last: right human, wrong object
    labels = match_pairs(gt_h, gt_o, det_h, det_o, torch.tensor([0.5, 0.9, 0.8]))
    assert labels.tolist() == [0, 1, 0]  # duplicate with the higher score wins


def test_average_precision():
    assert average_precision_11pt(torch.tensor([0.9, 0.8]), torch.tensor([1.0, 1.0]), 2) == pytest.approx(1.0)
    assert average_precision_11pt(torch.tensor([0.9, 0.8]), torch.tensor([0.0, 0.0]), 2) == 0
    # one of two ground truths found, ranked first: precision 1 up to recall 0.5 -> 6 of 11 points
    assert average_precision_11pt(torch.tensor([0.9, 0.8]), torch.tensor([1.0, 0.0]), 2) == pytest.approx(6 / 11)
    assert average_precision_11pt(torch.tensor([]), torch.tensor([]), 3) == 0


def target():
    return {"boxes_h": boxes([10, 20, 110, 220]), "boxes_o": boxes([300, 100, 400, 200]),
            "object": torch.tensor([5]), "labels": torch.tensor([3])}


def test_resize_and_flip_move_boxes_with_the_image():
    image = Image.new("RGB", (640, 480))
    flipped, t = hflip(image, target())
    assert t["boxes_h"].tolist() == [[530, 20, 630, 220]]
    resized, t = resize(image, target(), 240)
    assert resized.size == (320, 240) and t["boxes_o"].tolist() == [[150, 50, 200, 100]]
    resized, _ = resize(image, target(), 800, max_size=1000)
    assert resized.size == (1000, 750)


def test_crop_drops_pairs_with_a_box_outside():
    torch.manual_seed(0)
    image = Image.new("RGB", (640, 480))
    for _ in range(20):
        cropped, t = random_crop(image, target())
        assert 384 <= cropped.width <= 600 and 384 <= cropped.height <= 480
        assert len(t["boxes_h"]) == len(t["boxes_o"]) == len(t["labels"]) == len(t["object"])
        for k in ("boxes_h", "boxes_o"):
            assert (t[k][:, 2:] > t[k][:, :2]).all() and (t[k][:, 2] <= cropped.width).all()


def eval_transforms_return_normalised_tensors():
    image = Image.new("RGB", (640, 480))
    x, t = eval_transform(image, target())
    assert x.shape == (3, 800, 1066) and t["orig_size"] == (480, 640)
    x, t = train_transform(image, target())
    assert x.dim() == 3 and 480 <= min(x.shape[1:]) <= 800 and max(x.shape[1:]) <= 1333
