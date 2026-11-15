import os

import pytest
import torch

from upt.datasets import HOIDataset
from upt.layers import pair_spatial_features
from upt.model import UPT, object_to_target, to_detections

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARTITION = {"hicodet": "train2015", "vcoco": "trainval"}


def test_spatial_features():
    a = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 20.0]])
    f = pair_spatial_features(a, b, 100, 200)
    assert f.shape == (1, 36) and f.isfinite().all()
    assert f[0, 13] == pytest.approx(1 / 3)  # IoU
    assert f[0, 14] == pytest.approx(0.5) and f[0, 15] == 0  # b is half a box-width to the right of a


def test_object_to_target():
    hico = object_to_target(HOIDataset("hicodet", "train2015").meta, "hicodet")
    assert len(hico) == 80 and sum(len(v) for v in hico) == 600
    vcoco = object_to_target(HOIDataset("vcoco", "trainval").meta, "vcoco")
    assert len(vcoco) == 80 and all(0 <= a < 24 for v in vcoco for a in v)


@pytest.fixture(scope="module", params=["hicodet", "vcoco"])
def model(request):
    name = request.param
    path = f"weights/upt-r50-{name}.pt"
    if not os.path.exists(path):
        pytest.skip("weights not downloaded")
    ds = HOIDataset(name, PARTITION[name])
    net = UPT(ds.num_classes, object_to_target(ds.meta, name))
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    net.load_state_dict(state, strict=True)  # every released weight has a home, and nothing is left untrained
    return net.to(DEVICE)


def test_inference(model):
    model.eval()
    images = [torch.randn(3, 480, 640, device=DEVICE), torch.randn(3, 512, 384, device=DEVICE)]
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 2
    for out in outputs:
        m = len(out["h_idx"])
        assert m > 0 and out["logits"].shape == out["prior"].shape == (m, model.interaction_head.valid.shape[1])
        assert (out["h_idx"] != out["o_idx"]).all()
        det = to_detections(out)
        assert ((det["scores"] >= 0) & (det["scores"] <= 1)).all()
        assert det["boxes_h"].shape == det["boxes_o"].shape == (len(det["scores"]), 4)


def test_only_the_head_trains(model):
    model.train()
    out = model([torch.randn(3, 480, 640, device=DEVICE)])[0]
    out["logits"].sum().backward()
    assert all(p.grad is None for p in model.detector.parameters())
    assert all(p.grad is not None for p in model.interaction_head.parameters())
    model.zero_grad()
