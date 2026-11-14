import os

import pytest

from upt.datasets import HOIDataset

# Counts of images with annotated interactions, as loaded by the UPT authors' code
COUNTS = {
    ("hicodet", "train2015"): 37633,
    ("hicodet", "test2015"): 9546,
    ("vcoco", "trainval"): 4969,
    ("vcoco", "test"): 4532,
}


@pytest.mark.parametrize("name,partition", COUNTS)
def test_counts_and_targets(name, partition):
    ds = HOIDataset(name, partition)
    assert len(ds) == COUNTS[(name, partition)]
    for i in (0, len(ds) // 2, len(ds) - 1):
        t = ds.target(i)
        n = len(t["labels"])
        assert n > 0
        assert t["boxes_h"].shape == t["boxes_o"].shape == (n, 4)
        assert t["object"].shape == (n,)
        assert (t["boxes_h"][:, 2:] > t["boxes_h"][:, :2]).all()
        assert 0 <= t["labels"].min() and t["labels"].max() < ds.num_classes


@pytest.mark.parametrize("name,partition", COUNTS)
def test_every_image_exists(name, partition):
    ds = HOIDataset(name, partition)
    if not os.path.isdir(ds.image_dir):
        pytest.skip("images not downloaded")
    missing = [f for f, _ in ds.items if not os.path.exists(os.path.join(ds.image_dir, f))]
    assert not missing, missing[:5]
    image, target = ds[0]
    assert target["boxes_h"][:, 2].max() <= image.width
