"""Score a UPT checkpoint on the HICO-DET or V-COCO test split.

    python evaluate.py hicodet weights/upt-r50-hicodet.pt
    python evaluate.py vcoco weights/upt-r50-vcoco.pt
"""
import argparse
import os
import sys

import torch

from upt.datasets import HOIDataset
from upt.evaluation import dump_vcoco, hico_map
from upt.model import UPT, object_to_target
from upt.transforms import eval_transform

TEST_SPLIT = {"hicodet": "test2015", "vcoco": "test"}
VCOCO_EVAL = "data/v-coco"  # clone of github.com/s-gupta/v-coco, see download.ps1


def load_model(name, checkpoint, meta, num_classes, device):
    model = UPT(num_classes, object_to_target(meta, name))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=TEST_SPLIT)
    parser.add_argument("checkpoint")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = HOIDataset(args.dataset, TEST_SPLIT[args.dataset], transforms=eval_transform)
    model = load_model(args.dataset, args.checkpoint, dataset.meta, dataset.num_classes, device)

    if args.dataset == "hicodet":
        result = hico_map(model, dataset, device, args.workers)
        print("HICO-DET mAP  full {full:.4f}  rare {rare:.4f}  non-rare {non_rare:.4f}".format(**result))
        return

    os.makedirs("runs", exist_ok=True)
    predictions = os.path.abspath(f"runs/vcoco-{os.path.basename(args.checkpoint)}.pkl")
    dump_vcoco(model, dataset, device, predictions, args.workers)
    # Gupta's evaluator prints AP_role for scenario 1 and 2. Its average includes the class
    # point-instr, which has no valid pairs: multiply by 25/24 for the reported number.
    sys.path.insert(0, VCOCO_EVAL)
    from vsrl_eval import VCOCOeval

    data = os.path.join(VCOCO_EVAL, "data")
    evaluator = VCOCOeval(f"{data}/vcoco/vcoco_test.json", f"{data}/instances_vcoco_all_2014.json", f"{data}/splits/vcoco_test.ids")
    evaluator._do_eval(predictions, ovr_thresh=0.5)


if __name__ == "__main__":
    main()
