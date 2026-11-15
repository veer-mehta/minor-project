"""Train the UPT interaction head on HICO-DET or V-COCO with a frozen detector.

    python train.py vcoco
    python train.py hicodet            # re-run the same command to resume after an interruption

The paper trains with 16 images per step across 8 GPUs. Here each step accumulates
gradients over batches of --micro-batch images until --batch-size images are seen, and the
loss is divided by the positives of the whole step, which gives the same update on one GPU.
"""
import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from upt.datasets import HOIDataset, collate
from upt.loss import interaction_loss
from upt.model import UPT, object_to_target
from upt.transforms import train_transform

TRAIN_SPLIT = {"hicodet": "train2015", "vcoco": "trainval"}


def save(path, **state):
    """Write to a temporary file first, so an interruption never leaves a broken checkpoint."""
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=TRAIN_SPLIT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-drop", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=250, help="steps between checkpoints")
    parser.add_argument("--limit", type=int, help="train on the first N images only (for checks)")
    parser.add_argument("--out", help="output folder, default runs/<dataset>")
    args = parser.parse_args()
    assert args.batch_size % args.micro_batch == 0
    accumulate = args.batch_size // args.micro_batch

    out = args.out or f"runs/{args.dataset}"
    os.makedirs(out, exist_ok=True)
    checkpoint_path = f"{out}/checkpoint.pt"
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = HOIDataset(args.dataset, TRAIN_SPLIT[args.dataset], transforms=train_transform)
    size = min(args.limit or len(dataset), len(dataset))
    steps_per_epoch = size // args.batch_size

    model = UPT(dataset.num_classes, object_to_target(dataset.meta, args.dataset))
    detector = torch.load(f"weights/detr-r50-{args.dataset}.pth", map_location="cpu", weights_only=True)
    model.detector.load_state_dict(detector["model_state_dict"])
    model.to(device)
    params = list(model.interaction_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    epoch = step = 0  # step counts optimiser steps within the current epoch
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, step = state["epoch"], state["step"]
        print(f"resuming from epoch {epoch + 1}, step {step}")

    log = open(f"{out}/log.txt", "a")
    while epoch < args.epochs:
        model.train()
        # The image order of an epoch depends only on the seed and the epoch, so a resumed
        # run continues through the same order from where it stopped
        order = torch.randperm(size, generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loader = DataLoader(
            dataset, batch_size=args.micro_batch, sampler=order[step * args.batch_size:],
            collate_fn=collate, num_workers=args.workers, drop_last=True,
        )
        loss_sum = positives = 0
        running, seen, start = 0.0, 0, time.time()
        for i, (images, targets) in enumerate(loader, 1):
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            loss, n = interaction_loss(model([im.to(device) for im in images]), targets)
            if loss.isnan():
                raise ValueError("loss is NaN")
            if loss.requires_grad:  # false when no image in the batch produced a pair
                loss.backward()
            loss_sum += loss.item()
            positives += n
            if i % accumulate:
                continue

            # A step with no positive pair would divide by zero; fall back to 1
            positives = max(positives, 1)
            for p in params:
                if p.grad is not None:
                    p.grad /= positives
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running += loss_sum / positives
            seen += 1
            loss_sum = positives = 0

            if step % 50 == 0 or step == steps_per_epoch:
                line = (f"epoch {epoch + 1}/{args.epochs} step {step}/{steps_per_epoch} loss {running / seen:.4f} "
                        f"lr {optimizer.param_groups[0]['lr']:.0e} {seen * args.batch_size / (time.time() - start):.1f} img/s")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                running, seen, start = 0.0, 0, time.time()
            if step % args.save_every == 0 and step < steps_per_epoch:
                save(checkpoint_path, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, step=step)

        scheduler.step()
        epoch, step = epoch + 1, 0
        save(checkpoint_path, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
             scheduler=scheduler.state_dict(), epoch=epoch, step=step)


if __name__ == "__main__":
    main()
