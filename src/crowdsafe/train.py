"""Train the density network on a range of Mall frames.

    python -m crowdsafe.train --frames 0:800 --out weights/mall_density.pt

Fixed epoch budget and the last checkpoint is kept, so no held-out data is
used for model selection and the test split stays untouched.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .density import TARGET_SCALE, DensityNet, gt_density, to_tensor
from .mall import load_mall


def parse_range(s: str) -> range:
    a, b = s.split(":")
    return range(int(a), int(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mall_dataset")
    ap.add_argument("--frames", default="0:800", help="start:end frame indices to train on")
    ap.add_argument("--out", default="weights/mall_density.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--amp", action="store_true", help="mixed precision (slower on GTX 16xx cards, faster on RTX)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_mall(args.data)
    idx = list(parse_range(args.frames))

    print(f"loading {len(idx)} frames into memory...")
    imgs, dens = [], []
    for i in idx:
        f = ds.frame(i)
        imgs.append(to_tensor(f))
        dens.append(torch.from_numpy(gt_density(ds.points[i], *f.shape[:2])))
    imgs = torch.stack(imgs)
    dens = torch.stack(dens).unsqueeze(1) * TARGET_SCALE

    model = DensityNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(idx))
        loss_sum, abs_err = 0.0, 0.0
        for b in range(0, len(idx), args.batch):
            sel = perm[b : b + args.batch]
            x, y = imgs[sel].to(device), dens[sel].to(device)
            if torch.rand(1).item() < 0.5:  # horizontal flip
                x, y = x.flip(-1), y.flip(-1)
            x = x * (1 + 0.2 * (torch.rand(len(sel), 1, 1, 1, device=device) - 0.5))  # brightness
            with torch.autocast(device, enabled=use_amp):
                pred = model(x)
            loss = F.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(sel)
            abs_err += (pred.float().sum((1, 2, 3)) - y.sum((1, 2, 3))).abs().sum().item() / TARGET_SCALE
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch + 1:3d}/{args.epochs}  loss {loss_sum / len(idx):.4f}  "
                f"train MAE {abs_err / len(idx):.2f}  {time.time() - t0:.0f}s",
                flush=True,
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
