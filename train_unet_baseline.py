#!/usr/bin/env python3
"""
Train a baseline U-Net on ML_DATASET.
Inputs: (64,64,5) normalized; Outputs: (64,64,1) normalized.
- Loss: L1 (MAE)
- Optim: Adam
- Early stopping on val MAE

Usage:
  python train_unet_baseline.py --data ML_DATASET --epochs 50 --batch-size 8 --lr 1e-4
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# ---------------- Dataset ----------------
class NpyPairDataset(Dataset):
    def __init__(self, root_split_dir: str):
        self.in_paths = sorted(glob.glob(os.path.join(root_split_dir, 'inputs', 'sample_*.npy')))
        self.out_paths = [p.replace(os.path.sep + 'inputs' + os.path.sep, os.path.sep + 'outputs' + os.path.sep) for p in self.in_paths]
        assert len(self.in_paths) == len(self.out_paths) and len(self.in_paths) > 0, f"No samples in {root_split_dir}"
    def __len__(self):
        return len(self.in_paths)
    def __getitem__(self, idx):
        X = np.load(self.in_paths[idx]).astype(np.float32)  # (H,W,5)
        Y = np.load(self.out_paths[idx]).astype(np.float32) # (H,W,1)
        # to CHW tensors
        X = torch.from_numpy(X).permute(2,0,1)  # (5,H,W)
        Y = torch.from_numpy(Y).permute(2,0,1)  # (1,H,W)
        return X, Y

# ---------------- Model (U-Net small) ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k//2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = DoubleConv(base*8 + base*4, base*4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = DoubleConv(base*4 + base*2, base*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = DoubleConv(base*2 + base, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bn = self.bottleneck(self.pool3(d3))
        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        out = self.outc(u1)
        return out

# ---------------- Training ----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.L1Loss()
    total, count = 0.0, 0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = loss_fn(pred, Y)
        total += float(loss.item()) * X.size(0)
        count += X.size(0)
    return total / max(1, count)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    train_ds = NpyPairDataset(os.path.join(args.data, 'train'))
    val_ds   = NpyPairDataset(os.path.join(args.data, 'val'))

    pin_memory = args.pin_memory or (device.type == 'cuda')
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = UNet(in_ch=5, out_ch=1, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.L1Loss()

    best_val = float('inf')
    patience = args.patience
    wait = 0
    train_hist: List[float] = []
    val_hist: List[float] = []
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss, seen = 0.0, 0
        for X, Y in train_ld:
            X = X.to(device, non_blocking=pin_memory)
            Y = Y.to(device, non_blocking=pin_memory)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(X)
                    loss = loss_fn(pred, Y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X)
                loss = loss_fn(pred, Y)
                loss.backward()
                opt.step()
            run_loss += float(loss.item()) * X.size(0)
            seen += X.size(0)
        tr_loss = run_loss / max(1, seen)
        val_loss = evaluate(model, val_ld, device)
        print(f"Epoch {epoch:03d} | train MAE {tr_loss:.6f} | val MAE {val_loss:.6f}")
        train_hist.append(tr_loss)
        val_hist.append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model': model.state_dict(), 'val_mae': best_val}, os.path.join(args.out, 'best.pt'))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # final save
    os.makedirs(args.out, exist_ok=True)
    torch.save({'model': model.state_dict(), 'val_mae': best_val}, os.path.join(args.out, 'last.pt'))

    # write training log (CSV)
    log_csv = os.path.join(args.out, 'training_log.csv')
    with open(log_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_mae,val_mae\n')
        for i, (tr, va) in enumerate(zip(train_hist, val_hist), start=1):
            f.write(f"{i},{tr:.8f},{va:.8f}\n")

    # try to save a training curve plot
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6,4), dpi=150)
        plt.plot(range(1, len(train_hist)+1), train_hist, label='train MAE')
        plt.plot(range(1, len(val_hist)+1), val_hist, label='val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'training_curve.png'))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to save training curve: {e}")

    print(f"Done. Best val MAE: {best_val:.6f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ML_DATASET')
    ap.add_argument('--out', default='ML_CHECKPOINTS')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (0=main thread only; 2–4 often faster on GPU)')
    ap.add_argument('--pin-memory', action='store_true', help='Pin memory for faster GPU transfer (recommended with CUDA)')
    ap.add_argument('--amp', action='store_true', help='Use automatic mixed precision (faster, less VRAM; RTX 4060 recommended)')
    args = ap.parse_args()
    train(args)
