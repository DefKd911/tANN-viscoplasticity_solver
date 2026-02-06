#!/usr/bin/env python3
"""
Advanced training script: better than paper (Khorrami et al. 2023).

Target: Lower val/test MAE than paper (1.743 MPa) using:
- Residual U-Net (ResUNet-style) for better gradient flow and accuracy
- Larger capacity (base 48), optional 9×9 first layer
- AdamW + cosine LR or ReduceLROnPlateau + warmup
- Gradient clipping, weight decay
- Optional boundary-weighted loss (if mask provided)
- Early stopping with high patience (e.g. 50)

Usage:
  python train_advanced.py --data ML_DATASET --out ML_CHECKPOINTS/advanced
  python train_advanced.py --data ML_DATASET --epochs 500 --base 48 --lr 0.001 --scheduler cosine
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

# --------------------------- Dataset ---------------------------
class NpyPairDataset(Dataset):
    def __init__(self, root_split_dir: str):
        self.in_paths = sorted(glob.glob(os.path.join(root_split_dir, 'inputs', 'sample_*.npy')))
        self.out_paths = [p.replace(os.path.sep + 'inputs' + os.path.sep, os.path.sep + 'outputs' + os.path.sep) for p in self.in_paths]
        assert len(self.in_paths) == len(self.out_paths) and len(self.in_paths) > 0

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        X = np.load(self.in_paths[idx]).astype(np.float32)
        Y = np.load(self.out_paths[idx]).astype(np.float32)
        return torch.from_numpy(X).permute(2, 0, 1), torch.from_numpy(Y).permute(2, 0, 1)


# --------------------------- Residual block (better gradient flow) ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(ch, ch, k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResDoubleConv(nn.Module):
    """Conv down to out_ch then two residual blocks (or one conv + one res)."""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            ResidualBlock(out_ch, k),
            ResidualBlock(out_ch, k),
        )

    def forward(self, x):
        x = self.in_conv(x)
        return self.res(x)


# --------------------------- Advanced U-Net (ResUNet-style, wider) ---------------------------
class AdvancedUNet(nn.Module):
    """
    Residual U-Net: residual blocks in encoder/decoder, bilinear up, skip connections.
    base=48 for more capacity than paper (32). First layer can be 9×9 to match paper receptive field.
    """
    def __init__(self, in_ch=5, out_ch=1, base=48, first_kernel=3):
        super().__init__()
        p1 = first_kernel // 2
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, first_kernel, padding=p1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            ResidualBlock(base, 3),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResDoubleConv(base, base * 2, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResDoubleConv(base * 2, base * 4, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResDoubleConv(base * 4, base * 8, 3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ResDoubleConv(base * 8 + base * 4, base * 4, 3)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ResDoubleConv(base * 4 + base * 2, base * 2, 3)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ResDoubleConv(base * 2 + base, base, 3)
        self.outc = nn.Conv2d(base, out_ch, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
        return self.outc(u1)


# --------------------------- Training with scheduler, grad clip, AdamW ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        total += float(nn.functional.l1_loss(model(X), Y).item()) * X.size(0)
        n += X.size(0)
    return total / max(1, n)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("[ADVANCED] ResUNet + AdamW + LR schedule + grad clip (target: better than paper)")
    print(f"  Device: {device}, base: {args.base}, epochs: {args.epochs}, lr: {args.lr}")
    print(f"  Scheduler: {args.scheduler}, warmup: {args.warmup_epochs}, grad_clip: {args.grad_clip}")

    train_ds = NpyPairDataset(os.path.join(args.data, 'train'))
    val_ds = NpyPairDataset(os.path.join(args.data, 'val'))
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = AdvancedUNet(in_ch=5, out_ch=1, base=args.base, first_kernel=args.first_kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    loss_fn = nn.L1Loss()

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6)
    else:
        scheduler = None

    best_val = float('inf')
    wait = 0
    train_hist: List[float] = []
    val_hist: List[float] = []
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
            for g in optimizer.param_groups:
                g['lr'] = args.lr * (epoch / args.warmup_epochs)

        model.train()
        run_loss, seen = 0.0, 0
        for X, Y in train_ld:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            run_loss += float(loss.item()) * X.size(0)
            seen += X.size(0)
        tr_mae = run_loss / max(1, seen)
        val_mae = evaluate(model, val_ld, device)
        train_hist.append(tr_mae)
        val_hist.append(val_mae)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_mae)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} | train MAE {tr_mae:.6f} | val MAE {val_mae:.6f} | lr {current_lr:.2e}")

        if val_mae < best_val:
            best_val = val_mae
            wait = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_mae': best_val}, os.path.join(args.out, 'best.pt'))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {args.patience}).")
                break

    torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_mae': best_val}, os.path.join(args.out, 'last.pt'))
    with open(os.path.join(args.out, 'training_log.csv'), 'w') as f:
        f.write('epoch,train_mae,val_mae\n')
        for i, (tr, va) in enumerate(zip(train_hist, val_hist), start=1):
            f.write(f"{i},{tr:.8f},{va:.8f}\n")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(range(1, len(train_hist) + 1), train_hist, label='train MAE')
        plt.plot(range(1, len(val_hist) + 1), val_hist, label='val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Advanced (ResUNet + schedule)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'training_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: plot failed: {e}")
    print(f"Done. Best val MAE: {best_val:.6f} (paper val: 1.743 MPa)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Advanced training: better than paper.')
    ap.add_argument('--data', default='ML_DATASET')
    ap.add_argument('--out', default='ML_CHECKPOINTS/advanced')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--wd', type=float, default=1e-4, help='AdamW weight decay')
    ap.add_argument('--base', type=int, default=48, help='Base channels (paper 32)')
    ap.add_argument('--first-kernel', type=int, default=3, help='First conv kernel (9 for paper-like receptive field)')
    ap.add_argument('--scheduler', choices=['cosine', 'plateau', 'none'], default='cosine')
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--grad-clip', type=float, default=1.0, help='Max grad norm (0=off)')
    ap.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args()
    train(args)
