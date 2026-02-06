#!/usr/bin/env python3
"""
Train U-Net to replicate Khorrami et al. (npj Comput. Mater. 2023) setup.

Paper: "An artificial neural network for surrogate modeling of stress fields
in viscoplastic polycrystalline materials"

Exact paper settings (same data preparation and training strategy as paper):
- Data: 10-grain microstructures only; 64×64; E, ν, ξ0, h0, σvM normalized to 1 (Table 3 ranges).
  Build with: build_ml_dataset.py --train 800 --val 200 --max-seeds 1000 --max-grains 10
- Architecture: U-Net, 32 filters, 9×9 separable 2D convolution (depthwise 9×9 + pointwise 1×1),
  batch norm, ReLU, 2D max pooling, bilinear upsampling. Glorot (Xavier) init.
- Optimizer: Adam lr=0.001, momentum 0.9 (beta1=0.9).
- Loss: MAE (L1).
- Training: 500 epochs; 80% train / 20% val. No early stopping. Paper: train MAE 1.733 MPa, val 1.743 MPa.
- MAE in MPa = val_mae (normalized) × 1000.

Usage:
  python train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication
  python train_paper_replication.py --data ML_DATASET --epochs 500 --lr 0.001 --batch-size 8
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# --------------------------- Dataset (same as baseline) ---------------------------
class NpyPairDataset(Dataset):
    def __init__(self, root_split_dir: str):
        self.in_paths = sorted(glob.glob(os.path.join(root_split_dir, 'inputs', 'sample_*.npy')))
        self.out_paths = [p.replace(os.path.sep + 'inputs' + os.path.sep, os.path.sep + 'outputs' + os.path.sep) for p in self.in_paths]
        assert len(self.in_paths) == len(self.out_paths) and len(self.in_paths) > 0, f"No samples in {root_split_dir}"

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        X = np.load(self.in_paths[idx]).astype(np.float32)
        Y = np.load(self.out_paths[idx]).astype(np.float32)
        X = torch.from_numpy(X).permute(2, 0, 1)
        Y = torch.from_numpy(Y).permute(2, 0, 1)
        return X, Y


# --------------------------- Paper U-Net: 9×9 separable conv, 32 filters, Glorot ---------------------------
def _glorot_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PaperSepBlock(nn.Module):
    """Two separable 9×9 blocks as in paper (Fig. 12): depthwise 9×9 + pointwise 1×1, BN, ReLU."""
    def __init__(self, in_ch, out_ch, k=9):
        super().__init__()
        p = k // 2
        # First separable: depthwise 9×9 then pointwise 1×1
        self.dw1 = nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.dw2 = nn.Conv2d(out_ch, out_ch, k, padding=p, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.pw1(self.dw1(x))))
        x = self.relu2(self.bn2(self.pw2(self.dw2(x))))
        return x


class PaperUNet(nn.Module):
    """
    U-Net matching paper (Fig. 12): 32 filters, 9×9 separable 2D convolution,
    batch norm, max pooling, bilinear upsampling. Input 5 ch, output 1 ch, 64×64.
    """
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        # Encoder: separable 9×9 blocks (paper: "separable 2D convolution with 9×9 kernel")
        self.down1 = PaperSepBlock(in_ch, base, k=9)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = PaperSepBlock(base, base * 2, k=9)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = PaperSepBlock(base * 2, base * 4, k=9)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = PaperSepBlock(base * 4, base * 8, k=9)
        # Decoder: bilinear upsampling (paper)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = PaperSepBlock(base * 8 + base * 4, base * 4, k=9)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = PaperSepBlock(base * 4 + base * 2, base * 2, k=9)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = PaperSepBlock(base * 2 + base, base, k=9)
        self.outc = nn.Conv2d(base, out_ch, 1)
        self.apply(_glorot_init)

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


# --------------------------- Training (paper: Adam 0.001, MAE, 500 epochs) ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        total += float(torch.nn.functional.l1_loss(pred, Y).item()) * X.size(0)
        count += X.size(0)
    return total / max(1, count)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("[PAPER REPLICATION] Khorrami et al. npj Comput. Mater. 2023")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, lr: {args.lr}, batch_size: {args.batch_size}, MAE loss")
    print(f"  Architecture: U-Net 32 filters, 9×9 separable conv, Glorot init, bilinear upsampling")

    train_ds = NpyPairDataset(os.path.join(args.data, 'train'))
    val_ds = NpyPairDataset(os.path.join(args.data, 'val'))
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = PaperUNet(in_ch=5, out_ch=1, base=32).to(device)
    # Paper: Adam lr=0.001, momentum 0.9 (beta1=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.L1Loss()

    best_val = float('inf')
    train_hist: List[float] = []
    val_hist: List[float] = []
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for X, Y in train_ld:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            run_loss += float(loss.item()) * X.size(0)
            seen += X.size(0)
        tr_mae = run_loss / max(1, seen)
        val_mae = evaluate(model, val_ld, device)
        train_hist.append(tr_mae)
        val_hist.append(val_mae)
        print(f"Epoch {epoch:04d} | train MAE {tr_mae:.6f} | val MAE {val_mae:.6f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_mae': best_val}, os.path.join(args.out, 'best.pt'))

    torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'val_mae': best_val}, os.path.join(args.out, 'last.pt'))

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
        plt.ylabel('MAE (normalized)')
        plt.legend()
        plt.title('Paper replication (Khorrami et al. 2023)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'training_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: plot failed: {e}")

    print(f"Done. Best val MAE: {best_val:.6f} (×1000 = {best_val*1000:.2f} MPa; paper val 1.743 MPa)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train U-Net with exact paper (Khorrami et al. 2023) settings.')
    ap.add_argument('--data', default='ML_DATASET', help='Dataset root (train/val)')
    ap.add_argument('--out', default='ML_CHECKPOINTS/paper_replication', help='Checkpoint dir')
    ap.add_argument('--epochs', type=int, default=500, help='Paper: 500')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.001, help='Paper: 0.001')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args()
    train(args)
