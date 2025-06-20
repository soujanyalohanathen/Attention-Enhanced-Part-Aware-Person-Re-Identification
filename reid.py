import os
import argparse
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import models, transforms

# ------------------------------
# 0) Settings & Argument Parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Attention‐Enhanced Part‐Aware Re‐ID")
    parser.add_argument('--data_root', default="market/", help='Market-1501 data root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3.5e-5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--use_ca',  action='store_true', help='enable channel attention')
    parser.add_argument('--use_sa',  action='store_true', help='enable spatial attention')
    parser.add_argument('--use_pp',  action='store_true', help='enable part pooling')
    parser.add_argument('--use_tr',  action='store_true', help='enable Transformer encoding')
    parser.add_argument('--use_tri', action='store_true', help='enable batch-hard triplet loss')
    parser.add_argument('--re_ranking', action='store_true', help='apply re-ranking at eval')
    return parser.parse_args()

args = parse_args()

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

# ------------------------------
# 1) Dataset definitions
# ------------------------------
class Market1501Dataset(Dataset):
    def __init__(self, root, mode='train', transform=None, pid2label=None):
        self.transform = transform
        folder = {'train':'bounding_box_train','query':'query','gallery':'bounding_box_test'}[mode]
        img_dir = os.path.join(root, folder)

        samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith('.jpg'): continue
            raw_pid, cam, _ = fname.split('_',2)
            samples.append((os.path.join(img_dir, fname),
                            int(raw_pid), int(cam[1])))

        if pid2label is None:
            raw_pids = sorted({pid for _,pid,_ in samples})
            self.pid2label = {pid:idx for idx,pid in enumerate(raw_pids)}
        else:
            self.pid2label = pid2label

        self.samples = [
            (path, self.pid2label[pid], camid)
            for path,pid,camid in samples
            if pid in self.pid2label
        ]

    def __getitem__(self, idx):
        path,pid,camid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.samples)

class Market1501EvalDataset(Dataset):
    def __init__(self, root, mode='query', transform=None):
        self.transform = transform
        folder = {'query':'query','gallery':'bounding_box_test'}[mode]
        img_dir = os.path.join(root, folder)
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith('.jpg'): continue
            raw_pid, cam, _ = fname.split('_',2)
            self.samples.append((os.path.join(img_dir,fname),
                                  int(raw_pid), int(cam[1])))

    def __getitem__(self, idx):
        path,pid,cam = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, pid, cam

    def __len__(self):
        return len(self.samples)

# ------------------------------
# 2) Attention & PartTransformer
# ------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels//ratio, bias=False)
        self.relu= nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//ratio, channels, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        b,c,_,_ = x.shape
        y = F.adaptive_avg_pool2d(x,1).view(b,c)
        y = self.relu(self.fc1(y))
        y = self.sig(self.fc2(y)).view(b,c,1,1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size-1)//2
        self.conv = nn.Conv2d(2,1,kernel_size,padding=pad,bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self,x):
        avg = x.mean(1,keepdim=True)
        mx, _ = x.max(1,keepdim=True)
        y = torch.cat([avg,mx],1)
        y = self.sig(self.conv(y))
        return x * y.expand_as(x)

class PartTransformer(nn.Module):
    def __init__(self, num_parts, feat_dim, nhead=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=feat_dim,
                                           nhead=nhead,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self,x):
        return self.encoder(x)

# ------------------------------
# 3) Model
# ------------------------------
class ReIDModel(nn.Module):
    def __init__(self, num_parts, feat_dim, num_classes,
                 use_ca, use_sa, use_pp, use_tr):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        self.use_ca = use_ca
        self.use_sa = use_sa
        self.use_pp = use_pp
        self.use_tr = use_tr

        # determine number of output parts
        self.n_parts = num_parts if use_pp else 1

        # optional attention
        if use_ca:
            self.ca = ChannelAttention(feat_dim)
        if use_sa:
            self.sa = SpatialAttention()

        # pooling and transformer instantiated for original num_parts,
        # but actual parts used = self.n_parts
        self.pool = nn.AdaptiveAvgPool2d((num_parts,1))
        self.pt   = PartTransformer(num_parts, feat_dim)

        # final BN and FC match feat_dim * self.n_parts
        self.bn = nn.BatchNorm1d(feat_dim * self.n_parts)
        self.fc = nn.Linear(feat_dim * self.n_parts, num_classes)

    def forward(self, x, return_feats=False):
        f = self.backbone(x)

        if self.use_ca:
            f = self.ca(f)
        if self.use_sa:
            f = self.sa(f)

        # pooling
        if self.use_pp:
            # produce [B, P, C]
            p = self.pool(f).squeeze(-1).permute(0,2,1)
        else:
            # global pool as single part [B,1,C]
            gp = F.adaptive_avg_pool2d(f,1).view(f.size(0),1,-1)
            p = gp

        # transformer
        if self.use_tr:
            p = self.pt(p)

        # flatten to [B, n_parts*C]
        feat = p.reshape(x.size(0), -1)
        feat = self.bn(feat)

        if return_feats:
            return feat

        logits = self.fc(feat)
        return logits, feat

# ------------------------------
# 4) Triplet Loss
# ------------------------------
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, feats, labels):
        dist = torch.cdist(feats, feats, p=2)
        mask_pos = labels.unsqueeze(1)==labels.unsqueeze(0)
        mask_neg = ~mask_pos
        hardest_pos = dist.masked_fill(~mask_pos, -1).max(1)[0]
        hardest_neg = dist.masked_fill(~mask_neg, 1e6).min(1)[0]
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()

# ------------------------------
# 5) Train/Eval Routines
# ------------------------------
def train_epoch(model, loader, optimizer, criterion_ce, criterion_tri, scaler, device):
    model.train()
    total_loss, total_corr, total = 0,0,0
    for imgs, pids, _ in loader:
        imgs,pids = imgs.to(device), pids.to(device)
        optimizer.zero_grad()
        with autocast():
            logits, feats = model(imgs)
            loss_ce = criterion_ce(logits, pids)
            if args.use_tri:
                loss_tri = criterion_tri(feats, pids)
                loss = loss_ce + loss_tri
            else:
                loss = loss_ce
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1)
        total_corr += (preds==pids).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return total_loss/total, total_corr/total

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, camids = [],[],[]
    for imgs, raw_p, raw_c in loader:
        imgs = imgs.to(device)
        f = model(imgs, return_feats=True)
        feats.append(f.cpu())
        pids.extend(raw_p)
        camids.extend(raw_c)
    feats = torch.cat(feats, 0)
    return feats, np.array(pids), np.array(camids)

def compute_distance_matrix(qf, gf):
    q2 = np.sum(qf**2, axis=1, keepdims=True)
    g2 = np.sum(gf**2, axis=1, keepdims=True)
    dist = q2 + g2.T - 2*(qf @ gf.T)
    return np.sqrt(np.maximum(dist,0))

@torch.no_grad()
def evaluate(model, q_loader, g_loader, device):
    qf, qp, qc = extract_features(model, q_loader, device)
    gf, gp, gc = extract_features(model, g_loader, device)
    qf_np, gf_np = qf.numpy(), gf.numpy()
    distmat = compute_distance_matrix(qf_np, gf_np)

    idx     = np.argsort(distmat, axis=1)
    matches = (gp[idx] == qp[:,None])
    all_AP, rank1, valid = [], 0, 0
    for i in range(distmat.shape[0]):
        keep = (gp[idx[i]]!=qp[i]) | (gc[idx[i]]!=qc[i])
        y = matches[i][keep]
        if not y.any(): continue
        valid += 1
        if y[0]: rank1 += 1
        prec = (y.cumsum()/(1+np.arange(len(y)))) * y
        all_AP.append(prec.sum()/y.sum())
    mAP = np.mean(all_AP) if valid else 0.0
    r1  = rank1/valid     if valid else 0.0
    return mAP, r1

# ------------------------------
# 6) Main
# ------------------------------
def main():
    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.1,0.1,0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02,0.2), ratio=(0.3,3.3), value='random'),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
    ])

    # Data loaders
    train_ds = Market1501Dataset(args.data_root, 'train', train_tf, pid2label=None)
    pid2label = train_ds.pid2label
    query_ds   = Market1501EvalDataset(args.data_root, 'query',   test_tf)
    gallery_ds = Market1501EvalDataset(args.data_root, 'gallery', test_tf)

    train_loader   = DataLoader(train_ds,   batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    query_loader   = DataLoader(query_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model, losses, optimizer, scheduler, scaler
    model = ReIDModel(
        num_parts=6,
        feat_dim=2048,
        num_classes=len(pid2label),
        use_ca=args.use_ca,
        use_sa=args.use_sa,
        use_pp=args.use_pp,
        use_tr=args.use_tr
    ).to(DEVICE)

    criterion_ce  = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_tri = BatchHardTripletLoss(margin=0.3)
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler        = GradScaler()

    warmup    = LinearLR(optimizer, start_factor=1e-3, total_iters=2)
    cosine    = CosineAnnealingLR(optimizer, T_max=args.epochs-2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[2])

    # Training + ablation logging
    best_map, best_r1 = 0.0, 0.0
    for epoch in range(1, args.epochs+1):
        tloss, tacc = train_epoch(model, train_loader, optimizer,
                                  criterion_ce, criterion_tri,
                                  scaler, DEVICE)
        mAP, r1 = evaluate(model, query_loader, gallery_loader, DEVICE)
        print(f"Epoch {epoch:02d} | Train Loss: {tloss:.4f} | Train Acc: {tacc:.4f} | mAP: {mAP:.4f} | Rank-1: {r1:.4f}")
        if mAP > best_map:
            best_map, best_r1 = mAP, r1
            torch.save(model.state_dict(), "best_model.pth")
        scheduler.step()

    print(f"Training complete. Best mAP: {best_map:.4f}, Rank-1: {best_r1:.4f}")

    # Determine variant name
    variant = []
    if args.use_ca:  variant.append("CA")
    if args.use_sa:  variant.append("SA")
    if args.use_pp:  variant.append("PP")
    if args.use_tr:  variant.append("TR")
    if args.use_tri: variant.append("Tri")
    if args.re_ranking: variant.append("RR")
    variant_name = "+".join(variant) if variant else "Base"

    # Write ablation result
    csv_path = "ablation_results.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant", "mAP", "Rank1"])
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([variant_name, f"{best_map:.4f}", f"{best_r1:.4f}"])

if __name__ == "__main__":
    main()
