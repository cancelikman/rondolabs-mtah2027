"""Round 12: set/graph model over the ball and every player (attention pooling per team + temporal conv).

Inputs per frame (tracking only):
  ball   : visible, x/L, y/W, vx, vy, speed, seconds since last seen, inside pitch
  player : position relative to the (last seen) ball, distance, velocity, speed relative to the
           ball, pitch position; one set per team, attention-pooled (permutation invariant)
Window: 21 steps of 0.2 s (+-2 s) around the frame; dilated temporal convolutions; output H/A/D.
Home/away symmetry: random team swap in training, average with the swapped input at inference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PF, BF = 10, 8           # player / ball feature counts
STEP, HALF = 5, 10       # window: 2*HALF+1 steps of STEP frames (25 Hz -> 0.2 s, +-2 s)


def _ffill(x, ok):
    idx = np.where(ok, np.arange(len(ok)), 0)
    idx = np.maximum.accumulate(idx)
    out = x[idx].copy()
    first = np.argmax(ok) if ok.any() else len(ok)
    out[:first] = np.nan
    return out, idx


def frame_features(geo):
    """(ball [n, BF], players [n, k, PF], team [k]) as float16, NaN-free (missing -> 0 with flags)."""
    ball = np.asarray(geo["ball"], float)
    P = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"]).astype(int)
    fps = float(geo["fps"])
    L, W = [float(v) for v in np.asarray(geo["pitch"]).ravel()[:2]]
    n, k = P.shape[:2]
    vis = np.isfinite(ball).all(1)
    bf_pos, idx = _ffill(ball, vis)
    since = (np.arange(n) - idx) / fps
    since[~np.isfinite(bf_pos).all(1)] = 99.
    lag = STEP
    dt = lag / fps
    bv = np.zeros_like(ball)
    bv[lag:] = (ball[lag:] - ball[:-lag]) / dt
    bv[~np.isfinite(bv)] = 0.
    bv = np.clip(bv, -40, 40)
    bpos = np.nan_to_num(bf_pos, nan=0.)
    inside = ((bpos[:, 0] > 0) & (bpos[:, 0] < L) & (bpos[:, 1] > 0) & (bpos[:, 1] < W)).astype(float)
    B = np.column_stack([vis, bpos[:, 0] / L, bpos[:, 1] / W, bv[:, 0] / 20, bv[:, 1] / 20,
                         np.linalg.norm(bv, axis=1) / 20, np.minimum(since, 10.) / 5, inside]).astype(np.float16)
    pv = np.zeros_like(P)
    pv[lag:] = (P[lag:] - P[:-lag]) / dt
    pv[~np.isfinite(pv)] = 0.
    pv = np.clip(pv, -12, 12)
    present = np.isfinite(P).all(2)
    rel = P - bpos[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    relv = np.linalg.norm(pv - bv[:, None, :], axis=2)
    X = np.stack([present.astype(float),
                  np.clip(rel[..., 0], -40, 40) / 20, np.clip(rel[..., 1], -40, 40) / 20,
                  np.minimum(dist, 40) / 20, pv[..., 0] / 8, pv[..., 1] / 8,
                  np.minimum(relv, 30) / 10, P[..., 0] / L, P[..., 1] / W,
                  np.exp(-np.nan_to_num(dist, nan=99.) / 2.)], axis=2)
    X[~present] = 0.
    X = np.nan_to_num(X, nan=0.).astype(np.float16)
    return B, X, team


def build_net(hidden=64, heads=1):
    import torch
    from torch import nn

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pe = nn.Sequential(nn.Linear(PF, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.score = nn.Linear(hidden, 1)
            self.be = nn.Sequential(nn.Linear(BF, 32), nn.ReLU())
            self.frame = nn.Sequential(nn.Linear(32 + 4 * hidden, 128), nn.ReLU())
            self.convs = nn.ModuleList([nn.Conv1d(128, 128, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
            self.heads = heads
            self.head = nn.Linear(128, 3 * heads)

        def pool(self, e, m):
            s = self.score(e).squeeze(-1).masked_fill(~m, -1e4)
            a = torch.softmax(s, dim=-1).unsqueeze(-1)
            att = (a * e).sum(-2)
            mean = (e * m.unsqueeze(-1)).sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)
            return torch.cat([att, mean], -1)

        def forward(self, B, X, M0, M1, swap=False, head=None):
            # B [b,T,BF], X [b,T,k,PF], M0/M1 [b,T,k] team masks (present & team)
            e = self.pe(X)
            p0, p1 = self.pool(e, M0), self.pool(e, M1)
            if swap:
                p0, p1 = p1, p0
            f = self.frame(torch.cat([self.be(B), p0, p1], -1)).transpose(1, 2)   # [b,128,T]
            for c in self.convs:
                f = f + torch.relu(c(f))
            out = self.head(f[:, :, f.shape[2] // 2])
            if self.heads > 1:                      # Round 16: one H/A/D head per label source; 0 = target
                out = out.view(out.shape[0], self.heads, 3)
                idx = torch.zeros(out.shape[0], dtype=torch.long) if head is None else head
                out = out[torch.arange(out.shape[0]), idx]
            return out[:, [1, 0, 2]] if swap else out

    return SetNet()


class Item:
    def __init__(self, geo, labels=None):
        self.B, self.X, self.team = frame_features(geo)
        self.n = len(self.B)
        self.y = None if labels is None else np.asarray(labels).astype(int)
        present = self.X[..., 0] > .5
        self.M0 = present & (self.team == 0)[None, :]
        self.M1 = present & (self.team == 1)[None, :]

    def windows(self, centers, K=None):
        off = np.arange(-HALF, HALF + 1) * STEP
        idx = np.clip(centers[:, None] + off[None, :], 0, self.n - 1)
        X, M0, M1 = self.X[idx], self.M0[idx], self.M1[idx]
        k = X.shape[2]
        if K is not None and K > k:
            X = np.concatenate([X, np.zeros(X.shape[:2] + (K - k, PF), X.dtype)], 2)
            M0 = np.concatenate([M0, np.zeros(M0.shape[:2] + (K - k,), bool)], 2)
            M1 = np.concatenate([M1, np.zeros(M1.shape[:2] + (K - k,), bool)], 2)
        return self.B[idx], X, M0, M1


def _tensors(B, X, M0, M1):
    import torch
    return (torch.from_numpy(B.astype(np.float32)), torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(M0), torch.from_numpy(M1))


def train(groups, epochs=4, per_epoch=240000, batch=512, seed=20261116, threads=8, log=print, head_of=None,
          init=None, lr=2e-3):
    """groups: {source: [Item, ...]} (labels 0/1/2, -1 unscored). Equal samples per source."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    pools = {}
    for src, items in groups.items():
        cand = [(i, np.flatnonzero(it.y >= 0)) for i, it in enumerate(items)]
        pools[src] = cand
    counts = np.zeros(3)
    for items in groups.values():
        for it in items:
            counts += np.bincount(it.y[it.y >= 0], minlength=3)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    heads = 1 + max(head_of.values()) if head_of else 1
    net = build_net(heads=heads)
    if init is not None:                         # Round 17: fine-tune from a trained network
        net.load_state_dict(init.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (per_epoch // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    srcs = list(groups)
    K = max(it.X.shape[1] for items in groups.values() for it in items)
    t0 = time.time()
    for step in range(steps):
        bs = []
        for j, src in enumerate(srcs):
            m = batch // len(srcs)
            pick = rng.integers(0, len(pools[src]), m)
            for i in np.unique(pick):
                ii, frames = pools[src][i]
                c = frames[rng.integers(0, len(frames), int((pick == i).sum()))]
                it = groups[src][ii]
                bs.append((it.windows(c, K), it.y[c], (head_of or {}).get(src, 0)))
        B = np.concatenate([b[0][0] for b in bs]); X = np.concatenate([b[0][1] for b in bs])
        M0 = np.concatenate([b[0][2] for b in bs]); M1 = np.concatenate([b[0][3] for b in bs])
        y = np.concatenate([b[1] for b in bs])
        hd = torch.from_numpy(np.concatenate([np.full(len(b[1]), b[2]) for b in bs]).astype(np.int64))
        Bt, Xt, M0t, M1t = _tensors(B, X, M0, M1)
        yt = torch.from_numpy(y)
        swap = bool(rng.random() < .5)
        out = net(Bt, Xt, M0t, M1t, swap=swap, head=hd if heads > 1 else None)
        loss = lossf(out, yt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            log(json.dumps(dict(stage="gnn_train", step=step, of=steps, loss=round(float(loss), 4), s=round(time.time() - t0))))
    return net


def predict(net, geo, batch=1024, threads=8):
    """P(H, A, D) at every frame (computed every STEP frames, linearly interpolated)."""
    import torch
    torch.set_num_threads(threads)
    it = Item(geo)
    centers = np.arange(0, it.n, STEP)
    outs = []
    net.eval()
    with torch.no_grad():
        for s in range(0, len(centers), batch):
            B, X, M0, M1 = _tensors(*it.windows(centers[s:s + batch]))
            p = torch.softmax(net(B, X, M0, M1), -1) + torch.softmax(net(B, X, M0, M1, swap=True), -1)
            outs.append((p / 2).numpy())
    P = np.concatenate(outs)
    full = np.column_stack([np.interp(np.arange(it.n), centers, P[:, j]) for j in range(3)])
    return (full / full.sum(1, keepdims=True)).astype(np.float32)


def save(net, path, meta):
    import torch
    meta = dict(meta, heads=getattr(net, "heads", 1))
    torch.save(dict(state_dict=net.state_dict(), meta=meta), path)


def load(path):
    import torch
    cp = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(heads=int((cp.get("meta") or {}).get("heads", 1)))
    net.load_state_dict(cp["state_dict"])
    net.eval()
    return net
