"""G4/G5: C3 retraining with physical corruptions (and optionally gap-context HGB features).

Training views (rondo.corruptions.training_views_v2): six per match, no frozen dead ball
(stay 15% with smooth jitter, vanish 25%, three-stage 60%), live gaps of two kinds, smooth noise.

Budget mode (--budget): train on WOH, WN1, WMX, WQQ only; every 10 epochs score WPY `clean` and
`dead_restart` (raw spellF1, final decoder: team 4, temp 1.4, k 4) up to --max-epochs. WOY is
not used. Plateau epoch, fixed before looking: the smallest evaluated epoch whose mean score is
within 0.5 points of the best mean.

Final mode: train on the five IDSSE matches for --epochs. Groningen is not used (C3).

    python -m rondo.train_c3 --budget --out data/artifacts/rondo/c3_budget [--gap-features]
    python -m rondo.train_c3 --epochs E --out data/artifacts/rondo/cand_c3 [--gap-features]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import monotonic

import numpy as np

from . import corruptions
from .matches import ROOT

TRAIN5 = ("J03WOH", "J03WN1", "J03WMX", "J03WQQ", "J03WPY")
TRAIN4 = TRAIN5[:4]
SEED = 20261116
CACHE = ROOT / "data/artifacts/rondo/c3_features"
HGB_STRIDE = 8


def wcsf_full():
    """Groningen, coordinate-corrected, 25 Hz, both halves, nearest native label per row."""
    from .matches import load_wcsf
    m = load_wcsf(corrected=True)
    geo = m.geometry
    native = np.asarray(m.scorer_data["labels"])
    nb = np.asarray(m.scorer_data["blocks"])
    labels = np.full(len(geo["ball"]), -1, np.int8)
    for (a, b), (na, nbb) in zip(np.asarray(geo["blocks"]), nb):
        src = np.clip(na + np.rint(np.arange(b - a) / 2.5).astype(int), na, nbb - 1)
        labels[a:b] = native[src]
    return dict(ball=geo["ball"], players=geo["players"], team_of_track=geo["team_of_track"], fps=geo["fps"],
                pitch=geo["pitch"], blocks=np.asarray(geo["blocks"]), labels=labels, match_id="WCSF", source="WCSF")


def load_match(code):
    from src.train import load_external
    return load_external(ROOT / "data/idsse_external" / ("%s_external_v1.npz" % code))


def view_features(code, index, data):
    """{view: (x330 path, gap path)} for the six training views, cached on disk."""
    from src.features import build_features
    from .gap_features import gap_context
    CACHE.mkdir(parents=True, exist_ok=True)
    out, names = {}, None
    views = None
    for v in range(6):
        vname = "v2_%d" % v
        px, pg = CACHE / ("%s_%s.npy" % (code, vname)), CACHE / ("%s_%s_gap.npy" % (code, vname))
        if not (px.is_file() and pg.is_file()):
            if views is None:
                views = corruptions.training_views_v2(data, index)
            x, names = build_features(views[vname], profile="contact_motion")
            np.save(px, x.astype(np.float32)); del x
            g, _ = gap_context(views[vname])
            np.save(pg, g)
        out[vname] = (px, pg)
        print(json.dumps(dict(stage="features", match=code, view=vname)), flush=True)
    return out


def eval_views(code, data):
    """WPY evaluation inputs: clean and dead_restart (contact_motion x, gap x, geometry)."""
    from src.bundle import _geometry_only
    from src.features import build_features
    from .gap_features import gap_context
    geo = _geometry_only(data)
    dead = np.asarray(data["labels"]) == 2
    out = {}
    for cname in ("clean", "dead_restart"):
        g = corruptions.CONDITIONS[cname](geo, dead)
        px = CACHE / ("%s_eval_%s.npy" % (code, cname))
        if not px.is_file():
            x, _ = build_features(g, profile="contact_motion")
            np.save(px, x.astype(np.float32))
        gx, _ = gap_context(g)
        out[cname] = dict(geo=g, x=np.load(px, mmap_mode="r"), gap=gx)
    return out


def train_hgb(records, names, gap, threads):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    from src.symmetry import swap_feature_matrix
    from .gap_features import NAMES as GAP
    parent = joblib.load(ROOT / "models/ahmetcan_joint_checkpoint/hgb.joblib")
    base = list(parent["feature_names"])
    idx = [names.index(n) for n in base]
    hnames = base + (GAP if gap else [])
    xs, ys = [], []
    for r in records:
        labels = np.asarray(r["data"]["labels"])
        rows = np.arange(0, len(labels), HGB_STRIDE)
        rows = rows[labels[rows] >= 0]
        for px, pg in r["paths"]:
            sampled = np.asarray(np.load(px, mmap_mode="r")[rows][:, idx])
            if gap:
                sampled = np.concatenate([sampled, np.load(pg)[rows]], axis=1)
            y = labels[rows]
            xs += [sampled, swap_feature_matrix(sampled, hnames)]
            ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
    x, y = np.concatenate(xs), np.concatenate(ys).astype(int)
    del xs, ys
    w = (len(y) / (3 * np.maximum(np.bincount(y, minlength=3), 1)))[y]
    est = HistGradientBoostingClassifier(learning_rate=.05, max_iter=400, max_leaf_nodes=31,
                                         l2_regularization=1, min_samples_leaf=150,
                                         random_state=20261103, early_stopping=False)
    t = monotonic()
    with threadpool_limits(limits=threads):
        est.fit(x, y, sample_weight=w)
    art = dict(parent)
    art.update(classifier=est, feature_names=hnames, feature_profile="contact_gap" if gap else parent["feature_profile"],
               provenance=dict(train=[r["match_id"] for r in records], training_data="external IDSSE only, G4 views",
                               seed=20261103, hgb_row_stride=HGB_STRIDE))
    print(json.dumps(dict(stage="hgb_done", rows=int(len(y)), seconds=round(monotonic() - t))), flush=True)
    return art


def hgb_probability(art, x330, gapx, names):
    from src.bundle import _symmetric_probability
    base = [n for n in art["feature_names"] if not n.startswith("gap_") and "_gap_" not in n]
    idx = [names.index(n) for n in base]
    hx = np.asarray(x330)[:, idx]
    if art.get("feature_profile") == "contact_gap":
        hx = np.concatenate([hx, gapx], axis=1)
    return _symmetric_probability(art, hx, art["feature_names"])


def tcn_records(records):
    from experiments.transition_external import event_targets
    from src.features import blocks_of
    from src.train import spell_weights
    out = []
    for r in records:
        data = r["data"]
        weights = spell_weights(data, .5).astype(np.float32)
        for a, b in blocks_of(data):
            ch = a + np.flatnonzero(data["labels"][a + 1:b] != data["labels"][a:b - 1]) + 1
            for c in ch:
                weights[max(a, c - 5):min(b, c + 6)] *= 2.
        events, _ = event_targets(data)
        views = [np.load(px, mmap_mode="r") for px, _ in r["paths"]]
        out.append(dict(match_id=r["match_id"], data=data, x=views[0], feature_views=views,
                        weights=weights, events=events, condition="mixed"))
    return out


def train_tcn(records, names, epochs, threads, eval_every=0, eval_fn=None, seed=SEED):
    import torch
    from torch import nn
    from experiments import tcn_external as base
    from experiments import joint_transition_external as joint
    from experiments.can_spell_sampling import make_event_sampler
    from experiments.spell_objective_external import objective
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cp = torch.load(ROOT / "data/artifacts/can_five_match_spell_tcn_v1/five/best.pt", map_location="cpu",
                    weights_only=True)
    mean, scale = cp["mean"].numpy(), cp["scale"].numpy()
    permutation, sign = base.exchange_schema(names)
    recs = tcn_records(records)
    counts, evc = np.zeros(3), np.zeros(4)
    for r in recs:
        y, e = r["data"]["labels"], r["events"]
        counts += np.bincount(y[y >= 0], minlength=3); evc += np.bincount(e[e >= 0], minlength=4)
    counts[:2] = counts[:2].mean(); evc[1:3] = evc[1:3].mean()
    cw = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)
    ew = torch.tensor(evc.sum() / (4 * evc), dtype=torch.float32)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    rng, view_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    model = joint.JointTemporalNet(2 * len(names), 64)
    model.load_state_dict(cp["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    inv = base.sequence_inventory(recs)
    sampler = make_event_sampler(inv, view_rng=view_rng)
    log = []
    for epoch in range(1, epochs + 1):
        t = monotonic(); model.train(); losses = []
        for _ in range(64):
            x, y, w, e = sampler(inv, rng, mean, scale, permutation, sign, 8, 256)
            opt.zero_grad(set_to_none=True)
            st, ev = model(x)
            sl = slice(base.CONTEXT, base.CONTEXT + 256)
            loss = objective(st[:, :, sl], ev[:, :, sl], y, w, e, cw, ew, 0.)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(monotonic() - t, 1))
        if eval_every and epoch % eval_every == 0 and eval_fn is not None:
            row.update(eval_fn(model, mean, scale, permutation, sign))
        log.append(row)
        print(json.dumps(row), flush=True)
    return model, cp, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget", action="store_true")
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--gap-features", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--hgb-only", action="store_true", help="final mode: train only the HGB")
    p.add_argument("--tcn-only", type=Path, default=None, help="final mode: reuse this HGB, train only the TCN")
    p.add_argument("--with-wcsf", action="store_true", help="add the full Groningen match (same recipe)")
    a = p.parse_args()
    import joblib
    import torch
    a.out.mkdir(parents=True, exist_ok=True)
    codes = list(TRAIN4 if a.budget else TRAIN5)
    records, names = [], None
    for i, code in enumerate(codes):
        data = load_match(code)
        paths = view_features(code, TRAIN5.index(code), data)
        records.append(dict(match_id=code, data=data, paths=list(paths.values())))
    if a.with_wcsf:
        from experiments import can_spell_sampling
        can_spell_sampling.TRAIN_IDS = set(can_spell_sampling.TRAIN_IDS) | {"WCSF"}
        data = wcsf_full()
        records.append(dict(match_id="WCSF", data=data, paths=list(view_features("WCSF", 5, data).values())))
        codes.append("WCSF")
    from src.features import build_features
    g0 = {k: records[0]["data"][k] for k in ("ball", "players", "team_of_track", "fps", "pitch")}
    g0["ball"], g0["players"] = g0["ball"][:500], g0["players"][:500]
    _, names = build_features(g0, profile="contact_motion")
    if a.tcn_only:
        hgb = joblib.load(a.tcn_only)
    else:
        hgb = train_hgb(records, names, a.gap_features, a.threads)
    joblib.dump(hgb, a.out / "hgb.joblib", compress=3)
    report = dict(mode="budget" if a.budget else "final", train=list(codes), gap_features=a.gap_features,
                  seed=a.seed, hgb_reused=str(a.tcn_only) if a.tcn_only else None)
    if a.budget:
        from experiments.joint_transition_external import infer_joint
        from lab.model import decode
        from src.train import metrics
        wpy = load_match("J03WPY")
        ev = eval_views("J03WPY", wpy)
        dec = dict(dead_cost=6., event_strength=1., stride=2, team_cost=4., temperature=1.4, dead_bias=4.)
        hgb_p = {c: hgb_probability(hgb, v["x"], v["gap"], names) for c, v in ev.items()}

        def eval_fn(model, mean, scale, permutation, sign):
            model.eval()
            res = {}
            for c, v in ev.items():
                state, event = infer_joint(model, dict(data=v["geo"], x=v["x"]), mean, scale, permutation, sign,
                                           torch.device("cpu"))
                st = .5 * (state + hgb_p[c])
                pred = decode(st, event, v["geo"], dec)
                res["WPY_%s" % c] = float(metrics(wpy, pred)["spell_f1_mean"])
            model.train()
            res["WPY_mean"] = float(np.mean(list(res.values())))
            return res
        _, _, log = train_tcn(records, names, a.max_epochs, a.threads, 10, eval_fn, a.seed)
        evals = [r for r in log if "WPY_mean" in r]
        best = max(r["WPY_mean"] for r in evals)
        plateau = min(r["epoch"] for r in evals if r["WPY_mean"] >= best - .005)
        report.update(curve=log, best_mean=best, plateau_epoch=plateau)
    elif a.hgb_only:
        report.update(note="hgb only")
    else:
        model, cp, log = train_tcn(records, names, a.epochs, a.threads, seed=a.seed)
        torch.save(dict(state_dict=model.state_dict(), mean=cp["mean"], scale=cp["scale"], feature_names=list(names),
                        train_matches=list(codes), validation_match=None,
                        protocol=dict(version="rondo-c3", epochs=a.epochs, seed=a.seed, views="training_views_v2",
                                      gap_features=a.gap_features, selection="none (fixed epochs)")),
                   a.out / "tcn.pt")
        report.update(curve=log)
    (a.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
