"""Round 9 A3: HGB member retrained with the twin.

Training: the cached C3w views of IDSSE WOH/WN1/WMX/WQQ/WPY + full Groningen (rondo.train_c3,
labels T* = IDSSE <1 s merged, Groningen raw) plus the 6 SkillCorner twin training matches without
synthetic corruption (labels <0.5 s merged). Same 180 HGB features and hyper-parameters as C3w.
Twin rows carry {30%, 50%} of the total sample weight.

    python -m rondo.train_a3 --twin-share 0.3 --out data/artifacts/rondo/cand_a3_w30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .matches import ROOT, load, merged_labels
from .train_c3 import CACHE as C3CACHE, HGB_STRIDE, TRAIN5, load_match, wcsf_full
from .twin import TRAIN as TWIN_TRAIN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twin-share", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    from src.features import blocks_of, build_features
    from src.symmetry import swap_feature_matrix
    from .phase_prior import gt25
    from .d2_baseline import variant_match
    parent = joblib.load(ROOT / "models/ahmetcan_joint_checkpoint/hgb.joblib")
    base = list(parent["feature_names"])
    cache = ROOT / "data/artifacts/rondo/a3_twin_features"
    cache.mkdir(parents=True, exist_ok=True)
    xs, ys, src = [], [], []
    names = None
    for i, code in enumerate(list(TRAIN5) + ["WCSF"]):
        data = load_match(code) if code != "WCSF" else wcsf_full()
        lab = np.asarray(data["labels"])
        if code != "WCSF":
            blocks = [tuple(map(int, b)) for b in blocks_of(data)]
            lab = merged_labels(lab, blocks, 1., 25.)
        rows = np.arange(0, len(lab), HGB_STRIDE)
        rows = rows[lab[rows] >= 0]
        for v in range(6):
            x = np.load(C3CACHE / ("%s_v2_%d.npy" % (code, v)), mmap_mode="r")
            if names is None:
                g0 = {k: data[k] for k in ("ball", "players", "team_of_track", "fps", "pitch")}
                g0["ball"], g0["players"] = g0["ball"][:500], g0["players"][:500]
                _, names = build_features(g0, profile="contact_motion")
            idx = [names.index(n) for n in base]
            s = np.asarray(x[rows][:, idx]); y = lab[rows]
            xs += [s, swap_feature_matrix(s, base)]; ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
            src += [np.zeros(2 * len(y), bool)]
        print(json.dumps(dict(stage="views", match=code)), flush=True)
    for name in TWIN_TRAIN:
        m = load(name)
        mm = variant_match(m, "merge<0.5s")
        path = cache / ("%s.npy" % name)
        if path.is_file():
            x = np.load(path)
        else:
            x, _ = build_features(m.geometry, profile="contact_motion")
            np.save(path, x.astype(np.float32))
        idx = [names.index(n) for n in base]
        y25 = gt25(mm)
        rows = np.arange(0, len(y25), 3)
        rows = rows[y25[rows] >= 0]
        s = x[rows][:, idx]; y = y25[rows]
        xs += [s, swap_feature_matrix(s, base)]; ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
        src += [np.ones(2 * len(y), bool)]
        print(json.dumps(dict(stage="twin", match=name)), flush=True)
    x, y, tw = np.concatenate(xs), np.concatenate(ys).astype(int), np.concatenate(src)
    del xs, ys
    w = (len(y) / (3 * np.maximum(np.bincount(y, minlength=3), 1)))[y]
    w[tw] *= a.twin_share * w[~tw].sum() / ((1 - a.twin_share) * w[tw].sum())
    est = HistGradientBoostingClassifier(learning_rate=.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1,
                                         min_samples_leaf=150, random_state=20261103, early_stopping=False)
    with threadpool_limits(limits=a.threads):
        est.fit(x, y, sample_weight=w)
    art = dict(parent)
    art.update(classifier=est, feature_names=base,
               provenance=dict(train=list(TRAIN5) + ["WCSF"] + list(TWIN_TRAIN), twin_weight_share=a.twin_share,
                               labels="IDSSE <1 s merged (T*), Groningen raw, SkillCorner <0.5 s merged"))
    a.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, a.out / "hgb.joblib", compress=3)
    print(json.dumps(dict(stage="hgb_done", rows=int(len(y)), twin_rows=int(tw.sum()))), flush=True)


if __name__ == "__main__":
    main()
