"""External match loaders (IDSSE WOY/R9 and Groningen/WCSF, coordinate-corrected)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WCSF_DIR = ROOT / "data/artifacts/wcsf_schema"
# S2: experiments/can_wcsf_frozen_transfer.load_wcsf returns centre-origin metres,
# while src/features.py expects corner-origin metres (boundary_distance, out_of_pitch).
WCSF_SHIFT = np.array([52.5, 34.0], dtype=np.float32)


class Match:
    """geometry: model input at 25 Hz. score(pred25) -> official metrics on native labels."""

    def __init__(self, name, geometry, dead25, scorer_data, mapping=None):
        self.name, self.geometry, self.dead25 = name, geometry, dead25
        self.scorer_data, self.mapping = scorer_data, mapping

    def score(self, pred25):
        from src.train import metrics
        pred = np.asarray(pred25)
        if self.mapping is not None:
            pred = pred[self.mapping[:, 1]]
        m = metrics(self.scorer_data, pred)
        d = m["per_class"]["D"]
        return dict(spell=float(m["spell_f1_mean"]), macro=float(m["macro_f1"]),
                    d_recall=float(d["recall"]), d_precision=float(d["precision"]),
                    alive_dead=float(m["alive_dead_f1"]), team_acc=float(m["team_acc_gt_alive"]),
                    n_pred=int(m["n_pred_spells"]), n_gt=int(m["n_gt_spells"]))


def merged_labels(labels, blocks, n_seconds, fps):
    """Brief round 4, Ek B (scoring only): inside live play an A-B-A run with B shorter than
    n_seconds becomes A; a possession bounded by a stoppage is kept."""
    out = np.asarray(labels).copy()
    limit = n_seconds * fps
    for a, b in blocks:
        changed = True
        while changed:
            changed = False
            seg = out[a:b]
            edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
            runs = [(int(s), int(e), int(seg[s])) for s, e in zip(edges[:-1], edges[1:])]
            for i in range(1, len(runs) - 1):
                (_, _, pv), (s, e, v), (_, _, nv) = runs[i - 1], runs[i], runs[i + 1]
                if v in (0, 1) and pv == nv and pv in (0, 1) and pv != v and e - s < limit:
                    out[a + s:a + e] = pv
                    changed = True
                    break
    return out


def score_targets(match, pred25, targets=(0, 2, 3)):
    """{target: score dict}; target 0 = raw labels, N = merged with N seconds (Ek B)."""
    from src.features import blocks_of
    out = {}
    base = match.scorer_data
    fps = float(base["fps"])
    blocks = [tuple(map(int, b)) for b in blocks_of(base)]
    for n in targets:
        if n == 0:
            out["raw"] = match.score(pred25)
            continue
        key = ("_merged", n)
        if key not in match.__dict__:
            data = dict(base)
            data["labels"] = merged_labels(base["labels"], blocks, n, fps)
            match.__dict__[key] = data
        m = Match(match.name, match.geometry, match.dead25, match.__dict__[key], match.mapping)
        out["N%d" % n] = m.score(pred25)
    return out


def load_idsse(code):
    from src.bundle import _geometry_only
    from src.train import load_external
    code = code.upper() if code.upper().startswith("J03") else "J03" + code.upper()
    data = load_external(ROOT / "data/idsse_external" / ("%s_external_v1.npz" % code))
    labels = np.asarray(data["labels"])
    return Match(code, _geometry_only(data), labels == 2, data)


def load_wcsf(corrected=True):
    from experiments.can_skillcorner_audit import resample_geometry
    from experiments.can_wcsf_frozen_transfer import load_wcsf as raw_loader
    from src.bundle import _geometry_only
    schema = json.loads((ROOT / "reports/external/wcsf_possession_schema.json").read_text())
    data = raw_loader([WCSF_DIR / "first.zip", WCSF_DIR / "second.zip"], schema["halves"])
    if corrected:
        data["ball"] = data["ball"] + WCSF_SHIFT
        data["players"] = data["players"] + WCSF_SHIFT
    geometry, mapping = resample_geometry(_geometry_only(data))
    n25 = len(geometry["ball"])
    labels = np.asarray(data["labels"])
    dead25 = np.interp(np.arange(n25), mapping[:, 1], (labels == 2).astype(float)) > .5
    return Match("WCSF" if corrected else "WCSF_uncorrected", geometry, dead25, data, mapping)


def load_wcsf_second_half():
    """Groningen scored on the second half only (first half may be training data)."""
    m = load_wcsf(corrected=True)
    data = dict(m.scorer_data)
    labels = np.asarray(data["labels"]).copy()
    a, b = np.asarray(data["blocks"])[0]
    labels[a:b] = -1
    data["labels"] = labels
    h2 = Match("WCSF_H2", m.geometry, m.dead25, data, m.mapping)
    h2.cache_name = "WCSF"   # same geometry, same posteriors
    return h2


def load(name):
    if name.upper().startswith("SC"):
        from .open_data import load_skillcorner
        return load_skillcorner(name[2:])
    if name.upper().startswith("MT"):
        from .open_data import load_metrica
        game, definition = name[2:].split("_")
        m = load_metrica(int(game), definition)
        m.cache_name = "MT%s" % game   # both label definitions share geometry and posteriors
        return m
    if name.upper().startswith("PF"):
        from .pff_data import load_pff
        return load_pff(name[2:])
    if name.upper() == "WCSF_H2":
        return load_wcsf_second_half()
    if name.upper().startswith("WCSF"):
        return load_wcsf(corrected=not name.upper().endswith("_UNCORRECTED"))
    return load_idsse(name)
