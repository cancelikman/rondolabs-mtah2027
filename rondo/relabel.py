"""Round 8 R1: relabel EXTERNAL matches with the competition's possession definition.

CHALLENGE.md: possession changes only when the opponent gains controlled possession; it stays
with the last team through contests and loose balls; control proxies are holding the ball for a
few seconds or completing more than two consecutive passes; an uncontrolled single touch, a
deflection that does not break the pass and an aerial ball without following control do not
count; after a stoppage the restart has a clear owner; D is not possession and splits it.

Algorithm (inside each live stretch; D and unknown frames never change):
 1. split the raw label into islands = maximal runs of one team;
 2. evidence per island: hold = duration; passes = geometric passes (the team's player nearest
    to the ball within 1.5 m changes, and the ball moved >= 5 m between the two contacts);
    first_after_dead = the island opens the stretch right after a stoppage;
 3. controlled = first_after_dead or hold >= C or (P set and passes >= P);
 4. an uncontrolled island takes the team of the last controlled island before it in the
    stretch; at the start of a stretch it takes the first controlled island's team;
 5. the possession change happens at the first frame of a controlled island.
Passes come from geometry for every source (event data is not used for IDSSE here).

Variants: ch2 (C=2 s), ch3 (3 s), ch3p2 (3 s or 2 passes), ch3p3 (3 s or 3 passes).
Metrica also gets an event-based "challenge" definition in rondo.open_data.

    python -m rondo.relabel --out reports/rondo/r1_relabel.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H, A, D = 0, 1, 2
VARIANTS = {"ch2": (2., None), "ch3": (3., None), "ch3p2": (3., 2), "ch3p3": (3., 3)}
CONTACT_M = 1.5
PASS_M = 5.


def _runs(x):
    x = np.asarray(x)
    if not len(x):
        return []
    e = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    return [(int(s), int(t), int(x[s])) for s, t in zip(e[:-1], e[1:])]


def contacts(ball, players, team):
    """Per frame: (track index, team) of the player within CONTACT_M of the visible ball, else (-1, -1)."""
    n = len(ball)
    idx = np.full(n, -1, np.int64)
    tm = np.full(n, -1, np.int8)
    for s in range(0, n, 20000):
        d = np.linalg.norm(players[s:s + 20000] - ball[s:s + 20000, None, :], axis=2)
        dd = np.where(np.isfinite(d), d, np.inf)
        j = np.argmin(dd, axis=1)
        ok = dd[np.arange(len(j)), j] <= CONTACT_M
        idx[s:s + 20000] = np.where(ok, j, -1)
        tm[s:s + 20000] = np.where(ok, team[j], -1)
    return idx, tm


def count_passes(s, e, team_v, cidx, ctm, ball):
    passes, last_j, last_pos = 0, -1, None
    for i in range(s, e):
        if ctm[i] != team_v:
            continue
        j = cidx[i]
        if j != last_j:
            if last_j >= 0 and last_pos is not None and np.linalg.norm(ball[i] - last_pos) >= PASS_M:
                passes += 1
            last_j = j
        last_pos = ball[i]
    return passes


def relabel(labels, blocks, fps, C, P=None, ball=None, players=None, team=None):
    lab = np.asarray(labels).copy()
    if P is not None:
        cidx, ctm = contacts(np.asarray(ball, float), np.asarray(players, float), np.asarray(team))
    out = lab.copy()
    for a, b in blocks:
        seg = lab[a:b]
        live = np.isin(seg, [H, A]).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, live, 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            s, e = a + s, a + e
            after_dead = s > a and lab[s - 1] == D
            isl = [(s + x, s + y, v) for x, y, v in _runs(lab[s:e])]
            ctrl = []
            for k, (x, y, v) in enumerate(isl):
                c = (k == 0 and after_dead) or (y - x) >= C * fps
                if not c and P is not None:
                    c = count_passes(x, y, v, cidx, ctm, np.asarray(ball, float)) >= P
                ctrl.append(c)
            if not any(ctrl):
                continue
            first = next(v for (x, y, v), c in zip(isl, ctrl) if c)
            cur = None
            for (x, y, v), c in zip(isl, ctrl):
                if c:
                    cur = v
                out[x:y] = v if c else (cur if cur is not None else first)
    return out


def apply(data, variant):
    """Relabelled copy of an external data dict (native rate)."""
    from src.features import blocks_of
    C, P = VARIANTS[variant]
    blocks = [tuple(map(int, b)) for b in blocks_of(data)]
    lab = relabel(data["labels"], blocks, float(data["fps"]), C, P, data["ball"], data["players"],
                  data["team_of_track"])
    out = dict(data)
    out["labels"] = lab
    return out


def summary(labels, blocks, fps, raw=None):
    lens = []
    for a, b in blocks:
        for s, e, v in _runs(np.asarray(labels)[a:b]):
            if v in (H, A):
                lens.append((e - s) / fps)
    graded = (np.asarray(labels) >= 0).sum() / fps / 60
    res = dict(possessions_per_min=round(len(lens) / graded, 3), mean_s=round(float(np.mean(lens)), 2),
               median_s=round(float(np.median(lens)), 2))
    if raw is not None:
        live = np.isin(raw, [H, A])
        res["changed_frame_share"] = round(float((np.asarray(labels) != raw)[raw >= 0].mean()), 4)
        res["live_agreement_with_raw"] = round(float((np.asarray(labels)[live] == raw[live]).mean()), 4)
    return res


def datasets():
    from src.train import load_external
    from .matches import ROOT
    from .open_data import load_metrica_native, load_skillcorner_native, skillcorner_ids
    sets = {}
    for code in ("J03WOH", "J03WN1", "J03WMX", "J03WQQ", "J03WPY"):
        sets.setdefault("IDSSE_train", []).append(load_external(ROOT / "data/idsse_external" / ("%s_external_v1.npz" % code)))
    sets["WOY"] = [load_external(ROOT / "data/idsse_external/J03WOY_external_v1.npz")]
    sets["Metrica"] = [load_metrica_native(g, "control") for g in (1, 2)]
    sets["SkillCorner"] = [load_skillcorner_native(i) for i in skillcorner_ids()[:10]]
    return sets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from src.features import blocks_of
    sets = datasets()
    report = {}
    for name, ds in sets.items():
        for variant in ["raw", *VARIANTS]:
            rows = []
            for d in ds:
                blocks = [tuple(map(int, b)) for b in blocks_of(d)]
                lab = d["labels"] if variant == "raw" else apply(d, variant)["labels"]
                rows.append(summary(lab, blocks, float(d["fps"]), None if variant == "raw" else np.asarray(d["labels"])))
            report.setdefault(name, {})[variant] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
            print(name, variant, report[name][variant], flush=True)
    target = 14.2
    dist = {v: float(np.mean([report[s][v]["mean_s"] for s in report])) for v in VARIANTS}
    ranked = sorted(VARIANTS, key=lambda v: abs(dist[v] - target))
    best = abs(dist[ranked[0]] - target)
    tied = [v for v in ranked if abs(dist[v] - target) - best < .5]
    if len(tied) > 1 and "ch3p3" in tied:      # tie rule fixed in the round-8 note
        ranked.remove("ch3p3"); ranked.insert(0, "ch3p3")
    res = dict(target_mean_s=target, mean_live_possession_s_by_variant=dist, CH=ranked[0], CH_prime=ranked[1],
               sets=report)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(dict(dist=dist, CH=res["CH"], CH_prime=res["CH_prime"]), indent=1))


if __name__ == "__main__":
    main()
