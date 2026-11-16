"""Round 6 A0/A1: Smart Tagging loader, semantic map, inventory and tags-only baseline.

Smart Tagging is an organiser-provided automatic input ("a strong prior, for free", DATA.md).
This module reads it openly; everything it writes is aggregate. The semantic map below is fixed
from the tag names and DATA.md, never corrected with ground truth. An unlabelled consistency
check (team of the player nearest to the ball inside each phase) can only demote a type to
UNKNOWN when it carries no team information (share near 50%).

    python -m rondo.tags inventory --data <dir> --out reports/rondo/a0_tag_inventory.json
    python -m rondo.tags baseline  --data <dir> --package models/rondo_r5 --out reports/rondo/a1_tags_only.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H, A, D = 0, 1, 2
FPS = 25
MAP_VERSION = "r6-v1"
# Fixed semantic map (from the tag names / DATA.md: "every phase carries a Team tag").
SEMANTIC = {
    # The tagged team has the ball: it is building, progressing, attacking or playing it long.
    "BUILD UP": "POSSESSING",
    "PROGRESSION": "POSSESSING",
    "BALL IN FINAL THIRD": "POSSESSING",
    "BALL IN THE BOX": "POSSESSING",
    "CREATING CHANCES": "POSSESSING",
    "ATTACKING TRANSITION": "POSSESSING",   # just won the ball and attacks
    "LONG BALL": "POSSESSING",              # the team playing the long ball
    "PLAYERS IN THE BOX": "POSSESSING",     # attacking players arriving in the box
    # The tagged team is out of possession.
    "DEFENDING IN DEFENSIVE THIRD": "DEFENDING",
    "DEFENDING IN MIDDLE THIRD": "DEFENDING",
    "DEFENDING IN ATTACKING THIRD": "DEFENDING",
    "DEFENSIVE TRANSITION": "DEFENDING",    # just lost the ball
    # Handled elsewhere or no possession meaning.
    "SET PIECES": "SET_PIECE",
    "GOALS": "UNKNOWN",
}
HOME, AWAY = "barcelona", "atletic club"


def load(data_dir):
    """[(code, type, team(0/1/None), start_frame, end_frame)] via the organiser's loader."""
    from starter.load_smart_tagging import load as load_tags
    out = []
    for t in load_tags(data_dir=str(data_dir)):
        team = (t["tags"].get("Team") or "").strip().lower()
        out.append(dict(code=(t["code"] or "").strip().upper(), type=(t["tags"].get("Type") or "").strip(),
                        team=0 if team == HOME else 1 if team == AWAY else None,
                        start=int(t["start_frame"]), end=int(t["end_frame"]), raw=t))
    return out


def scored_mask(n):
    from .rules import published_halves
    m = np.zeros(n, bool)
    for a, b in published_halves():
        m[a:b] = True
    return m


def nearest_team(geo):
    """Team (0/1) of the player nearest to the visible ball per frame, -1 if unknown."""
    ball = np.asarray(geo["ball"], float)
    players = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"])
    out = np.full(len(ball), -1, np.int8)
    dist = np.full(len(ball), np.nan)
    chunk = 20000
    for s in range(0, len(ball), chunk):
        d = np.linalg.norm(players[s:s + chunk] - ball[s:s + chunk, None, :], axis=2)
        ok = np.isfinite(d).any(1)
        j = np.where(ok, np.nanargmin(np.where(np.isfinite(d), d, np.inf), axis=1), 0)
        out[s:s + chunk] = np.where(ok, team[j], -1)
        dist[s:s + chunk] = np.where(ok, d[np.arange(len(j)), j], np.nan)
    return out, dist


def prior_team(tag, semantic=SEMANTIC):
    cls = semantic.get(tag["code"], "UNKNOWN")
    if tag["team"] is None:
        return None
    if cls == "POSSESSING":
        return tag["team"]
    if cls == "DEFENDING":
        return 1 - tag["team"]
    return None


def consistency(tags, geo, n):
    """Per code: share of frames whose nearest-player team equals the semantic prior team."""
    nt, _ = nearest_team(geo)
    sc = scored_mask(n)
    res = {}
    for code, cls in SEMANTIC.items():
        if cls not in ("POSSESSING", "DEFENDING"):
            continue
        hit = tot = 0
        for t in tags:
            if t["code"] != code or t["team"] is None:
                continue
            pt = prior_team(t)
            seg = nt[t["start"]:t["end"]][sc[t["start"]:t["end"]]]
            seg = seg[seg >= 0]
            hit += int((seg == pt).sum()); tot += len(seg)
        share = hit / tot if tot else None
        res[code] = dict(semantic=cls, frames=tot, share_nearest_is_prior_team=round(share, 4) if share else None,
                         confidence=round(max(0., 2 * share - 1), 4) if share else 0.,
                         demoted_to_unknown=bool(share is not None and abs(share - .5) < .05))
    return res


def inventory(tags, n):
    sc = scored_mask(n)
    graded = sc.sum()
    by = {}
    for t in tags:
        r = by.setdefault(t["code"], dict(n=0, seconds=[], teams={"Barcelona": 0, "Atletic Club": 0, "N/A": 0}, cover=np.zeros(n, bool)))
        r["n"] += 1
        r["seconds"].append((t["end"] - t["start"]) / FPS)
        r["teams"][{0: "Barcelona", 1: "Atletic Club", None: "N/A"}[t["team"]]] += 1
        r["cover"][max(0, t["start"]):min(n, t["end"])] = True
    out = {}
    for code, r in sorted(by.items()):
        out[code] = dict(n=r["n"], total_s=round(float(np.sum(r["seconds"])), 1), median_s=round(float(np.median(r["seconds"])), 2),
                         teams=r["teams"], graded_coverage=round(float((r["cover"] & sc).sum() / graded), 4),
                         semantic=SEMANTIC.get(code, "UNKNOWN"))
    # phases with a prior team: overlap conflicts and team changes between consecutive phases
    vote = np.zeros((n, 2), np.int16)
    phases = sorted([t for t in tags if prior_team(t) is not None], key=lambda t: t["start"])
    for t in phases:
        vote[max(0, t["start"]):min(n, t["end"]), prior_team(t)] += 1
    conflict = (vote[:, 0] > 0) & (vote[:, 1] > 0)
    covered = (vote.sum(1) > 0)
    changes = sum(1 for p, q in zip(phases, phases[1:]) if prior_team(p) != prior_team(q))
    mins = graded / FPS / 60
    return out, dict(prior_phase_graded_coverage=round(float((covered & sc).sum() / graded), 4),
                     conflicting_overlap_share_of_covered=round(float((conflict & sc).sum() / max(1, (covered & sc).sum())), 4),
                     prior_team_changes=changes, prior_team_changes_per_min=round(changes / mins, 3),
                     reference_possessions_per_min=3.4)


def tags_only(tags, n, conf=None):
    """A1: POSSESSING -> tagged team, DEFENDING -> opponent, set-piece windows -> D, carry state."""
    from .rules import published_halves
    score = np.zeros((n, 2))
    for t in tags:
        pt = prior_team(t)
        if pt is None:
            continue
        w = 1. if conf is None else conf.get(t["code"], 0.)
        score[max(0, t["start"]):min(n, t["end"]), pt] += w
    pred = np.full(n, -1, np.int8)
    for a, b in published_halves():
        state = D
        for i in range(a, b):
            s0, s1 = score[i]
            if s0 > s1:
                state = H
            elif s1 > s0:
                state = A
            pred[i] = state
    for t in tags:
        if t["code"] == "SET PIECES":
            pred[max(0, t["start"] - 3 * FPS):t["start"]] = np.where(pred[max(0, t["start"] - 3 * FPS):t["start"]] >= 0, D, -1)
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["inventory", "baseline"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--package", type=Path, default=ROOT / "models/rondo_r5")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from src.target_io import read_target
    geo = read_target(a.data)
    n = len(geo["ball"])
    tags = load(a.data)
    if a.cmd == "inventory":
        inv, summary = inventory(tags, n)
        cons = consistency(tags, geo, n)
        rep = dict(map_version=MAP_VERSION, semantic_map=SEMANTIC, inventory=inv, summary=summary, consistency=cons)
    else:
        from .diagnose_f3 import package_parts, package_pred
        inv = json.loads((ROOT / "reports/rondo/a0_tag_inventory.json").read_text(encoding="utf-8"))
        conf = {k: v["confidence"] for k, v in inv["consistency"].items() if not v["demoted_to_unknown"]}
        base = tags_only(tags, n)
        S, E, man, _ = package_parts(geo, a.package)
        from starter.load_smart_tagging import load as load_raw
        f4 = package_pred(geo, S, E, man, load_raw(data_dir=str(a.data)))
        sc = scored_mask(n)
        live = sc & np.isin(f4, [H, A]) & np.isin(base, [H, A])
        by = {}
        for t in tags:
            pt = prior_team(t)
            if pt is None:
                continue
            seg = slice(max(0, t["start"]), min(n, t["end"]))
            m = live[seg]
            r = by.setdefault(t["code"], [0, 0])
            r[0] += int((f4[seg][m] == pt).sum()); r[1] += int(m.sum())
        rep = dict(note="aggregate agreement between the tags-only baseline and F4 (no ground truth)",
                   live_team_agreement=round(float((f4[live] == base[live]).mean()), 4),
                   f4_live_share_covered_by_prior_phase=round(float(live.sum() / max(1, (sc & np.isin(f4, [H, A])).sum())), 4),
                   f4_agreement_with_phase_prior_by_code={k: dict(frames=v[1], agreement=round(v[0] / max(1, v[1]), 4)) for k, v in sorted(by.items())},
                   dead_share=dict(tags_only=round(float((base[sc] == D).mean()), 4), F4=round(float((f4[sc] == D).mean()), 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
