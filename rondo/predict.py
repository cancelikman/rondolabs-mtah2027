"""Produce the submission CSV for the competition match from a frozen package.

    python -m rondo.predict --package models/rondo_final --data data --out <team_name>.csv

Deterministic on CPU (fixed threads, deterministic torch). Every setting is read
from <package>/manifest.json; nothing here is tuned on the competition match.
The script prints only the output path and its SHA-256, never frame-level data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package_posteriors(geometry, package, manifest):
    from .candidates import Candidate
    os.environ["RONDO_THREADS"] = str(int(manifest.get("threads", 2)))
    members = manifest.get("members") or [manifest]
    states, events, weights = [], [], []
    for m in members:
        weights.append(float(m.get("weight", 1.)))
        for fname, expected in m.get("sha256", {}).items():
            actual = hashlib.sha256((package / fname).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("weight file %s does not match manifest sha256" % fname)
        cand = Candidate(m.get("name", "member"), checkpoint=package / m["tcn"], parent=package,
                         hgb=package / m["hgb"], tcn_weight=float(m.get("tcn_weight", .5)),
                         hgb_when_hidden=float(m.get("hgb_when_hidden", 0.)))
        state, event = cand.combine(cand.networks(geometry), geometry)
        states.append(state)
        events.append(event)
    if all(w == weights[0] for w in weights):
        return np.mean(states, axis=0), np.mean(events, axis=0)   # F0/F1: unchanged arithmetic
    w = np.asarray(weights) / np.sum(weights)
    return (np.tensordot(w, np.asarray(states), axes=1).astype(np.float32),
            np.tensordot(w, np.asarray(events), axes=1).astype(np.float32))


def apply_phase_tags(state, geometry, data_dir, cfg):
    """Round 6 A2: Smart Tagging phases as a soft team prior (manifest "tags" block)."""
    from . import phase_prior as pp
    from .tags import load as load_tags
    semantic, conf = cfg["semantic_map"], cfg["confidence"]
    phases = []
    for t in load_tags(data_dir):
        cls = semantic.get(t["code"], "UNKNOWN")
        if t["team"] is None or cls not in ("POSSESSING", "DEFENDING") or not conf.get(t["code"]):
            continue
        team = t["team"] if cls == "POSSESSING" else 1 - t["team"]
        phases.append((t["start"], t["end"], team, float(conf[t["code"]])))
    signal = pp.phase_signal(len(state), phases, 25., float(cfg.get("ramp_s", 1.)))
    return pp.apply_phase_bias(state, signal, float(cfg["phase_bias"]))


def apply_stack(state, event, geometry, package, cfg):
    """Round 9 A2m: multi-source correction model on the blended posteriors (manifest "stack" block)."""
    import joblib
    from .twin import stack_state, stack_features
    path = package / cfg["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
        raise ValueError("stack file does not match manifest sha256")
    est = joblib.load(path)["estimator"]
    if cfg.get("control_memory"):          # Round 10 packages (F7); F6 and earlier never enter here
        from .r10 import features as r10_features, state as r10_state
        st = r10_state(est, r10_features(geometry, state, event, True, cfg["control_memory"]), True)
    else:
        st = stack_state(est, stack_features(geometry, state, event), state, bool(cfg.get("team_only", False)))
    a = float(cfg["alpha"])
    return (a * st + (1 - a) * np.asarray(state, float)).astype(np.float32)


def run(package, data_dir):
    from lab.model import decode
    from src.target_io import read_target
    from . import rules
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    geometry = read_target(Path(data_dir))
    state, event = package_posteriors(geometry, package, manifest)
    if manifest.get("stack"):
        state = apply_stack(state, event, geometry, package, manifest["stack"])
    if manifest.get("gnn"):                     # Round 12 packages only; F6 and earlier never enter here
        from .r12 import apply_gnn
        state = apply_gnn(state, geometry, package, manifest["gnn"])
    state = rules.apply_evidence(state, geometry, manifest.get("evidence", []), manifest.get("factor", 20.))
    tags_cfg = manifest.get("tags", {})
    if tags_cfg.get("phase_bias"):
        state = apply_phase_tags(state, geometry, data_dir, tags_cfg)
    if manifest.get("event_detector"):          # Round 11 packages only; F6 and earlier never enter here
        from .r11 import apply_detector
        event = apply_detector(state, event, geometry, package, manifest["event_detector"])
    pred = decode(state, event, geometry, manifest["decoder"])
    post = manifest.get("post", {})
    pred = rules.apply_post(pred, geometry, post)
    if post.get("kickoff_tags"):
        from starter.load_smart_tagging import load as load_tags
        tags = load_tags(data_dir=str(data_dir))
        pred = rules.apply_tag_rules(pred, post["kickoff_tags"], tags, rules.published_halves())
    return pred


def main():
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=ROOT / "models/rondo_final")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from src.target_io import write_labels
    pred = run(a.package, a.data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_labels(a.out, pred)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print("wrote %s sha256 %s" % (a.out, digest))


if __name__ == "__main__":
    main()
