"""Independently implemented external-only directed transition experiment.

Motivation: public competitor source research suggests separating rare switch
evidence from frame ownership. No competitor code, constants or weights are used.
Event features contain geometry only, so in-sample upstream probabilities cannot
leak into event training. WOH/WN1 train, WOY development; WPY is never opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_VERSION, build_features, blocks_of, occlude_external
from src.train import load_external, metrics, fingerprint

VERSION = "directed-transition-geometry-v1"
EVENT_RADIUS_FRAMES = 2
TRAIN_MATCHES = ("J03WOH", "J03WN1")
ALLOWED_EXTERNAL_TRAIN = ("J03WOH", "J03WN1", "J03WMX", "J03WQQ")
VALIDATION_MATCH = "J03WOY"
SEED = 20261103


def event_targets(data, radius=EVENT_RADIUS_FRAMES):
    """0=no event; 1/2/3=new H/A/D in a symmetric boundary neighbourhood."""
    labels = np.asarray(data["labels"])
    target = np.where(labels >= 0, 0, -1).astype(np.int8)
    centers = []
    distance = np.full(len(labels), radius+1, dtype=np.int16)
    for a, b in blocks_of(data):
        valid = (labels[a+1:b] >= 0) & (labels[a:b-1] >= 0)
        changes = np.flatnonzero(valid & (labels[a+1:b] != labels[a:b-1]))+a+1
        centers.extend(changes.tolist())
        for center in changes:
            index = np.arange(max(a, center-radius), min(b, center+radius+1))
            nearer = (np.abs(index-center) < distance[index]) & (labels[index] >= 0)
            index = index[nearer]
            target[index] = labels[center]+1
            distance[index] = np.abs(index-center)
    return target, np.asarray(centers, dtype=np.int64)


def swap_feature_matrix(x, names):
    index = {name: i for i, name in enumerate(names)}
    columns = []
    for name in names:
        counterpart = name.replace("team0", "team1") if "team0" in name else name.replace("team1", "team0")
        columns.append(index[counterpart])
    result = x[:, columns].copy()
    for i, name in enumerate(names):
        if name.startswith("distance_advantage_home"):
            result[:, i] *= -1
    return result


def swap_event_targets(values):
    result = values.copy()
    result[values == 1], result[values == 2] = 2, 1
    return result


def predict_events(model, x, names):
    if names != model["feature_names"] or not np.array_equal(model["classifier"].classes_, [0, 1, 2, 3]):
        raise ValueError("event feature/class schema mismatch")
    estimator = model["classifier"]
    return (estimator.predict_proba(x) +
            estimator.predict_proba(swap_feature_matrix(x, names))[:, [0, 2, 1, 3]])*.5


def frame_probabilities(model, x, names):
    if model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("frame feature implementation mismatch")
    index = {name: i for i, name in enumerate(names)}
    chosen = [index[name] for name in model["feature_names"]]
    estimator = model["classifier"]
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("invalid frame class order")
    original = estimator.predict_proba(x[:, chosen])
    swapped = estimator.predict_proba(swap_feature_matrix(x, names)[:, chosen])[:, [1, 0, 2]]
    return (original+swapped)*.5


def directed_viterbi(frame_probability, event_probability, team_cost, dead_cost,
                     strength, temperature=1., minimum_cost=.1):
    """Three-state DP with bounded, observation-dependent incoming-state costs.

    An event's log contrast against no-event can lower a switching penalty. The
    diagonal stays zero; every off-diagonal remains positive. The specialist can
    relax a transition, but cannot create a negatively rewarded flicker cycle.
    """
    p, q = np.asarray(frame_probability, float), np.asarray(event_probability, float)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4):
        raise ValueError("expected frame[n,3], event[n,4]")
    if (not np.isfinite(p).all() or not np.isfinite(q).all() or
            (p < 0).any() or (q < 0).any() or temperature <= 0 or
            min(team_cost, dead_cost, strength, minimum_cost) < 0):
        raise ValueError("invalid probabilities or transition settings")
    if not len(p):
        return np.zeros(0, dtype=np.int8)
    base = np.full((3, 3), float(team_cost))
    base[2, :2] = base[:2, 2] = float(dead_cost)
    np.fill_diagonal(base, 0.)
    contrast = np.maximum(0., np.log(np.clip(q[:, 1:], 1e-7, 1))-
                          np.log(np.clip(q[:, :1], 1e-7, 1)))
    # Match src.model.viterbi's emission floor exactly for the zero-reward ablation.
    emission = -np.log(np.clip(p, 1e-6, 1))/temperature
    score = emission[0]
    back = np.zeros((len(p), 3), dtype=np.int8)
    for t in range(1, len(p)):
        cost = np.maximum(minimum_cost, base-strength*contrast[t][None, :]) if strength else base.copy()
        np.fill_diagonal(cost, 0.)
        paths = score[:, None]+cost
        parent = np.argmin(paths, axis=0)
        back[t] = parent
        score = paths[parent, np.arange(3)]+emission[t]
        score -= score.min()
    result = np.empty(len(p), dtype=np.int8)
    result[-1] = np.argmin(score)
    for t in range(len(p)-1, 0, -1):
        result[t-1] = back[t, result[t]]
    return result


def decode(frame_probability, event_probability, data, config):
    prediction = np.full(len(frame_probability), -1, dtype=np.int8)
    stride = int(config.get("stride", 5))
    if stride < 1:
        raise ValueError("decoder stride must be positive")
    for a, b in blocks_of(data):
        starts = np.arange(a, b, stride)
        frames = np.asarray([frame_probability[i:min(i+stride, b)].mean(axis=0) for i in starts])
        events = np.asarray([event_probability[i:min(i+stride, b)].mean(axis=0) for i in starts])
        labels = directed_viterbi(frames, events, config["team_cost"], config["dead_cost"],
                                  config["event_strength"], config["temperature"])
        prediction[a:b] = np.repeat(labels, stride)[:b-a]
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frame-model", type=Path, required=True)
    parser.add_argument("--train-match", action="append", choices=ALLOWED_EXTERNAL_TRAIN,
                        help="explicit external training matches; default is original WOH/WN1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train_matches = tuple(args.train_match or TRAIN_MATCHES)
    if len(train_matches) != len(set(train_matches)):
        raise ValueError("duplicate external training match")
    if (args.out/"training_report.json").exists():
        raise FileExistsError("use a fresh experiment directory; old reports stay frozen")
    args.out.mkdir(parents=True, exist_ok=True)
    base_model = joblib.load(args.frame_model)
    provenance = base_model["provenance"]
    train_ids = provenance["train"] if isinstance(provenance["train"], list) else [provenance["train"]]
    if set(train_ids) != set(train_matches) or provenance["validation"] != VALIDATION_MATCH:
        raise ValueError("frozen frame model must use only the declared external train/validation")
    start = monotonic()
    parts, targets = [], []
    counts = {}
    for match_id in train_matches:
        data = load_external(args.data_dir / (match_id+"_external_v1.npz"))
        y, boundaries = event_targets(data)
        counts[match_id] = {"boundaries": len(boundaries), "classes": np.bincount(y[y >= 0], minlength=4).tolist()}
        for condition, current in (("clean", data), ("occluded", occlude_external(data, 101))):
            x, names = build_features(current, profile="contact_motion")
            extra = np.clip(boundaries[:, None]+np.arange(-2, 3), 0, len(y)-1).ravel()
            indices = np.unique(np.r_[np.arange(0, len(y), 5), extra])
            indices = indices[y[indices] >= 0]
            sampled = x[indices]
            parts.extend([sampled, swap_feature_matrix(sampled, names)])
            targets.extend([y[indices], swap_event_targets(y[indices])])
            print(f"event features {match_id} {condition}: {sampled.shape}", flush=True)
    x, y = np.concatenate(parts), np.concatenate(targets)
    del parts, targets
    frequencies = np.bincount(y, minlength=4)
    sample_weights = len(y)/(4*np.maximum(frequencies, 1))
    estimator = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,
        learning_rate=.08, l2_regularization=2., early_stopping=False, random_state=SEED)
    with threadpool_limits(limits=4):
        estimator.fit(x, y, sample_weight=sample_weights[y])
    event_model = {"classifier": estimator, "feature_names": names, "feature_profile": "contact_motion",
        "version": VERSION, "event_radius_frames": EVENT_RADIUS_FRAMES,
        "provenance": {"train": list(train_matches), "validation": VALIDATION_MATCH,
                       "geometry_only_event_inputs": True, "seed": SEED}}
    joblib.dump(event_model, args.out/"event_classifier.joblib", compress=3)
    del x, y
    report = {"version": VERSION, "train_matches": list(train_matches), "validation_match": VALIDATION_MATCH,
        "frame_model_sha256": fingerprint(args.frame_model), "event_training_counts": counts,
        "event_classifier_sha256": fingerprint(args.out/"event_classifier.joblib"),
        "emission_floor": 1e-6,
        "numerical_contract": "zero-event reward exactly matches original Viterbi at temperature1",
        "event_classes": ["none", "newH", "newA", "newD"], "event_radius_frames": EVENT_RADIUS_FRAMES,
        "feature_profile": "contact_motion", "seed": SEED,
        "selection": "mean clean and occluded WOY official spellF1, then macroF1",
        "target_or_heldout_access": False, "trials": []}
    print(f"event classifier saved in {monotonic()-start:.1f}s", flush=True)
    data = load_external(args.data_dir/(VALIDATION_MATCH+"_external_v1.npz"))
    validation = [data, occlude_external(data, 202)]
    frame_p, event_p = [], []
    with threadpool_limits(limits=4):
        for condition, current in zip(("clean", "occluded"), validation):
            x, names = build_features(current, profile="contact_motion")
            frame_p.append(frame_probabilities(base_model, x, names))
            event_p.append(predict_events(event_model, x, names))
            print(f"validation {condition} frame+event probabilities saved", flush=True)
    np.savez_compressed(args.out/"validation_probabilities.npz", frame_clean=frame_p[0],
        frame_occluded=frame_p[1], event_clean=event_p[0], event_occluded=event_p[1])
    best = None
    for team_cost in (2., 4.):
        for strength in (0., .5, 1., 2.):
            for temperature in (.7, 1., 1.4):
                config = {"team_cost": team_cost, "dead_cost": 6., "event_strength": strength,
                          "temperature": temperature, "stride": 5}
                scores = [metrics(v, decode(p, q, v, config)) for v, p, q in zip(validation, frame_p, event_p)]
                key = (float(np.mean([s["spell_f1_mean"] for s in scores])),
                       float(np.mean([s["macro_f1"] for s in scores])))
                trial = {"decoder": config, "selection_spell": key[0], "selection_macro": key[1],
                         "clean": scores[0], "occluded": scores[1]}
                report["trials"].append(trial)
                if best is None or key > best:
                    best = key
                    report["selected"] = trial
                (args.out/"training_report.json").write_text(json.dumps(report, indent=2)+"\n")
    report["elapsed_seconds"] = monotonic()-start
    (args.out/"training_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"selected": report["selected"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
