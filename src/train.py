"""Train/tune on external matches only. Target files are not accepted here."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .features import FEATURE_VERSION, build_features, swap_sides, occlude_external, blocks_of
from .model import predict_probabilities, decode
from .classifiers import TwoHeadClassifier
from .symmetry import swap_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm


def load_external(path):
    with np.load(path, allow_pickle=False) as f:
        data = dict(f)
    # The source is a provenance guard, not a substitute for the documented licence.
    if str(data.get("source", "")) != "IDSSE":
        raise ValueError("training accepts only declared external IDSSE arrays")
    if str(data.get("match_id", "")) in ("", "BAR-ATH"):
        raise ValueError("external match id required")
    labels = np.asarray(data["labels"])
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError("invalid external labels")
    return data


def metrics(data, pred, bootstrap_n=0):
    gt = np.asarray(data["labels"]).copy()
    gt[gt < 0] = pm.X
    # Insert an ungraded separator across discontinuities. Do not overwrite a
    # real external frame: every originally labelled frame must be evaluated.
    blocks = blocks_of(data)
    p = np.where(pred < 0, pm.X, pred)
    cuts = blocks[1:, 0]
    if len(cuts):
        gt = np.insert(gt, cuts, pm.X)
        p = np.insert(p, cuts, pm.X)
    result = pm.evaluate(gt, p, bootstrap_n=bootstrap_n)
    keys = ["macro_f1", "spell_f1_mean", "accuracy",
        "alive_dead_f1", "team_acc_gt_alive", "n_gt_spells", "n_pred_spells", "per_class"]
    if bootstrap_n:
        keys += ["bootstrap_low", "bootstrap_high", "spell_bootstrap_low", "spell_bootstrap_high"]
    return {key: result[key] for key in keys}


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spell_weights(data, power):
    labels = np.asarray(data["labels"])
    weights = np.ones(len(labels), dtype=float)
    if not power:
        return weights
    for a, b in blocks_of(data):
        edges = np.r_[a, a+np.flatnonzero(labels[a+1:b] != labels[a:b-1])+1, b]
        lengths = np.diff(edges)
        typical = float(np.median(lengths))
        for start, end in zip(edges[:-1], edges[1:]):
            weights[start:end] = min(5., (typical / (end-start)) ** power)
    return weights / weights.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--profile", choices=["geometry", "contact", "contact_motion", "contact_reach", "contact_shape", "contact_motion_shape"], default="geometry")
    p.add_argument("--segment-weight", type=float, choices=[0., .5, 1.], default=0.)
    p.add_argument("--leaves", type=int, nargs="+", default=[15, 31])
    p.add_argument("--two-head", action="store_true")
    p.add_argument("--iterations", type=int, default=180)
    p.add_argument("--min-leaf", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=.08)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trains = [load_external(path) for path in args.train]
    validation = load_external(args.validation)
    train_ids = [str(train["match_id"]) for train in trains]
    if len(set(train_ids)) != len(train_ids) or str(validation["match_id"]) in train_ids:
        raise ValueError("train and validation must be different matches")
    start = monotonic()
    report = {"feature_version": FEATURE_VERSION, "feature_profile": args.profile,
              "train_match": train_ids, "validation_match": str(validation["match_id"]),
              "train_sha256": [fingerprint(path) for path in args.train], "validation_sha256": fingerprint(args.validation),
              "segment_weight": args.segment_weight,
              "two_head": args.two_head,
              "iterations": args.iterations, "min_samples_leaf": args.min_leaf,
              "learning_rate": args.learning_rate,
              "selection": "mean external clean and synthetic-occlusion spellF1, then macroF1",
              "seed": 20261103, "trials": []}
    xparts, yparts, wparts = [], [], []
    names = None
    # All augmentation choices are fixed in code before any competition inference.
    for train in trains:
        for variant, data in enumerate((train, occlude_external(train, 101))):
            x, names = build_features(data, profile=args.profile)
            indices = np.arange(0, len(x), 5)
            if args.segment_weight:
                # Preserve very brief external controls near transition frames.
                changes = np.flatnonzero(np.diff(data["labels"]) != 0)+1
                boundaries = np.clip(changes[:, None]+np.arange(-2, 3), 0, len(x)-1).ravel()
                indices = np.unique(np.r_[indices, boundaries])
            indices = indices[data["labels"][indices] >= 0]
            sampled = x[indices]
            labels = data["labels"][indices]
            current_weights = spell_weights(data, args.segment_weight)[indices]
            for swapped in (False, True):
                if swapped:
                    flipped = labels.copy()
                    flipped[labels == 0] = 1
                    flipped[labels == 1] = 0
                    xparts.append(swap_feature_matrix(sampled, names))
                    yparts.append(flipped)
                else:
                    xparts.append(sampled)
                    yparts.append(labels)
                wparts.append(current_weights)
            print("external %s feature stage %d/2 complete" % (str(train["match_id"]), variant+1), flush=True)
    x, y = np.concatenate(xparts), np.concatenate(yparts)
    segment_weight = np.concatenate(wparts)
    del xparts, yparts, wparts
    val_sets = [validation, occlude_external(validation, 202)]
    best_key, best_model = None, None
    with threadpool_limits(limits=args.threads):
        for leaves in args.leaves:
            classifier = TwoHeadClassifier if args.two_head else HistGradientBoostingClassifier
            estimator = classifier(learning_rate=args.learning_rate, max_iter=args.iterations,
                max_leaf_nodes=leaves, l2_regularization=1, random_state=20261103,
                early_stopping=False, min_samples_leaf=args.min_leaf)
            counts = np.bincount(y.astype(int), minlength=3)
            weights = len(y) / (3 * np.maximum(counts, 1))
            estimator.fit(x, y, sample_weight=weights[y.astype(int)] * segment_weight)
            model = {"classifier": estimator, "feature_names": names,
                     "feature_version": FEATURE_VERSION, "feature_profile": args.profile}
            joblib.dump(model, out / ("classifier_leaves%d.joblib" % leaves), compress=3)
            probabilities = [predict_probabilities(model, v) for v in val_sets]
            np.savez_compressed(out / ("validation_leaves%d.npz" % leaves),
                                clean=probabilities[0], occluded=probabilities[1])
            for cost in (0., 1., 2., 4., 8.):
                for dead_cost in (1., 3., 6.):
                    config = {"switch_cost": cost, "dead_switch_cost": dead_cost, "stride": 5}
                    scores = [metrics(v, decode(prob, v, config))
                              for prob, v in zip(probabilities, val_sets)]
                    key = (np.mean([s["spell_f1_mean"] for s in scores]),
                           np.mean([s["macro_f1"] for s in scores]))
                    trial = {"leaves": leaves, "decoder": config,
                             "selection_spell": float(key[0]), "selection_macro": float(key[1]),
                             "clean": scores[0], "occluded": scores[1]}
                    report["trials"].append(trial)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_model = dict(model, decoder=config, provenance={
                            "train": report["train_match"], "validation": report["validation_match"],
                            "training_data": "external IDSSE only", "seed": report["seed"]})
                        report["selected"] = trial
            print("external classifier leaves=%d finished, best spell=%.4f macro=%.4f" %
                  (leaves, *best_key), flush=True)
            (out / "training_report.json").write_text(json.dumps(report, indent=2))
    report["elapsed_seconds"] = monotonic() - start
    joblib.dump(best_model, out / "model.joblib", compress=3)
    report["model_sha256"] = fingerprint(out / "model.joblib")
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"external_validation": report["selected"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
