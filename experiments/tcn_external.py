"""Small residual temporal network, trained/tuned entirely on external IDSSE.

Protocol: train J03WOH + J03WN1, select epochs/decoder on J03WOY clean and
synthetically occluded. This program refuses other IDs and never opens the
provided match, its video, public reference, or the held-out J03WPY test.
An explicitly selected expanded protocol also permits training J03WMX/J03WQQ;
both J03WPY and the additional untouched J03WR9 remain inaccessible here.

PyTorch pinned and verified against official v2.15.0 release:
https://github.com/pytorch/pytorch/releases/tag/v2.15.0
https://docs.pytorch.org/docs/2.15/notes/mps.html
Residual dilated-convolution motivation: Bai, Kolter and Koltun (2018),
https://arxiv.org/abs/1803.01271 (this is a small noncausal adaptation).
CPU inference from a frozen checkpoint is the reproduction path. Training on
MPS is optional; identical retraining is not a competition requirement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_VERSION, blocks_of, build_features, occlude_external
from src.model import decode
from src.symmetry import _swap_plan
from src.train import load_external, metrics, fingerprint, spell_weights

SEED = 20261103
STRIDE = 2
CONTEXT = 63  # Six kernel-3 layers with dilation 1/2/4/8/16/32.
PROFILE = "contact"
TRAIN_IDS = ("J03WOH", "J03WN1")
ALLOWED_TRAIN_IDS = (*TRAIN_IDS, "J03WMX", "J03WQQ")
VALIDATION_ID = "J03WOY"
TRACK_VIEW_PARAMETERS = {"fragment_mean_seconds": 30., "dropout_probability": .1,
                         "dropout_mean_seconds": .5, "n_slots_per_team": 23}


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.point = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        return x + self.drop(self.point(F.gelu(self.conv(x))))


class TemporalNet(nn.Module):
    def __init__(self, input_channels, channels=32):
        super().__init__()
        self.project = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(ResidualBlock(channels, dilation)
                                    for dilation in (1, 2, 4, 8, 16, 32))
        self.head = nn.Conv1d(channels, 3, 1)

    def forward(self, x):
        x = F.gelu(self.project(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def exchange_schema(names):
    """Exact raw-feature team exchange, before external training normalisation."""
    permutation, odd = _swap_plan(tuple(names))
    sign = np.ones(len(names), dtype=np.float32)
    sign[odd] = -1.
    return permutation, sign


def exchange(x, permutation, sign):
    return x[:, permutation] * sign


def transform(x, mean, scale):
    valid = np.isfinite(x)
    filled = np.where(valid, x, mean)
    normal = np.clip((filled - mean) / scale, -8., 8.)
    return np.concatenate([normal, valid.astype(np.float32)], axis=1).astype(np.float32)


def implementation_fingerprint():
    digest = hashlib.sha256()
    modules = ["features.py", "contact_features.py"]
    if PROFILE == "contact_motion":
        modules += ["motion_features.py"]
    if PROFILE == "contact_shape":
        modules += ["motion_features.py", "team_shape_features.py"]
    for name in modules:
        digest.update((ROOT / "src" / name).read_bytes())
    return digest.hexdigest()[:16]


def check_cache_equivalence(data, cached, names, profile=None):
    """Recompute interior external probes after benign code-dispatch changes.

    Each probe has >20 s context on both sides, covering the longest capped
    contact feature. This is a sampled semantic check, not a full-matrix proof.
    """
    a, b = blocks_of(data)[0]
    if b-a < 2000:
        raise ValueError("External recording too short for reviewed-cache probes")
    start = min(int(a)+10000, int(b)-2000)
    end = start+2000
    subset = {key:data[key] for key in ("fps", "pitch", "team_of_track")}
    subset.update(ball=data["ball"][start:end], players=data["players"][start:end],
                  blocks=np.array([[0, 2000]]))
    current, current_names = build_features(subset, profile=profile or PROFILE)
    if current_names != names:
        raise ValueError("Reviewed-cache feature schema changed")
    probes = np.array([900, 1000, 1100])
    np.testing.assert_allclose(current[probes], cached[start+probes], rtol=1e-5,
                               atol=1e-4, equal_nan=True,
                               err_msg="Reviewed-cache semantic probes changed")


def track_view_spec(match_id):
    """Hash-bound fixed training corruption; WOY never receives this view."""
    if match_id not in ALLOWED_TRAIN_IDS:
        raise ValueError("Anonymous track view is restricted to declared external training IDs")
    return {"version": "training-tracks-plus-ball-occlusion-v1",
            "seed": SEED + 1000 + ALLOWED_TRAIN_IDS.index(match_id),
            "parameters": dict(TRACK_VIEW_PARAMETERS),
            "augmenter_sha256": fingerprint(ROOT / "src" / "track_augmentation.py"),
            "ball_occlusion_seed": 101 + ALLOWED_TRAIN_IDS.index(match_id),
            "ball_occlusion_source_sha256": fingerprint(ROOT / "src" / "features.py")}


def condition_views(base, match_id, player_augmentation=False):
    seed = 202 if match_id == VALIDATION_ID else 101 + ALLOWED_TRAIN_IDS.index(match_id)
    yield "clean", base, None
    occluded = occlude_external(base, seed)
    yield "occluded", occluded, None
    if player_augmentation and match_id != VALIDATION_ID:
        from src.track_augmentation import fragment_tracks
        spec = track_view_spec(match_id)
        augmented = fragment_tracks(occluded, spec["seed"], **spec["parameters"])
        yield "tracks_occluded", augmented, spec


def validate_cache_metadata(metadata, digest, condition, seed, augmentation, profile=None):
    if (metadata.get("source") != "external IDSSE" or metadata.get("source_sha256") != digest
            or metadata.get("condition") != condition
            or metadata.get("seed") != (seed if condition != "clean" else None)
            or metadata.get("feature_profile", "contact") != (profile or PROFILE)
            or not isinstance(metadata.get("implementation"), str)
            or not metadata.get("implementation")
            or metadata.get("player_augmentation") != augmentation):
        raise ValueError("External feature-cache provenance/corruption mismatch")


def frozen_normalisation(path, names, sources, train_ids, profile):
    """Reuse an externally fitted transform, never fitting validation/corruption.

    The original checkpoint and protocol must agree on IDs, raw sources, schema
    and preprocessing. A checkpoint from target/held-out/other training cannot be
    used as a convenient unverified mean/scale file.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    protocol_path = path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected = set(train_ids)
    checkpoint_ids = checkpoint.get("train_matches", [])
    if (not expected or len(train_ids) != len(expected) or expected-set(ALLOWED_TRAIN_IDS)
            or set(sources) != expected | {VALIDATION_ID}
            or len(checkpoint_ids) != len(expected) or set(checkpoint_ids) != expected
            or len(protocol.get("train_matches", [])) != len(expected)
            or set(protocol.get("train_matches", [])) != expected
            or checkpoint.get("validation_match") != VALIDATION_ID
            or protocol.get("validation_match") != VALIDATION_ID
            or protocol.get("test_opened") is not False or protocol.get("target_opened") is not False
            or protocol.get("normalisation_lineage", {}).get("validation_used_for_fit", False) is not False
            or checkpoint.get("feature_names") != names
            or checkpoint.get("feature_profile") != profile or protocol.get("feature_profile") != profile
            or checkpoint.get("source_sha256") != sources or protocol.get("source_sha256") != sources):
        raise ValueError("Frozen preprocessing must have identical reviewed external source/schema provenance")
    caches = protocol.get("feature_caches", [])
    if any(not any(r.get("match_id") == match_id and r.get("condition") == condition
                   for r in caches) for match_id in expected for condition in ("clean", "occluded")):
        raise ValueError("Frozen preprocessing lacks original training-cache provenance")
    mean, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
    if (mean.shape != (len(names),) or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Frozen preprocessing contains invalid mean/scale")
    saved_path = path.parent / "external_preprocessing.npz"
    permutation, sign = exchange_schema(names)
    with np.load(saved_path, allow_pickle=False) as saved:
        for key, value in (("mean",mean),("scale",scale),("permutation",permutation),("sign",sign)):
            if not np.array_equal(saved[key], value):
                raise ValueError("Frozen preprocessing checkpoint/sidecar mismatch")
    lineage = {"mode": "frozen-external-training-transform", "checkpoint": str(path),
               "checkpoint_sha256": fingerprint(path), "protocol_sha256": fingerprint(protocol_path),
               "preprocessing_sha256": fingerprint(saved_path), "train_matches": list(train_ids),
               "source_sha256": {key: sources[key] for key in train_ids},
               "validation_used_for_fit": False, "new_corruption_used_for_fit": False}
    return mean, scale, permutation, sign, lineage


def validate_warm_source(checkpoint, sources, train_ids, profile):
    """Allow verified external subset weights; retain exact transform checks."""
    training = set(checkpoint.get("train_matches", []))
    previous_sources = checkpoint.get("source_sha256", {})
    if (not training or training-set(ALLOWED_TRAIN_IDS) or training-set(train_ids)
            or checkpoint.get("feature_profile") != profile
            or checkpoint.get("validation_match") != VALIDATION_ID
            or set(previous_sources) != training | {VALIDATION_ID}
            or any(previous_sources[key] != sources.get(key) for key in previous_sources)):
        raise ValueError("Warm-start must have matching reviewed external source/schema provenance")


def prepare(data_dir, cache_dir, frozen_signature=None, train_ids=TRAIN_IDS, reuse_cache=None,
            segment_power=.5, player_augmentation=False):
    """Disk-backed external caches; no target files or fitted target transform."""
    train_ids = tuple(train_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids) or set(train_ids)-set(ALLOWED_TRAIN_IDS):
        raise ValueError("Only unique predeclared external training IDs are accepted")
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = frozen_signature or implementation_fingerprint()
    reuse_roots = [] if reuse_cache is None else ([reuse_cache] if isinstance(reuse_cache, Path) else list(reuse_cache))
    records, validation, names, sources = [], [], None, {}
    for match_id in (*train_ids, VALIDATION_ID):
        path = data_dir / (match_id + "_external_v1.npz")
        base = load_external(path)
        if str(base["match_id"]) != match_id:
            raise ValueError("External input filename/id mismatch")
        digest = fingerprint(path)
        sources[match_id] = digest
        seed = 202 if match_id == VALIDATION_ID else 101 + ALLOWED_TRAIN_IDS.index(match_id)
        for condition, data, augmentation in condition_views(base, match_id, player_augmentation):
            # A new corruption cannot claim a legacy frozen implementation ID.
            view_signature = implementation_fingerprint() if augmentation else signature
            corruption = ("_" + hashlib.sha256(json.dumps(augmentation, sort_keys=True).encode()).hexdigest()[:16]
                          if augmentation else "")
            stem = cache_dir / (match_id + "_" + condition + "_" + PROFILE + "_" + view_signature + corruption + "_" + digest[:12])
            x_path, metadata_path = stem.with_suffix(".npy"), stem.with_suffix(".json")
            reused = False
            if reuse_roots:
                for folder in (*reuse_roots, cache_dir):
                    candidates = sorted(folder.glob(match_id + "_" + condition + "_*_" + digest[:12] + ".npy"))
                    reviewed = []
                    for path in candidates:
                        meta = json.loads(path.with_suffix(".json").read_text())
                        if meta.get("feature_profile", "contact") != PROFILE:
                            continue
                        # Wrong source/condition/seed is never silently trusted.
                        # Another explicitly bound augmentation is a distinct cache.
                        if meta.get("player_augmentation") != augmentation:
                            if condition == "tracks_occluded":
                                continue
                            raise ValueError("External feature-cache augmentation mismatch")
                        validate_cache_metadata(meta, digest, condition, seed, augmentation)
                        reviewed.append(path)
                    candidates = reviewed
                    if candidates:
                        # Preserve the existing filename and metadata: do not claim an old
                        # cache was computed by a new implementation with the same schema.
                        x_path = candidates[0]
                        metadata_path = x_path.with_suffix(".json")
                        reused = True
                        break
            if not x_path.exists() or not metadata_path.exists():
                if frozen_signature and not augmentation:
                    raise ValueError("Frozen cache missing; refusing to fabricate its implementation provenance")
                component = None
                if PROFILE in {"contact_shape", "contact_motion"}:
                    # Reuse a reviewed contact component rather than rebuilding it.
                    for folder in reuse_roots:
                        for path in sorted(folder.glob(match_id + "_" + condition + "_*_" + digest[:12] + ".npy")):
                            meta = json.loads(path.with_suffix(".json").read_text())
                            if meta.get("feature_profile", "contact") == "contact":
                                if meta.get("player_augmentation") != augmentation:
                                    continue
                                validate_cache_metadata(meta, digest, condition, seed, augmentation, "contact")
                                component = (path, meta)
                                break
                        if component is not None:
                            break
                if component is not None:
                    base_path, base_metadata = component
                    contact = np.load(base_path, mmap_mode="r")
                    check_cache_equivalence(data, contact, base_metadata["feature_names"], profile="contact")
                    if PROFILE == "contact_shape":
                        from src.team_shape_features import team_shape_features
                        extra, extra_names = team_shape_features(data)
                    else:
                        from src.motion_features import motion_features
                        extra, extra_names = motion_features(data)
                    current_names = base_metadata["feature_names"] + extra_names
                    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                        shape=(len(contact), len(current_names)))
                    x[:, :contact.shape[1]] = contact
                    x[:, contact.shape[1]:] = extra
                    x.flush()
                    component = {"contact_cache": str(base_path),
                        "contact_implementation": base_metadata["implementation"],
                        "contact_sampled_semantic_equivalence": True,
                        "motion_dependency_sha256": fingerprint(ROOT / "src" / "motion_features.py")}
                    if PROFILE == "contact_shape":
                        component["shape_source_sha256"] = fingerprint(ROOT / "src" / "team_shape_features.py")
                    del extra
                else:
                    x, current_names = build_features(data, profile=PROFILE)
                    np.save(x_path, x)
                metadata_path.write_text(json.dumps({"feature_names": current_names,
                    "feature_profile": PROFILE, "components": component,
                    "source": "external IDSSE", "source_sha256": digest,
                    "implementation": view_signature, "condition": condition,
                    "seed": seed if condition != "clean" else None,
                    "player_augmentation": augmentation}, indent=2) + "\n")
                del x
            metadata = json.loads(metadata_path.read_text())
            validate_cache_metadata(metadata, digest, condition, seed, augmentation)
            if names is None:
                names = metadata["feature_names"]
            if names != metadata["feature_names"]:
                raise ValueError("External cache feature schema mismatch")
            x = np.load(x_path, mmap_mode="r")
            if x.shape != (len(base["ball"]), len(names)) or x.dtype != np.float32:
                raise ValueError("External feature-cache matrix shape/dtype mismatch")
            if reused:
                check_cache_equivalence(data, x, names)
            weights = spell_weights(base, segment_power).astype(np.float32)
            # Boundary weighting comes only from permitted external labels.
            for a, b in blocks_of(base):
                changes = a + np.flatnonzero(base["labels"][a+1:b] != base["labels"][a:b-1]) + 1
                for change in changes:
                    weights[max(a, change-5):min(b, change+6)] *= 2.
            record = {"x": x, "data": base, "weights": weights,
                      "condition": condition, "match_id": match_id,
                      "cache": str(x_path), "cache_implementation": metadata["implementation"],
                      "sampled_cache_equivalence": reused,
                      "player_augmentation": augmentation}
            if match_id == VALIDATION_ID:
                validation.append(record)
            else:
                records.append(record)
            print(json.dumps({"cached_external": match_id, "condition": condition,
                              "features": int(x.shape[1]), "frames": int(x.shape[0])}), flush=True)
    return records, validation, names, sources


def fit_normalisation(records, names):
    """Fit on training games only; exchange augmentation is included analytically."""
    if not records or any(record.get("match_id") not in ALLOWED_TRAIN_IDS for record in records):
        raise ValueError("Normalisation accepts external training records only")
    permutation, sign = exchange_schema(names)
    n_features = len(names)
    count = np.zeros(n_features)
    total = np.zeros(n_features)
    square = np.zeros(n_features)
    for record in records:
        for a, b in blocks_of(record["data"]):
            x = np.asarray(record["x"][a:b:STRIDE])
            for current in (x, exchange(x, permutation, sign)):
                valid = np.isfinite(current)
                value = np.where(valid, current, 0).astype(np.float64)
                count += valid.sum(axis=0)
                total += value.sum(axis=0)
                square += np.square(value).sum(axis=0)
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean**2, 0))
    # Shape fractions/covariances are naturally much smaller than metre features.
    scale = np.maximum(scale, .001 if PROFILE == "contact_shape" else .05)
    return mean.astype(np.float32), scale.astype(np.float32), permutation, sign


def sequence_inventory(records):
    output = []
    for record in records:
        for a, b in blocks_of(record["data"]):
            output.append((record, int(a), int(b), len(range(a, b, STRIDE))))
    return output


def batch(sequences, rng, mean, scale, permutation, sign, batch_size, core):
    # Sampling proportional to recording length; no half can cross a chunk.
    probabilities = np.asarray([seq[3] for seq in sequences], dtype=float)
    probabilities /= probabilities.sum()
    xs, ys, ws = [], [], []
    for _ in range(batch_size):
        record, a, b, length = sequences[rng.choice(len(sequences), p=probabilities)]
        start = rng.integers(max(1, length-core+1))
        positions = np.arange(start-CONTEXT, start+core+CONTEXT)
        valid = (positions >= 0) & (positions < length)
        indices = a + np.clip(positions, 0, length-1) * STRIDE
        raw = np.array(record["x"][indices])
        labels = record["data"]["labels"][indices].copy()
        weights = record["weights"][indices].copy()
        if rng.random() < .5:
            raw = exchange(raw, permutation, sign)
            labels = np.where(labels == 0, 1, np.where(labels == 1, 0, labels))
        x = transform(raw, mean, scale)
        x[~valid] = 0
        labels[~valid] = -1
        weights[~valid] = 0
        xs.append(x.T)
        ys.append(labels[CONTEXT:CONTEXT+core])
        ws.append(weights[CONTEXT:CONTEXT+core])
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)).long(),
            torch.from_numpy(np.stack(ws)))


@torch.no_grad()
def infer(model, record, mean, scale, permutation, sign, device, ownership_only=False):
    """Block-local overlapped inference with exact team-exchange averaging."""
    model.eval()
    n_classes = 2 if ownership_only else 3
    probabilities = np.full((len(record["data"]["ball"]), n_classes), np.nan, dtype=np.float32)
    for a, b in blocks_of(record["data"]):
        indices = np.arange(a, b, STRIDE)
        x = np.asarray(record["x"][indices])
        normal = transform(x, mean, scale)
        flipped = transform(exchange(x, permutation, sign), mean, scale)
        p = np.zeros((len(indices), n_classes), dtype=np.float32)
        for start in range(0, len(indices), 1024):
            end = min(start + 1024, len(indices))
            lo, hi = max(0, start-CONTEXT), min(len(indices), end+CONTEXT)
            window = np.zeros((2, normal.shape[1], end-start+2*CONTEXT), dtype=np.float32)
            offset = CONTEXT + lo - start
            window[0, :, offset:offset+hi-lo] = normal[lo:hi].T
            window[1, :, offset:offset+hi-lo] = flipped[lo:hi].T
            raw = model(torch.from_numpy(window).to(device))[:, :, CONTEXT:CONTEXT+end-start]
            if ownership_only:
                raw = raw[:, :2]
            prob = raw.softmax(dim=1).cpu().numpy().transpose(0, 2, 1)
            order = [1, 0] if ownership_only else [1, 0, 2]
            p[start:end] = (prob[0] + prob[1][:, order]) * .5
        probabilities[a:b] = np.repeat(p, STRIDE, axis=0)[:b-a]
    return probabilities


def combine_status(ownership, dead):
    return np.column_stack([ownership * (1-dead[:, None]), dead])


def load_frozen_status(model_path, validation, names, output, reusable=None):
    """A frozen external HGB status head is used only for validation/inference."""
    estimator = joblib.load(model_path)
    status_names = estimator["feature_names"]
    if any(name not in names for name in status_names) or not np.array_equal(estimator["classifier"].classes_, [0, 1, 2]):
        raise ValueError("Status model feature/class schema mismatch")
    projection = np.array([names.index(name) for name in status_names])
    training = estimator.get("provenance", {}).get("train", [])
    training = [training] if isinstance(training, str) else training
    if not training or set(training)-set(ALLOWED_TRAIN_IDS):
        raise ValueError("Status model training must be external-only")
    digest = fingerprint(model_path)
    permutation, sign = exchange_schema(status_names)
    cached = None
    if reusable is not None:
        metadata_path = reusable.with_suffix(".json")
        if not metadata_path.exists() or json.loads(metadata_path.read_text())["model_sha256"] != digest:
            raise ValueError("Reusable status probability model hash mismatch")
        with np.load(reusable, allow_pickle=False) as saved:
            cached = [saved["clean"], saved["occluded"]]
    result = []
    with threadpool_limits(limits=2):
        for index, record in enumerate(validation):
            x = record["x"]
            if cached is not None:
                p = cached[index]
                if p.shape != (len(x), 3):
                    raise ValueError("Reusable status probability shape mismatch")
                # An automatic external-only probe verifies source/condition order.
                probe = np.arange(0, len(x), 5000)
                projected = x[probe][:, projection]
                a = estimator["classifier"].predict_proba(projected)
                b = estimator["classifier"].predict_proba(exchange(projected, permutation, sign))[:, [1, 0, 2]]
                np.testing.assert_allclose(p[probe], (a+b)*.5, atol=1e-6, rtol=1e-5)
            else:
                projected = x[:, projection]
                a = estimator["classifier"].predict_proba(projected)
                b = estimator["classifier"].predict_proba(exchange(projected, permutation, sign))[:, [1, 0, 2]]
                p = (a+b)*.5
            result.append(p[:, 2].astype(np.float32))
    np.savez_compressed(output / "status_validation_probabilities.npz", clean=result[0], occluded=result[1])
    (output / "status_validation_probabilities.json").write_text(json.dumps({"model_sha256": digest,
        "model_path": str(model_path), "training_matches": training,
        "status_feature_projection": projection.tolist(), "status_feature_names": status_names,
        "validation_match": VALIDATION_ID, "used_as_training_input": False}, indent=2)+"\n")
    return result


def score_epoch(model, validation, mean, scale, permutation, sign, device,
                ownership_only=False, status_probabilities=None):
    probabilities = [infer(model, r, mean, scale, permutation, sign, device, ownership_only)
                     for r in validation]
    if ownership_only:
        if status_probabilities is None:
            raise ValueError("Ownership validation requires frozen external status probabilities")
        probabilities = [combine_status(q, d) for q, d in zip(probabilities, status_probabilities)]
    best, selected = None, None
    configurations = [{"switch_cost": 0., "dead_switch_cost": 0., "stride": STRIDE},
                      {"switch_cost": 1., "dead_switch_cost": 3., "stride": STRIDE}]
    if ownership_only:
        # Whole-spell weighting can make brief false controls very confident;
        # evaluate stronger externally selected persistence during epoch choice.
        configurations += [{"switch_cost": 2., "dead_switch_cost": 6., "stride": STRIDE},
                           {"switch_cost": 4., "dead_switch_cost": 6., "stride": STRIDE}]
    for config in configurations:
        scores = []
        for record, p in zip(validation, probabilities):
            pred = (np.argmax(p, axis=1).astype(np.int8) if config["switch_cost"] == 0
                    else decode(p, record["data"], config))
            scores.append(metrics(record["data"], pred))
        key = tuple(float(np.mean([score[k] for score in scores]))
                    for k in ("spell_f1_mean", "macro_f1"))
        if best is None or key > best:
            best = key
            selected = {"decoder": config, "mean_spell": key[0], "mean_macro": key[1],
                        "clean": scores[0], "occluded": scores[1]}
    return selected, probabilities


def validation_probabilities(output):
    """Prefer audited CPU probabilities only when bound to the current checkpoint."""
    path = output / "best_validation_probabilities.npz"
    audit = output / "cpu_audit.json"
    if audit.exists():
        metadata = json.loads(audit.read_text())
        if metadata["checkpoint_sha256"] == fingerprint(output / "best.pt"):
            path = output / "best_validation_probabilities_cpu.npz"
    with np.load(path, allow_pickle=False) as saved:
        return [saved["clean"], saved["occluded"]]


def audit_cpu(output, validation, protocol):
    """Repeat frozen CPU validation; no test match or target is accessed."""
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    names = checkpoint["feature_names"]
    mean, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
    permutation, sign = exchange_schema(names)
    network = TemporalNet(len(names)*2, checkpoint["channels"]).eval()
    network.load_state_dict(checkpoint["state_dict"])
    with np.load(output / "best_validation_probabilities.npz", allow_pickle=False) as saved:
        original = [saved["clean"], saved["occluded"]]
    report = {"checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": fingerprint(output / "best.pt"),
              "validation_match": VALIDATION_ID, "target_opened": False,
              "test_opened": False, "device": "cpu", "conditions": {}}
    probabilities = []
    for record, old in zip(validation, original):
        candidates = [entry for entry in protocol.get("feature_caches", [])
                      if entry["match_id"] == VALIDATION_ID and entry["condition"] == record["condition"]]
        if not candidates:
            raise ValueError("CPU audit requires recorded external feature-cache provenance")
        path = Path(candidates[0]["cache"])
        if json.loads(path.with_suffix(".json").read_text())["feature_names"] != names:
            raise ValueError("CPU audit cache schema mismatch")
        current = dict(record, x=np.load(path, mmap_mode="r"))
        ownership_only = checkpoint.get("ownership_only", False)
        p = infer(network, current, mean, scale, permutation, sign, torch.device("cpu"), ownership_only)
        repeated = infer(network, current, mean, scale, permutation, sign, torch.device("cpu"), ownership_only)
        if ownership_only:
            status_metadata = json.loads((output / "status_validation_probabilities.json").read_text())
            if status_metadata["model_sha256"] != checkpoint["status_model_sha256"]:
                raise ValueError("CPU audit status model provenance mismatch")
            with np.load(output / "status_validation_probabilities.npz", allow_pickle=False) as saved:
                dead = saved[record["condition"]]
            p, repeated = combine_status(p, dead), combine_status(repeated, dead)
        identical = bool(np.array_equal(p, repeated))
        if not identical:
            raise ValueError("Frozen CPU probabilities failed exact repeatability")
        scores = metrics(record["data"], decode(p, record["data"], checkpoint["decoder"]))
        report["conditions"][record["condition"]] = {
            "cpu_repeat_identical": identical,
            "cpu_probabilities_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
            "max_abs_original_device_difference": float(np.max(np.abs(p-old))),
            "argmax_disagreement_fraction": float(np.mean(p.argmax(axis=1) != old.argmax(axis=1))),
            "metrics": scores}
        probabilities.append(p)
    np.savez_compressed(output / "best_validation_probabilities_cpu.npz",
                        clean=probabilities[0], occluded=probabilities[1])
    (output / "cpu_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cpu_audit": report}), flush=True)


def tune_decoder(output, validation):
    """Select decoder only on the predeclared external validation probabilities."""
    probabilities = validation_probabilities(output)
    report = {"validation_match": VALIDATION_ID, "target_opened": False,
              "test_opened": False, "trials": []}
    best = None
    for cost in (0., .25, .5, 1., 2., 4.):
        for dead_cost in (0., 1., 3., 6.):
            config = {"switch_cost": cost, "dead_switch_cost": dead_cost, "stride": STRIDE}
            scores = []
            for record, probability in zip(validation, probabilities):
                prediction = (probability.argmax(axis=1).astype(np.int8)
                              if cost == 0 and dead_cost == 0
                              else decode(probability, record["data"], config))
                scores.append(metrics(record["data"], prediction))
            key = tuple(float(np.mean([s[k] for s in scores]))
                        for k in ("spell_f1_mean", "macro_f1"))
            trial = {"decoder": config, "mean_spell": key[0], "mean_macro": key[1],
                     "clean": scores[0], "occluded": scores[1]}
            report["trials"].append(trial)
            if best is None or key > best:
                best, report["selected"] = key, trial
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    checkpoint["decoder"] = report["selected"]["decoder"]
    torch.save(checkpoint, output / "best_decoded.pt")
    report["checkpoint_epoch"] = checkpoint["epoch"]
    report["checkpoint_sha256"] = fingerprint(output / "best_decoded.pt")
    (output / "decoder_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decoder_grid_selected": report["selected"]}), flush=True)


def tune_ensemble(output, validation, ensemble_path, refine_only=False):
    """WOY-only posterior/decoder fusion; target and held-out test stay unopened."""
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    names = checkpoint["feature_names"]
    estimator = joblib.load(ensemble_path)
    hgb_names = estimator["feature_names"]
    projection = None if any(name not in names for name in hgb_names) else np.array([names.index(name) for name in hgb_names])
    permutation, sign = exchange_schema(hgb_names)
    training = estimator.get("provenance", {}).get("train", [])
    training = [training] if isinstance(training, str) else training
    if set(training) - set(ALLOWED_TRAIN_IDS) or not training:
        raise ValueError("Ensemble HGB must be trained only on declared external games")
    if not np.array_equal(estimator["classifier"].classes_, [0, 1, 2]):
        raise ValueError("Ensemble HGB class order mismatch")
    tcn_fingerprint, hgb_fingerprint = fingerprint(output / "best.pt"), fingerprint(ensemble_path)
    temporal = validation_probabilities(output)
    cache = output / "hgb_validation_probabilities.npz"
    metadata = output / "hgb_validation_probabilities.json"
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text())["model_sha256"] == hgb_fingerprint:
        with np.load(cache, allow_pickle=False) as saved:
            geometric = [saved["clean"], saved["occluded"]]
    else:
        if projection is None:
            raise ValueError("Cross-profile fusion requires a hash-bound external HGB validation cache")
        geometric = []
        with threadpool_limits(limits=2):
            for record in validation:
                candidates = sorted((output / "feature_cache").glob(VALIDATION_ID + "_" + record["condition"] + "_*.npy"))
                if not candidates:
                    protocol = json.loads((output / "protocol.json").read_text())
                    candidates = [Path(entry["cache"]) for entry in protocol.get("feature_caches", [])
                                  if entry["match_id"] == VALIDATION_ID and entry["condition"] == record["condition"]]
                if not candidates:
                    raise ValueError("External feature cache missing from protocol")
                path = candidates[0]
                if json.loads(path.with_suffix(".json").read_text())["feature_names"] != names:
                    raise ValueError("Cached external feature schema mismatch")
                x = np.load(path, mmap_mode="r")
                classifier = estimator["classifier"]
                projected = x[:, projection]
                p = classifier.predict_proba(projected)
                p = (p + classifier.predict_proba(exchange(projected, permutation, sign))[:, [1, 0, 2]]) * .5
                geometric.append(p.astype(np.float32))
                print(json.dumps({"hgb_cached_condition": record["condition"]}), flush=True)
        np.savez_compressed(cache, clean=geometric[0], occluded=geometric[1])
        metadata.write_text(json.dumps({"model_sha256": hgb_fingerprint, "validation_match": VALIDATION_ID}) + "\n")
    for record, probability in zip(validation, geometric):
        if probability.shape != (len(record["data"]["ball"]), 3) or not np.isfinite(probability).all():
            raise ValueError("HGB validation cache shape/finite-value mismatch")
        if (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1., atol=1e-6):
            raise ValueError("HGB validation cache is not a probability distribution")
    report = {"tcn_checkpoint_epoch": checkpoint["epoch"], "tcn_sha256": tcn_fingerprint,
              "hgb_sha256": hgb_fingerprint, "hgb_model": str(ensemble_path),
              "validation_match": VALIDATION_ID, "target_opened": False,
              "test_opened": False, "trials": []}
    best = None
    best_probabilities = None
    hgb_config = dict(estimator["decoder"])
    factor = hgb_config["stride"] / STRIDE
    scaled_config = {"stride": STRIDE, "switch_cost": hgb_config["switch_cost"]*factor,
                     "dead_switch_cost": hgb_config["dead_switch_cost"]*factor}
    configurations = ([hgb_config, scaled_config] if refine_only else
        [{"switch_cost": cost, "dead_switch_cost": dead_cost, "stride": STRIDE}
         for cost in (0., .5, 1., 2., 4.) for dead_cost in (0., 3., 6.)]
        + [hgb_config, scaled_config])
    if refine_only:
        report = json.loads((output / "ensemble_report.json").read_text())
        if report["tcn_sha256"] != tcn_fingerprint or report["hgb_sha256"] != hgb_fingerprint:
            raise ValueError("Cannot refine an ensemble of different frozen models")
        best = (report["selected"]["mean_spell"], report["selected"]["mean_macro"])
        with np.load(output / "best_ensemble_validation_probabilities.npz", allow_pickle=False) as saved:
            best_probabilities = [saved["clean"], saved["occluded"]]
    for mode in ("posterior_mix", "hgb_dead"):
        for alpha in (0., .25, .5, .75, 1.):
            if mode == "hgb_dead" and alpha == 0:
                continue  # Same as posterior_mix alpha=0.
            probabilities = []
            for pt, pg in zip(temporal, geometric):
                if mode == "posterior_mix":
                    mixed = alpha*pt + (1-alpha)*pg
                else:
                    ct = pt[:, :2] / np.maximum(pt[:, :2].sum(axis=1, keepdims=True), 1e-7)
                    cg = pg[:, :2] / np.maximum(pg[:, :2].sum(axis=1, keepdims=True), 1e-7)
                    conditional = alpha*ct + (1-alpha)*cg
                    mixed = np.column_stack([conditional * (1-pg[:, 2:]), pg[:, 2]])
                probabilities.append(mixed)
            for config in configurations:
                scores = []
                for record, probability in zip(validation, probabilities):
                    pred = (probability.argmax(axis=1).astype(np.int8)
                            if config["switch_cost"] == 0 and config["dead_switch_cost"] == 0
                            else decode(probability, record["data"], config))
                    scores.append(metrics(record["data"], pred))
                key = tuple(float(np.mean([s[k] for s in scores]))
                            for k in ("spell_f1_mean", "macro_f1"))
                trial = {"mode": mode, "tcn_weight": alpha, "decoder": config,
                         "mean_spell": key[0], "mean_macro": key[1],
                         "clean": scores[0], "occluded": scores[1]}
                report["trials"].append(trial)
                if best is None or key > best:
                    best, report["selected"], best_probabilities = key, trial, probabilities
            print(json.dumps({"ensemble_mode": mode, "tcn_weight": alpha, "best_spell": best[0]}), flush=True)
            (output / "ensemble_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "best_ensemble_validation_probabilities.npz",
                        clean=best_probabilities[0], occluded=best_probabilities[1])
    descriptor = {"schema_version": 1, "kind": "tcn_hgb_fusion",
                  "tcn_checkpoint": "best.pt", "tcn_sha256": tcn_fingerprint,
                  "hgb_model": str(ensemble_path), "hgb_sha256": hgb_fingerprint,
                  "feature_profile": checkpoint["feature_profile"], "feature_names": names,
                  "hgb_feature_profile": estimator.get("feature_profile", "geometry"),
                  "ownership_only": checkpoint.get("ownership_only", False),
                  "status_model": checkpoint.get("status_model"),
                  "status_model_sha256": checkpoint.get("status_model_sha256"),
                  "training_matches": checkpoint["train_matches"],
                  "validation_match": VALIDATION_ID, **report["selected"]}
    (output / "ensemble_model.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(json.dumps({"ensemble_selected": report["selected"]}), flush=True)


def main():
    global PROFILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-profile", choices=["contact", "contact_shape", "contact_motion"], default=PROFILE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--core", type=int, default=256)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tune-decoder", action="store_true")
    parser.add_argument("--audit-cpu", action="store_true")
    parser.add_argument("--ensemble-model", type=Path)
    parser.add_argument("--ensemble-refine", action="store_true",
                        help="Append the original HGB decoder and its stride-scaled equivalent to an existing fusion report")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--frozen-preprocessing", type=Path,
                        help="Verified external checkpoint whose train-only mean/scale stay fixed for warm-start")
    parser.add_argument("--train-player-augmentation", action="store_true",
                        help="Add a third training-only anonymous-player + ball-occlusion view; WOY stays clean/occluded")
    parser.add_argument("--ownership-only", action="store_true")
    parser.add_argument("--segment-power", type=float, choices=[0., .5, 1.], default=.5)
    parser.add_argument("--status-model", type=Path)
    parser.add_argument("--status-cache", type=Path)
    parser.add_argument("--cache-signature", help="Explicit reviewed frozen feature-cache implementation signature")
    parser.add_argument("--reuse-cache", type=Path, nargs="+",
                        help="Reuse reviewed external contact caches, preserving original cache provenance")
    parser.add_argument("--train-ids", nargs="+", choices=ALLOWED_TRAIN_IDS, default=list(TRAIN_IDS))
    args = parser.parse_args()
    PROFILE = args.feature_profile
    if args.resume and args.warm_start:
        raise ValueError("Choose resume or warm-start")
    if args.frozen_preprocessing and not (args.warm_start or args.resume):
        raise ValueError("Frozen preprocessing requires an explicit warm-start or resume checkpoint")
    if args.ownership_only and not args.status_model:
        raise ValueError("Ownership-only experiment requires frozen external status model")
    train_ids = tuple(args.train_ids)
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Training matches must be unique")
    if min(args.epochs, args.steps, args.batch_size, args.core, args.channels, args.threads) <= 0:
        raise ValueError("All size and count arguments must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=args.device == "mps")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; run with --device cpu")
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tune_decoder or args.ensemble_model or args.audit_cpu:
        data_path = args.data_dir / (VALIDATION_ID + "_external_v1.npz")
        data = load_external(data_path)
        protocol = json.loads((args.out / "protocol.json").read_text())
        if fingerprint(data_path) != protocol["source_sha256"][VALIDATION_ID]:
            raise ValueError("Decoder validation source changed")
        validation = [{"data": data, "condition": condition} for condition in ("clean", "occluded")]
        if args.audit_cpu:
            audit_cpu(args.out, validation, protocol)
        elif args.ensemble_model:
            tune_ensemble(args.out, validation, args.ensemble_model, args.ensemble_refine)
        else:
            tune_decoder(args.out, validation)
        return
    records, validation, names, sources = prepare(args.data_dir, args.out / "feature_cache", args.cache_signature,
                                                 train_ids, args.reuse_cache, args.segment_power,
                                                 args.train_player_augmentation)
    if args.frozen_preprocessing:
        mean, scale, permutation, sign, normalisation_lineage = frozen_normalisation(
            args.frozen_preprocessing, names, sources, train_ids, PROFILE)
    else:
        mean, scale, permutation, sign = fit_normalisation(records, names)
        normalisation_lineage = {"mode": "fit-external-training-only", "train_matches": list(train_ids),
                                 "validation_used_for_fit": False,
                                 "new_corruption_used_for_fit": args.train_player_augmentation}
    np.savez(args.out / "external_preprocessing.npz", mean=mean, scale=scale,
             permutation=permutation, sign=sign)
    report = {"architecture": "six noncausal residual kernel3 dilated TCN blocks",
              "channels": args.channels, "stride": STRIDE, "receptive_field_seconds": 127*STRIDE/25,
              "seed": SEED, "torch_version": str(torch.__version__), "device": args.device,
              "train_matches": list(train_ids), "validation_match": VALIDATION_ID,
              "source_sha256": sources, "feature_profile": PROFILE, "feature_version": FEATURE_VERSION,
              "normalisation_scale_floor": .001 if PROFILE == "contact_shape" else .05,
              "feature_caches": [{k:r[k] for k in ("match_id", "condition", "cache", "cache_implementation",
                                                    "sampled_cache_equivalence", "player_augmentation")} for r in records+validation],
              "train_player_augmentation": args.train_player_augmentation,
              "normalisation_lineage": normalisation_lineage,
              "selection": "mean clean/occluded external validation spellF1, then macroF1",
              "ownership_only": args.ownership_only, "segment_power": args.segment_power,
              "warm_start_sha256": fingerprint(args.warm_start) if args.warm_start else None,
              "status_model_sha256": fingerprint(args.status_model) if args.status_model else None,
              "test_opened": False, "target_opened": False, "epochs": []}
    if args.resume and (args.out / "training_report.json").exists():
        previous_report = json.loads((args.out / "training_report.json").read_text())
        if (previous_report["source_sha256"] != sources
                or previous_report.get("train_player_augmentation", False) != args.train_player_augmentation):
            raise ValueError("Resume protocol/source mismatch")
        # Preserve the original warm-start lineage across later resume stages.
        report["warm_start_sha256"] = previous_report.get("warm_start_sha256")
        report["resume_checkpoint_sha256"] = fingerprint(args.resume)
    (args.out / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.prepare_only:
        return
    model = TemporalNet(len(names)*2, args.channels).to(device)
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        if warm["feature_names"] != names or warm["channels"] != args.channels:
            raise ValueError("Warm-start architecture/feature mismatch")
        validate_warm_source(warm, sources, train_ids, PROFILE)
        np.testing.assert_allclose(warm["mean"].numpy(), mean, atol=1e-6)
        np.testing.assert_allclose(warm["scale"].numpy(), scale, atol=1e-6)
        model.load_state_dict(warm["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    rng = np.random.default_rng(SEED)
    first_epoch, best_key = 1, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["feature_names"] != names or checkpoint["source_sha256"] != sources:
            raise ValueError("Resume checkpoint/source mismatch")
        np.testing.assert_array_equal(checkpoint["mean"].numpy(), mean)
        np.testing.assert_array_equal(checkpoint["scale"].numpy(), scale)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        first_epoch = checkpoint["epoch"] + 1
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        report_path = args.out / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.setdefault("normalisation_lineage", normalisation_lineage)
            report.setdefault("train_player_augmentation", args.train_player_augmentation)
            best_key = (report["selected"]["mean_spell"], report["selected"]["mean_macro"])
    counts = np.zeros(3)
    for record in records:
        y = record["data"]["labels"]
        counts += np.bincount(y[y >= 0], minlength=3)
    # Team-exchange augmentation balances H/A counts analytically.
    counts[:2] = counts[:2].mean()
    class_weights = (torch.ones(2, dtype=torch.float32, device=device) if args.ownership_only
                     else torch.tensor(counts.sum() / (3*np.maximum(counts, 1)),
                                       dtype=torch.float32, device=device))
    status_probabilities = (load_frozen_status(args.status_model, validation, names, args.out, args.status_cache)
                            if args.ownership_only else None)
    sequences = sequence_inventory(records)
    started = monotonic()
    patience_anchor = first_epoch-1
    report.setdefault("training_stages", []).append({"start_epoch": first_epoch,
        "max_epoch": args.epochs, "learning_rate": args.learning_rate, "patience": args.patience,
        "device": args.device})
    for epoch in range(first_epoch, args.epochs+1):
        epoch_start = monotonic()
        model.train()
        loss_total = 0.
        for _ in range(args.steps):
            x, y, weights = batch(sequences, rng, mean, scale, permutation, sign,
                                  args.batch_size, args.core)
            x, y, weights = x.to(device), y.to(device), weights.to(device)
            if args.ownership_only:
                weights = weights * ((y >= 0) & (y < 2))
                y = torch.where(y == 2, -1, y)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[:, :, CONTEXT:CONTEXT+args.core]
            if args.ownership_only:
                logits = logits[:, :2]
            ce = F.cross_entropy(logits, y, weight=class_weights, ignore_index=-1, reduction="none")
            loss = (ce * weights).sum() / weights.sum().clamp(min=1)
            # External same-label smoothing avoids penalising genuine short controls.
            logp = logits.log_softmax(dim=1)
            same = (y[:, 1:] == y[:, :-1]) & (y[:, 1:] >= 0)
            change = (logp[:, :, 1:] - logp[:, :, :-1]).square().mean(dim=1).clamp(max=16)
            loss = loss + .01 * (change * same).sum() / same.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            loss_total += float(loss.detach().cpu())
        selected, probabilities = score_epoch(model, validation, mean, scale, permutation, sign, device,
                                               args.ownership_only, status_probabilities)
        key = (selected["mean_spell"], selected["mean_macro"])
        entry = {"epoch": epoch, "loss": loss_total/args.steps, "learning_rate": args.learning_rate,
                 "elapsed_seconds": monotonic()-epoch_start, **selected}
        report["epochs"].append(entry)
        checkpoint = {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                      "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "channels": args.channels, "feature_names": names,
                      "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                      "source_sha256": sources, "train_matches": list(train_ids),
                      "validation_match": VALIDATION_ID, "decoder": selected["decoder"],
                      "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                      "feature_version": FEATURE_VERSION, "feature_profile": PROFILE}
        checkpoint.update(ownership_only=args.ownership_only, segment_power=args.segment_power,
                          status_model_sha256=fingerprint(args.status_model) if args.status_model else None,
                          status_model=str(args.status_model) if args.status_model else None,
                          warm_start_sha256=report.get("warm_start_sha256"),
                          normalisation_lineage=report["normalisation_lineage"],
                          train_player_augmentation=report["train_player_augmentation"])
        torch.save(checkpoint, args.out / "last.pt")
        if best_key is None or key > best_key:
            best_key = key
            report["selected"] = entry
            torch.save(checkpoint, args.out / "best.pt")
            np.savez_compressed(args.out / "best_validation_probabilities.npz",
                                clean=probabilities[0], occluded=probabilities[1])
        report["elapsed_seconds"] = monotonic() - started
        report["best_checkpoint_sha256"] = fingerprint(args.out / "best.pt")
        (args.out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "loss": entry["loss"],
                          "spell": key[0], "macro": key[1], "best_spell": best_key[0],
                          "elapsed_seconds": entry["elapsed_seconds"]}), flush=True)
        if epoch - max(report["selected"]["epoch"], patience_anchor) >= args.patience:
            print(json.dumps({"early_stop_epoch": epoch, "best_epoch": report["selected"]["epoch"],
                              "patience": args.patience}), flush=True)
            break


if __name__ == "__main__":
    main()
