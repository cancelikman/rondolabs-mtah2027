"""Score the frozen WOY reference on native WCSF labels."""
import argparse
import json
from pathlib import Path
import sys

import zipfile
import pandas as pd
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import tcn_external as base
from experiments.joint_transition_external import JointTemporalNet, infer_joint
from experiments.transition_external import decode
from src.bundle import _geometry_only, _symmetric_probability, load_manifest
from src.features import build_features, occlude_external
from src.train import fingerprint, metrics
from experiments.can_skillcorner_audit import resample_geometry

def load_wcsf(paths, expected):
    balls, players, labels, blocks = [], [], [], []
    team_rows = []
    for path, receipt in zip(paths, expected):
        if fingerprint(path) != receipt["file_sha256"]:
            raise ValueError("WCSF archive differs from schema audit")
        with zipfile.ZipFile(path) as archive:
            member, = [name for name in archive.namelist() if name.startswith("positions_")]
            with archive.open(member) as stream:
                rows = pd.read_csv(stream, usecols=[
                    "timestamp_ms", "group_id", "player_id", "x", "y",
                    "x_ball", "y_ball", "AliveFlag", "PossessionFlag",
                ])
        team_rows.append(rows)
    tracks = pd.concat([rows[["group_id", "player_id"]] for rows in team_rows]).drop_duplicates()
    tracks = tracks.sort_values(["group_id", "player_id"])
    slots = pd.MultiIndex.from_frame(tracks)
    cursor = 0
    for rows in team_rows:
        sign = np.where(rows.group_id == 1, 1, -1)
        columns = ["x", "y", "x_ball", "y_ball"]
        rows[columns] = rows[columns].mul(sign, axis=0)
        frames = rows.groupby("timestamp_ms", sort=True)
        flags = frames[["AliveFlag", "PossessionFlag"]].first()
        inconsistent = frames[["AliveFlag", "PossessionFlag"]].nunique(dropna=False).gt(1).any(axis=1)
        own = flags.PossessionFlag.map({1: 0, 2: 1}).where(flags.AliveFlag.eq(1))
        label = own.mask(flags.AliveFlag.eq(0), 2).mask(inconsistent).fillna(-1).to_numpy(dtype=np.int8)
        timestamps = flags.index.to_numpy()
        if not np.all(np.diff(timestamps) == 100):
            raise ValueError("Unexpected WCSF timestamp gap")
        features = []
        for axis in ("x", "y"):
            table = rows.pivot(index="timestamp_ms", columns=["group_id", "player_id"], values=axis)
            features.append(table.reindex(index=timestamps, columns=slots).to_numpy(dtype=np.float32))
        players.append(np.stack(features, axis=-1))
        balls.append(frames[["x_ball", "y_ball"]].median().to_numpy(dtype=np.float32))
        labels.append(label)
        blocks.append((cursor, cursor + len(label)))
        cursor += len(label)
    return dict(players=np.concatenate(players), ball=np.concatenate(balls), labels=np.concatenate(labels),
                blocks=np.asarray(blocks), fps=10., pitch=np.array([105., 68.]),
                team_of_track=tracks.group_id.to_numpy(dtype=np.int8) - 1)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halves", type=Path, nargs=2, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = (None, 202)

    receipt = json.loads((ROOT / "models/development_checkpoints/can_spell5_validation_hashes.json").read_text())
    reference = json.loads((ROOT / "reports/external/can_spell5_WOY_audit.json").read_text())
    checkpoint_sha = fingerprint(args.checkpoint)
    if checkpoint_sha != receipt["checkpoint_sha256"] or checkpoint_sha != reference["checkpoint_sha256"]:
        raise ValueError("Frozen spell5 checkpoint required")
    schema_path = ROOT / "reports/external/wcsf_possession_schema.json"
    schema = json.loads(schema_path.read_text())
    data = load_wcsf(args.halves, schema["halves"])
    for (start, stop), half in zip(data["blocks"], schema["halves"]):
        counts = {key: int((data["labels"][start:stop] == code).sum())
                  for key, code in zip("HADX", (0, 1, 2, -1))}
        if counts != half["labels"]:
            raise ValueError("Native label counts did not replay")
    geometry, mapping = resample_geometry(_geometry_only(data))
    if not np.array_equal(mapping[:, 0], np.arange(len(data["labels"]))):
        raise ValueError("Incomplete native frame mapping")

    manifest, paths, manifest_sha = load_manifest(ROOT / "models/ahmetcan_joint_checkpoint")
    if manifest_sha != reference["parent_manifest_sha256"]:
        raise ValueError("Frozen parent manifest required")
    hgb = joblib.load(paths["hgb"])
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = JointTemporalNet(660, 64).eval()
    model.load_state_dict(cp["state_dict"])
    names = cp["feature_names"]
    mean, scale = cp["mean"].numpy(), cp["scale"].numpy()
    permutation, sign = base.exchange_schema(names)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = dict(
        source=schema["source"], schema_sha256=fingerprint(schema_path),
        checkpoint_sha256=checkpoint_sha, parent_manifest_sha256=manifest_sha,
        driver_sha256=fingerprint(Path(__file__)), WR9_access=False,
        fit=False, selection=False, competition_access=False, provider_label_parity_verified=False,
        native_fps=10, inference_fps=25, synthetic_occlusion_seeds=seeds, conditions={},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    with args.out.with_suffix(".protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    config = manifest["decoder"]

    with threadpool_limits(limits=2):
        for seed in seeds:
            condition = "clean" if seed is None else f"occluded_seed{seed}"
            current = geometry if seed is None else occlude_external(geometry, seed)
            features, actual_names = build_features(current, profile="contact_motion")
            if actual_names != names or actual_names[:180] != hgb["feature_names"]:
                raise ValueError("Frozen geometry schema changed")
            state, event = infer_joint(model, dict(data=current, x=features), mean, scale,
                                       permutation, sign, torch.device("cpu"))
            state = .5 * (state + _symmetric_probability(hgb, features[:, :180], hgb["feature_names"]))
            prediction = decode(state, event, current, config)[mapping[:, 1]]
            report["conditions"][condition] = metrics(data, prediction)
            print(condition, report["conditions"][condition]["spell_f1_mean"], flush=True)
    report["complete"] = True
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
