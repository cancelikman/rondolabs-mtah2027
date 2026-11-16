"""Hash-verified, portable frozen HGB/TCN/event inference from geometry only.

The caller provides positions and recording boundaries. This module never opens
match data/video/GT, reads labels/status, fits/calibrates, or selects a model.
Only model component files named relative to a reviewed manifest are loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .features import FEATURE_VERSION, blocks_of, build_features
from .symmetry import swap_feature_matrix

SCHEMA_VERSION = 1
JOINT_SCHEMA_VERSION = 2
JOINT_ARCHITECTURE = "shared-state-directed-event-tcn-v1"
JOINT_STATE_CLASSES = ["H", "A", "D"]
JOINT_EVENT_CLASSES = ["none", "newH", "newA", "newD"]
JOINT_DECODER = {"stride": 2, "team_cost": 4., "dead_cost": 6.,
                 "event_strength": 1., "temperature": 1.4}
ALLOWED_TRAIN_MATCHES = {"J03WOH", "J03WN1", "J03WMX", "J03WQQ"}
FUSION_MODES = {"posterior_mix", "conditional_ownership", "hgb_dead"}
COMPONENT_PROFILES = {"geometry", "contact", "contact_motion", "contact_shape", "contact_motion_shape"}
BUILD_PROFILES = {"contact_motion", "contact_motion_shape"}
OPTIONAL_TEMPORAL_PROVENANCE = ("normalisation_lineage", "train_player_augmentation", "warm_start_sha256")


def _temporal_provenance(values, training_matches):
    """Optional inspectable training metadata, bound in manifest and checkpoint."""
    result = {key:values[key] for key in OPTIONAL_TEMPORAL_PROVENANCE
              if key in values and values[key] is not None}
    try:
        result = json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("temporal training metadata must be finite JSON") from exc
    if "train_player_augmentation" in result and not isinstance(result["train_player_augmentation"], bool):
        raise ValueError("invalid temporal augmentation metadata")
    if "warm_start_sha256" in result:
        value = result["warm_start_sha256"]
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid temporal warm-start SHA-256")
    if "normalisation_lineage" in result:
        lineage = result["normalisation_lineage"]
        if (not isinstance(lineage,dict) or lineage.get("validation_used_for_fit") is not False
                or not set(_training_ids(lineage.get("train_matches",[]))) <= set(training_matches)):
            raise ValueError("temporal normalization lineage must remain external training only")
    return result


def component_build_profile(components):
    """Select a reviewed single-extraction superset for saved component schemas."""
    profiles = {record.get("feature_profile") for record in components.values()}
    if "contact_reach" in profiles:
        raise ValueError("contact_reach bundle requires an unimplemented contact_motion_reach union")
    if not profiles <= COMPONENT_PROFILES:
        raise ValueError("invalid component feature profile")
    return ("contact_motion_shape" if profiles & {"contact_shape", "contact_motion_shape"}
            else "contact_motion")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_ids(values):
    values = [values] if isinstance(values, str) else list(values)
    if (not values or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values) or not set(values) <= ALLOWED_TRAIN_MATCHES):
        raise ValueError("bundle training provenance is not the declared external split")
    return values


def _schema(names):
    if (not isinstance(names, list) or not names or any(not isinstance(v, str) for v in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid component feature schema")
    return names


def _component_path(folder, record):
    name = record.get("file")
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("component path must be relative to the bundle")
    path = (folder / name).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise ValueError("component missing or outside bundle directory")
    expected = record.get("sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("component requires a lowercase SHA-256")
    if file_sha256(path) != expected:
        raise ValueError("component SHA-256 mismatch: %s" % name)
    return path


def _joint_contract(record):
    if (record.get("architecture") != JOINT_ARCHITECTURE
            or record.get("feature_profile") != "contact_motion"
            or len(_schema(record.get("feature_names"))) != 330
            or record.get("state_class_order") != JOINT_STATE_CLASSES
            or record.get("event_class_order") != JOINT_EVENT_CLASSES
            or record.get("ownership_only") is not False
            or record.get("stride") != 2 or record.get("context") != 63
            or not isinstance(record.get("channels"), int) or isinstance(record.get("channels"), bool)
            or not 1 <= record["channels"] <= 512
            or set(_training_ids(record.get("training_matches", []))) != ALLOWED_TRAIN_MATCHES):
        raise ValueError("unsupported joint temporal architecture/class/source contract")
    for key in ("source_checkpoint_sha256", "source_training_report_sha256",
                "source_cpu_audit_sha256", "source_parent_manifest_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("joint temporal source requires SHA-256 provenance")


def load_manifest(bundle_dir):
    """Validate metadata and every file hash before deserializing a component."""
    folder = Path(bundle_dir).resolve()
    path = folder / "manifest.json"
    if not path.is_file():
        raise ValueError("bundle manifest.json missing")
    manifest = json.loads(path.read_text())
    version = manifest.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version not in {SCHEMA_VERSION, JOINT_SCHEMA_VERSION}:
        raise ValueError("unsupported bundle schema_version")
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("bundle feature implementation mismatch")
    if manifest.get("build_profile") not in BUILD_PROFILES:
        raise ValueError("bundle must use a reviewed single-extraction superset")
    if manifest.get("runtime") != {"device": "cpu", "threads": 2}:
        raise ValueError("bundle requires fixed CPU inference with two threads")
    if manifest.get("validation_match") != "J03WOY":
        raise ValueError("bundle validation must be declared external J03WOY")
    top_train = set(_training_ids(manifest.get("training_matches", [])))
    components = manifest.get("components", {})
    if not isinstance(components, dict) or not {"hgb", "tcn"} <= set(components) or not set(components) <= {"hgb", "tcn", "event"}:
        raise ValueError("bundle requires HGB/TCN and optional event components")
    if any(not isinstance(record, dict) for record in components.values()):
        raise ValueError("invalid component record")
    if version == JOINT_SCHEMA_VERSION:
        if (manifest.get("kind") != "frozen_geometry_joint_fusion"
                or manifest.get("event_source") != "tcn"
                or set(components) != {"hgb", "tcn"}
                or manifest.get("build_profile") != "contact_motion"
                or manifest.get("fusion") != {"mode": "posterior_mix", "tcn_weight": .5}
                or manifest.get("decoder") != JOINT_DECODER
                or manifest.get("selection", {}).get("target_used_for_selection") is not False
                or manifest.get("selection", {}).get("heldout_used_for_selection") is not False):
            raise ValueError("joint schema requires the reviewed shared-event/fixed-fusion contract")
        _joint_contract(components["tcn"])
        evidence = manifest.get("evidence", {})
        if (not isinstance(evidence, dict) or
                set(evidence) != {"parent_manifest", "training_report", "cpu_audit", "ablation_protocol", "ablation_report"}
                or any(not isinstance(record, dict) for record in evidence.values())):
            raise ValueError("joint schema requires complete source evidence")
        for name, record in evidence.items():
            _component_path(folder, record)
        for name, source in (("parent_manifest", "source_parent_manifest_sha256"),
                             ("training_report", "source_training_report_sha256"),
                             ("cpu_audit", "source_cpu_audit_sha256")):
            if evidence[name]["sha256"] != components["tcn"][source]:
                raise ValueError("joint temporal/evidence source hash mismatch")
    elif manifest.get("event_source") == "tcn" or components["tcn"].get("architecture") == JOINT_ARCHITECTURE:
        raise ValueError("joint temporal components require schema version 2")
    if manifest["build_profile"] != component_build_profile(components):
        raise ValueError("bundle build profile does not match component superset")
    fusion = manifest.get("fusion", {})
    weight = fusion.get("tcn_weight")
    if (fusion.get("mode") not in FUSION_MODES or not isinstance(weight, (float, int))
            or not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError("invalid bundle fusion settings")
    config = manifest.get("decoder", {})
    required = {"stride", "team_cost", "dead_cost", "event_strength", "temperature"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("bundle decoder settings missing")
    if (not isinstance(config["stride"], int) or isinstance(config["stride"], bool)
            or config["stride"] < 1 or not all(isinstance(config[k], (int,float))
            and np.isfinite(config[k]) for k in required-{"stride"})
            or min(config[k] for k in ("team_cost", "dead_cost", "event_strength")) < 0
            or config["temperature"] <= 0):
        raise ValueError("invalid bundle decoder settings")
    if version == SCHEMA_VERSION and "event" not in components and config["event_strength"] != 0:
        raise ValueError("event reward requires an event component")
    paths, union = {}, set()
    for name, record in components.items():
        if not isinstance(record, dict):
            raise ValueError("invalid component record")
        _schema(record.get("feature_names"))
        if record.get("feature_profile") not in COMPONENT_PROFILES:
            raise ValueError("invalid component feature profile")
        train = set(_training_ids(record.get("training_matches", [])))
        if name == "tcn":
            _temporal_provenance(record, train)
        if record.get("validation_match") != manifest["validation_match"]:
            raise ValueError("component validation provenance mismatch")
        union |= train
        paths[name] = _component_path(folder, record)
    if union != top_train:
        raise ValueError("manifest/component training provenance mismatch")
    return manifest, paths, file_sha256(path)


def _geometry_only(data):
    # Do not pass unknown fields to helper modules, even when a caller supplies labels.
    geometry = {key: data[key] for key in ("ball", "players", "team_of_track", "fps", "pitch")}
    for key in ("blocks", "period", "timestamps"):
        if key in data:
            geometry[key] = data[key]
    return geometry


def _select(x, names, wanted):
    _schema(wanted)
    index = {name: i for i, name in enumerate(names)}
    if any(name not in index for name in wanted):
        raise ValueError("component schema not present in extracted geometry")
    return np.take(x, [index[name] for name in wanted], axis=1)


def _check_artifact_schema(artifact, record, event=False):
    if artifact.get("feature_names") != record["feature_names"]:
        raise ValueError("artifact/manifest feature schema mismatch")
    if artifact.get("feature_profile", "geometry") != record["feature_profile"]:
        raise ValueError("artifact/manifest feature profile mismatch")
    if not event and artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("frame artifact implementation mismatch")
    provenance = artifact.get("provenance", {})
    if (set(_training_ids(provenance.get("train", []))) != set(record["training_matches"])
            or provenance.get("validation") != record["validation_match"]):
        raise ValueError("artifact/manifest training provenance mismatch")
    if event and not provenance.get("geometry_only_event_inputs"):
        raise ValueError("event artifact must declare geometry-only inputs")
    if not event and provenance.get("training_data") != "external IDSSE only":
        raise ValueError("frame artifact must declare external-only training")
    classes = [0,1,2,3] if event else [0,1,2]
    if not np.array_equal(artifact["classifier"].classes_, classes):
        raise ValueError("invalid artifact class ordering")


def _symmetric_probability(artifact, selected, names, event=False):
    estimator = artifact["classifier"]
    probability = estimator.predict_proba(selected)
    order = [0,2,1,3] if event else [1,0,2]
    exchanged = estimator.predict_proba(swap_feature_matrix(selected, names))[:,order]
    return (probability+exchanged)*.5


def _fuse(hgb, temporal, fusion, ownership_only):
    weight = float(fusion["tcn_weight"])
    if fusion["mode"] == "posterior_mix":
        if ownership_only or temporal.shape[1] != 3:
            raise ValueError("posterior_mix requires a three-class temporal model")
        return weight*temporal+(1-weight)*hgb
    temporal_owner = temporal[:,:2]
    temporal_owner = temporal_owner/np.maximum(temporal_owner.sum(axis=1,keepdims=True),1e-7)
    geometric_owner = hgb[:,:2]/np.maximum(hgb[:,:2].sum(axis=1,keepdims=True),1e-7)
    conditional = weight*temporal_owner+(1-weight)*geometric_owner
    return np.column_stack([conditional*(1-hgb[:,2:]),hgb[:,2]])


def predict_bundle(data, bundle_dir):
    """Return integer labels (-1 outside blocks, H/A/D=0/1/2) and provenance.

    Hash verification precedes model deserialization. Feature extraction runs once;
    each component receives its exact saved schema. CPU threads/deterministic
    settings apply locally to inference and are restored before returning.
    """
    import torch
    from experiments.tcn_external import TemporalNet, infer, exchange_schema, STRIDE, CONTEXT
    from experiments.transition_external import decode as directed_decode

    manifest, paths, manifest_digest = load_manifest(bundle_dir)
    components = manifest["components"]
    frame_model = joblib.load(paths["hgb"])
    _check_artifact_schema(frame_model, components["hgb"])
    event_model = None
    if "event" in paths:
        event_model = joblib.load(paths["event"])
        _check_artifact_schema(event_model, components["event"], event=True)
    checkpoint = torch.load(paths["tcn"], map_location="cpu", weights_only=True)
    temporal_record = components["tcn"]
    if (checkpoint.get("feature_names") != temporal_record["feature_names"]
            or checkpoint.get("feature_profile") != temporal_record["feature_profile"]
            or checkpoint.get("feature_version") != FEATURE_VERSION
            or set(_training_ids(checkpoint.get("train_matches", []))) != set(temporal_record["training_matches"])
            or checkpoint.get("validation_match") != temporal_record["validation_match"]):
        raise ValueError("temporal checkpoint schema/provenance mismatch")
    temporal_provenance = _temporal_provenance(checkpoint, temporal_record["training_matches"])
    if temporal_provenance != _temporal_provenance(temporal_record, temporal_record["training_matches"]):
        raise ValueError("temporal checkpoint/manifest training metadata mismatch")
    if temporal_record.get("stride") != STRIDE or temporal_record.get("context") != CONTEXT:
        raise ValueError("temporal stride/receptive-field contract mismatch")
    ownership_only = bool(checkpoint.get("ownership_only", False))
    if ownership_only != bool(temporal_record.get("ownership_only", False)):
        raise ValueError("temporal ownership mode mismatch")
    is_joint = manifest["schema_version"] == JOINT_SCHEMA_VERSION
    if is_joint:
        _joint_contract(dict(checkpoint, training_matches=checkpoint.get("train_matches", [])))
        for key in ("architecture", "channels", "state_class_order", "event_class_order",
                    "source_checkpoint_sha256", "source_training_report_sha256",
                    "source_cpu_audit_sha256", "source_parent_manifest_sha256"):
            if checkpoint.get(key) != temporal_record.get(key):
                raise ValueError("joint checkpoint/manifest contract mismatch")
        if checkpoint.get("version") != JOINT_ARCHITECTURE:
            raise ValueError("unknown joint checkpoint version")
        sources = checkpoint.get("source_sha256", {})
        if (set(sources) != ALLOWED_TRAIN_MATCHES | {"J03WOY"} or
                any(not isinstance(value, str) or len(value) != 64 or
                    any(c not in "0123456789abcdef" for c in value) for value in sources.values())):
            raise ValueError("joint checkpoint source provenance must be exactly four training matches plus WOY")
        folder = Path(bundle_dir).resolve()
        training = json.loads(_component_path(folder, manifest["evidence"]["training_report"]).read_text())
        audit = json.loads(_component_path(folder, manifest["evidence"]["cpu_audit"]).read_text())
        if (training.get("source_sha256") != sources or audit.get("source_sha256") != sources
                or audit.get("checkpoint_sha256") != temporal_record["source_checkpoint_sha256"]
                or audit.get("cpu_repeat_identical") is not True
                or training.get("test_opened") is not False or training.get("target_opened") is not False
                or audit.get("target_or_test_opened") is not False
                or training.get("source_code_sha256") != file_sha256(
                    Path(__file__).resolve().parents[1]/"experiments/joint_transition_external.py")):
            raise ValueError("joint checkpoint source/audit evidence mismatch")
    mean, scale = checkpoint["mean"].cpu().numpy(), checkpoint["scale"].cpu().numpy()
    count = len(temporal_record["feature_names"])
    if mean.shape != (count,) or scale.shape != (count,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid frozen external temporal normalization")
    geometry = _geometry_only(data)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=2):
            x, names = build_features(geometry, profile=manifest["build_profile"])
            hgb_x = _select(x,names,components["hgb"]["feature_names"])
            hgb = _symmetric_probability(frame_model,hgb_x,components["hgb"]["feature_names"])
            temporal_x = _select(x,names,temporal_record["feature_names"])
            permutation, sign = exchange_schema(temporal_record["feature_names"])
            if is_joint:
                from experiments.joint_transition_external import JointTemporalNet, infer_joint
                network = JointTemporalNet(count*2,int(checkpoint["channels"])).eval()
                network.load_state_dict(checkpoint["state_dict"])
                temporal, events = infer_joint(network,{"data":geometry,"x":temporal_x},mean,scale,
                                               permutation,sign,torch.device("cpu"))
                posterior = _fuse(hgb,temporal,manifest["fusion"],False)
            else:
                network = TemporalNet(count*2,int(checkpoint["channels"])).eval()
                network.load_state_dict(checkpoint["state_dict"])
                temporal = infer(network,{"data":geometry,"x":temporal_x},mean,scale,
                                 permutation,sign,torch.device("cpu"),ownership_only)
                posterior = _fuse(hgb,temporal,manifest["fusion"],ownership_only)
                events = np.tile([1.,0.,0.,0.],(len(x),1))
                if event_model is not None:
                    selected = _select(x,names,components["event"]["feature_names"])
                    events = _symmetric_probability(event_model,selected,components["event"]["feature_names"],event=True)
            for start,end in blocks_of(geometry):
                if (not np.isfinite(posterior[start:end]).all() or (posterior[start:end] < 0).any()
                        or not np.allclose(posterior[start:end].sum(axis=1),1.,atol=1e-5)):
                    raise ValueError("bundle produced invalid block posterior")
                if is_joint and (not np.isfinite(events[start:end]).all() or (events[start:end] < 0).any()
                        or not np.allclose(events[start:end].sum(axis=1),1.,atol=1e-5)):
                    raise ValueError("bundle produced invalid joint event posterior")
            labels = directed_decode(posterior,events,geometry,manifest["decoder"])
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
    metadata = {"bundle_schema_version":manifest["schema_version"],"manifest_sha256":manifest_digest,
        "component_sha256":{name:record["sha256"] for name,record in components.items()},
        "runtime":manifest["runtime"],"feature_profile":manifest["build_profile"],
        "component_features":{name:len(record["feature_names"]) for name,record in components.items()},
        "fusion":manifest["fusion"],"decoder":manifest["decoder"],
        "training_matches":manifest["training_matches"],"validation_match":manifest["validation_match"],
        "frames":len(labels),"prediction_codes_sha256":hashlib.sha256(labels.tobytes()).hexdigest(),
        "posterior_sha256":hashlib.sha256(posterior.tobytes()).hexdigest(),
        "event_posterior_sha256":hashlib.sha256(events.tobytes()).hexdigest(),
        "labels_or_status_used_as_inputs":False,"fit_or_calibration_performed":False}
    if temporal_provenance:
        metadata["temporal_training_metadata"] = temporal_provenance
    if is_joint:
        metadata.update(event_source="tcn",architecture=JOINT_ARCHITECTURE,
                        state_class_order=list(JOINT_STATE_CLASSES),event_class_order=list(JOINT_EVENT_CLASSES))
    return labels, metadata
