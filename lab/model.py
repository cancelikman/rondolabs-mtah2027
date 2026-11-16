"""Dondurulmus referans model: 5 macli TCN + HGB4, 50:50 karisim, yonlu Viterbi.

experiments/can_five_candidate_audit.py icindeki tarifin sadelestirilmis hali.
Agir kisim (ozellik + ag) bir kez calisir ve olasiliklar diske yazilir; cozucu
ayarlari sonradan saniyeler icinde yeniden denenebilir (bkz. lab decode).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def posteriors(geometry, checkpoint, parent_bundle, threads=2):
    """(state[n,3] = P(H,A,D), event[n,4]) dondurur. Etiket/GT okumaz."""
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    from experiments import tcn_external as base
    from experiments.joint_transition_external import JointTemporalNet, infer_joint
    from src.bundle import _geometry_only, _symmetric_probability
    from src.features import build_features

    parent = json.loads((Path(parent_bundle) / "manifest.json").read_text(encoding="utf-8"))
    hgb = joblib.load(Path(parent_bundle) / parent["components"]["hgb"]["file"])
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = cp["feature_names"]
    network = JointTemporalNet(2 * len(names), int(cp.get("channels", 64))).eval()
    network.load_state_dict(cp["state_dict"])
    mean, scale = cp["mean"].numpy(), cp["scale"].numpy()
    permutation, sign = base.exchange_schema(names)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    geometry = _geometry_only(geometry)
    with threadpool_limits(limits=threads):
        x, built = build_features(geometry, profile="contact_motion")
        if built != names:
            raise ValueError("Ozellik semasi checkpoint ile uyusmuyor")
        temporal, event = infer_joint(network, dict(data=geometry, x=x), mean, scale,
                                      permutation, sign, torch.device("cpu"))
        index = {name: i for i, name in enumerate(built)}
        hgb_x = np.take(x, [index[name] for name in hgb["feature_names"]], axis=1)
        geometric = _symmetric_probability(hgb, hgb_x, hgb["feature_names"])
    state = .5 * (geometric + temporal)
    return state.astype(np.float32), event.astype(np.float32), parent["decoder"]


def decode(state, event, geometry, settings):
    """Olasiliklardan H/A/D dizisi. -1: puanlanmayan kareler."""
    from experiments.transition_external import decode as directed_decode
    from src.features import blocks_of
    safe_state, safe_event = np.array(state, dtype=float), np.array(event, dtype=float)
    inside = np.zeros(len(safe_state), dtype=bool)
    for a, b in blocks_of(geometry):
        inside[a:b] = True
    bias = float(settings.get("dead_bias", 1.))
    if "dead_bias_visible" in settings or "dead_bias_missing" in settings:
        # Gorunurluge bagli onsel (4. tur G2): top gorunurken k_visible, kayipken k_missing.
        visible = np.isfinite(np.asarray(geometry["ball"], dtype=float)).all(axis=1)
        k_vis = float(settings.get("dead_bias_visible", bias))
        k_mis = float(settings.get("dead_bias_missing", bias))
        factor = np.where(visible, k_vis, k_mis)
        safe_state[inside, 2] *= factor[inside]
        safe_state[inside] /= safe_state[inside].sum(axis=1, keepdims=True)
    elif bias != 1.:
        # Olu top onseli: P(D) carpilip yeniden normalize edilir. Deger yalnizca dis maclarda secilir.
        safe_state[inside, 2] *= bias
        safe_state[inside] /= safe_state[inside].sum(axis=1, keepdims=True)
    safe_state[~inside] = [1 / 3, 1 / 3, 1 / 3]
    safe_event[~inside] = [1, 0, 0, 0]
    return directed_decode(safe_state, safe_event, geometry, settings)
