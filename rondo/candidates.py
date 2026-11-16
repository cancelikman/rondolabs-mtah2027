"""Candidate models. Each yields (state[n,3], event[n,4]) posteriors from geometry only.

The heavy part (features + networks) produces separate TCN and HGB posteriors so that
mixing variants can be re-scored from the cache without re-running the networks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .matches import ROOT

PARENT = ROOT / "models/ahmetcan_joint_checkpoint"
ART = ROOT / "data/artifacts"
REF_TCN = ART / "can_five_match_spell_tcn_v1/five/best.pt"


def network_posteriors(geometry, checkpoint, parent, extra_features=None, threads=2, hgb_path=None):
    """Returns dict(tcn=[n,3], hgb=[n,3], event=[n,4]). Never reads labels."""
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    from experiments import tcn_external as base
    from experiments.joint_transition_external import JointTemporalNet, infer_joint
    from src.bundle import _geometry_only, _symmetric_probability
    from src.features import build_features

    if hgb_path is None:
        manifest = json.loads((Path(parent) / "manifest.json").read_text(encoding="utf-8"))
        hgb_path = Path(parent) / manifest["components"]["hgb"]["file"]
    hgb = joblib.load(hgb_path)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = list(cp["feature_names"])
    network = JointTemporalNet(2 * len(names), int(cp.get("channels", 64) or 64)).eval()
    network.load_state_dict(cp["state_dict"])
    mean, scale = cp["mean"].numpy(), cp["scale"].numpy()
    permutation, sign = base.exchange_schema(names)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    geometry = _geometry_only(geometry)
    with threadpool_limits(limits=threads):
        x, built = build_features(geometry, profile="contact_motion")
        hgb_index = {name: i for i, name in enumerate(built)}
        hgb_names = list(hgb["feature_names"])
        if hgb.get("feature_profile") == "contact_control":
            from .control_features import control_features
            cx, cnames = control_features(geometry)
            base_names = [n for n in hgb_names if n not in cnames]
            hgb_x = np.concatenate([np.take(x, [hgb_index[n] for n in base_names], axis=1), cx], axis=1)
        elif hgb.get("feature_profile") == "contact_gap":
            from .gap_features import gap_context
            gx, gnames = gap_context(geometry)
            base_names = [n for n in hgb_names if n not in gnames]
            hgb_x = np.concatenate([np.take(x, [hgb_index[n] for n in base_names], axis=1), gx], axis=1)
            if base_names + gnames != hgb_names:
                raise ValueError("contact_gap schema order mismatch")
        else:
            hgb_x = np.take(x, [hgb_index[name] for name in hgb_names], axis=1)
        if extra_features is not None:
            extra, extra_names = extra_features(geometry)
            x = np.concatenate([x, np.asarray(extra, dtype=np.float32)], axis=1)
            built = [*built, *extra_names]
        if list(built) != names:
            raise ValueError("feature schema does not match checkpoint")
        temporal, event = infer_joint(network, dict(data=geometry, x=x), mean, scale,
                                      permutation, sign, torch.device("cpu"))
        geometric = _symmetric_probability(hgb, hgb_x, hgb["feature_names"])
    return dict(tcn=temporal.astype(np.float32), hgb=geometric.astype(np.float32),
                event=event.astype(np.float32))


def _missing(geometry):
    return ~np.isfinite(np.asarray(geometry["ball"], dtype=float)).all(1)


class Candidate:
    def __init__(self, name, checkpoint=REF_TCN, parent=PARENT, note="", extra=None,
                 tcn_weight=.5, hgb_when_hidden=0., network_key=None, hgb=None):
        self.name, self.checkpoint, self.parent, self.note = name, Path(checkpoint), Path(parent), note
        self.hgb = Path(hgb) if hgb else None
        self.extra, self.tcn_weight, self.hgb_when_hidden = extra, tcn_weight, hgb_when_hidden
        # candidates sharing networks share the cache
        self.network_key = network_key or name
        self.decoder = json.loads((self.parent / "manifest.json").read_text(encoding="utf-8"))["decoder"]

    def networks(self, geometry):
        import os
        threads = int(os.environ.get("RONDO_THREADS", "2"))
        return network_posteriors(geometry, self.checkpoint, self.parent, self.extra, threads=threads,
                                  hgb_path=self.hgb)

    def combine(self, parts, geometry):
        tcn, hgb = np.asarray(parts["tcn"], float), np.asarray(parts["hgb"], float)
        state = self.tcn_weight * tcn + (1 - self.tcn_weight) * hgb
        if self.hgb_when_hidden:
            m = _missing(geometry) & np.isfinite(state).all(1)
            team = state[m, :2]
            alive = team.sum(axis=1, keepdims=True)
            cond = np.divide(team, alive, out=np.full_like(team, .5), where=alive > 0)
            dead = (1 - self.hgb_when_hidden) * state[m, 2] + self.hgb_when_hidden * hgb[m, 2]
            state[m, :2] = cond * (1 - dead[:, None])
            state[m, 2] = dead
        return state.astype(np.float32), np.asarray(parts["event"], np.float32)

    def postprocess(self, pred, geometry):
        return pred

    def describe(self):
        return dict(name=self.name, checkpoint=str(self.checkpoint.relative_to(ROOT)),
                    parent=str(self.parent.relative_to(ROOT)), decoder=self.decoder, note=self.note,
                    tcn_weight=self.tcn_weight, hgb_when_hidden=self.hgb_when_hidden)


def _player_motion(geometry):
    from src.player_motion_summary import player_motion_summary
    return player_motion_summary(geometry)


REGISTRY = {
    "ref": lambda: Candidate("ref", note="five-match spell TCN + HGB4 (candidate A)"),
    "ref_hgbhidden": lambda: Candidate("ref_hgbhidden", hgb_when_hidden=.5, network_key="ref",
                                       note="prefer HGB dead posterior where the ball is missing"),
    "motion": lambda: Candidate("motion", ART / "can_five_match_player_motion_v1/five/best.pt",
                                extra=_player_motion, note="player-motion TCN"),
    "multiocc": lambda: Candidate("multiocc", ART / "can_five_match_multi_occlusion_v1/five/best.pt",
                                  note="multi-pattern occlusion TCN"),
    "dropout": lambda: Candidate("dropout", ART / "can_five_match_player_ball_dropout_v2/five/best.pt",
                                 note="player/ball dropout TCN"),
    "joint": lambda: Candidate("joint", PARENT / "tcn.pt", note="ahmetcan joint package"),
    # S5 retrained on dead-ball realistic views (rondo.train_c)
    "c1": lambda: Candidate("c1", ART / "rondo/cand_c_v1/tcn.pt", hgb=ART / "rondo/cand_c_v1/hgb.joblib",
                            note="S5: new TCN + new HGB"),
    "c2": lambda: Candidate("c2", ART / "rondo/cand_c_v2_wcsf/tcn.pt", hgb=ART / "rondo/cand_c_v2_wcsf/hgb.joblib",
                            note="S5 + Groningen first half in training (score WCSF_H2 only)"),
    "c1_tcn_refhgb":lambda: Candidate("c1_tcn_refhgb", ART / "rondo/cand_c_v1/tcn.pt",
                                       note="S5: new TCN + original HGB4"),
}


def _blend(name, a, b, note):
    """Equal posterior mixture of two candidates (Derekko's safe option)."""
    ca, cb = REGISTRY[a](), REGISTRY[b]()

    class Blend(Candidate):
        def networks(self, geometry):
            raise RuntimeError("blend uses cached parts of its members")

    c = Blend(name, note=note)
    c.members = (ca, cb)
    return c


REGISTRY["c1_blend_ref"] = lambda: _blend("c1_blend_ref", "c1", "ref", "S5: 50:50 mixture of c1 and ref")


def get(name):
    return REGISTRY[name]()
REGISTRY["c2_blend_ref"] = lambda: _blend("c2_blend_ref", "c2", "ref", "S5 arm 2: 50:50 mixture of c2 and ref")
REGISTRY["c3"] = lambda: Candidate("c3", ART / "rondo/cand_c3/tcn.pt", hgb=ART / "rondo/cand_c3/hgb.joblib",
                                   note="G4: C3 TCN + HGB on physical corruptions, IDSSE only")
REGISTRY["c3gap"] = lambda: Candidate("c3gap", ART / "rondo/cand_c3/tcn.pt", hgb=ART / "rondo/cand_c3_gaphgb/hgb.joblib",
                                      note="G5: C3 TCN + gap-context HGB")
REGISTRY["c3_blend_ref"] = lambda: _blend("c3_blend_ref", "c3", "ref", "G4: 50:50 ref + C3")
REGISTRY["c3gap_blend_ref"] = lambda: _blend("c3gap_blend_ref", "c3gap", "ref", "G5: 50:50 ref + C3gap")


def _mean(name, member_names, note, weights=None):
    """Average of several candidates' posteriors (used for seed ensembles and blends)."""
    c = _blend(name, member_names[0], member_names[1], note)
    c.members = tuple(REGISTRY[m]() for m in member_names)
    return c


REGISTRY["c3_s2"] = lambda: Candidate("c3_s2", ART / "rondo/cand_c3_s2/tcn.pt", hgb=ART / "rondo/cand_c3/hgb.joblib",
                                      note="G8 seed 2")
REGISTRY["c3_s3"] = lambda: Candidate("c3_s3", ART / "rondo/cand_c3_s3/tcn.pt", hgb=ART / "rondo/cand_c3/hgb.joblib",
                                      note="G8 seed 3")
REGISTRY["c3_seeds"] = lambda: _mean("c3_seeds", ["c3", "c3_s2", "c3_s3"], "G8: three-seed C3")
REGISTRY["c3_seeds_blend_ref"] = lambda: _mean("c3_seeds_blend_ref", ["c3_seeds", "ref"], "G8: ref + three-seed C3")
REGISTRY["c3w"] = lambda: Candidate("c3w", ART / "rondo/cand_c3w/tcn.pt", hgb=ART / "rondo/cand_c3w/hgb.joblib",
                                    note="G4 final recipe incl. full Groningen")
REGISTRY["c3w_blend_ref"] = lambda: _blend("c3w_blend_ref", "c3w", "ref", "ref + C3w")
REGISTRY["c3w_s2"] = lambda: Candidate("c3w_s2", ART / "rondo/cand_c3w_s2/tcn.pt", hgb=ART / "rondo/cand_c3w/hgb.joblib",
                                       note="C3w seed 2")
REGISTRY["c3w_s3"] = lambda: Candidate("c3w_s3", ART / "rondo/cand_c3w_s3/tcn.pt", hgb=ART / "rondo/cand_c3w/hgb.joblib",
                                       note="C3w seed 3")
REGISTRY["c3w_seeds"] = lambda: _mean("c3w_seeds", ["c3w", "c3w_s2", "c3w_s3"], "three-seed C3w")
REGISTRY["c3w_seeds_blend_ref"] = lambda: _mean("c3w_seeds_blend_ref", ["c3w_seeds", "ref"], "ref + three-seed C3w")
for _w in (.5, .35, .2):   # TCN weight; HGB weight = 1 - w  (0.5 / 0.65 / 0.8)
    REGISTRY["b2a_tw%g" % _w] = (lambda w=_w: Candidate("b2a_tw%g" % w, ART / "rondo/cand_c3w/tcn.pt",
                                                        hgb=ART / "rondo/cand_b2a/hgb.joblib", tcn_weight=w,
                                                        network_key="b2a", note="B2a HGB + C3w TCN"))
    REGISTRY["b2a_tw%g_blend_ref" % _w] = (lambda w=_w: _blend("b2a_tw%g_blend_ref" % w, "b2a_tw%g" % w, "ref",
                                                               "ref + (C3w TCN, B2a HGB)"))
REGISTRY["r3"] = lambda: Candidate("r3", ART / "rondo/cand_c3w/tcn.pt", hgb=ART / "rondo/cand_r3/hgb.joblib",
                                   note="R3: C3w TCN + HGB on competition-definition labels with control features")
REGISTRY["r3_blend_ref"] = lambda: _blend("r3_blend_ref", "r3", "ref", "ref + R3 member")
for _w in (30, 50):
    REGISTRY["a3_w%d" % _w] = (lambda w=_w: Candidate("a3_w%d" % w, ART / "rondo/cand_c3w/tcn.pt",
                                                      hgb=ART / ("rondo/cand_a3_w%d/hgb.joblib" % w),
                                                      note="A3: C3w TCN + HGB with twin (%d%% weight)" % w))
    REGISTRY["a3_w%d_blend_ref" % _w] = (lambda w=_w: _blend("a3_w%d_blend_ref" % w, "a3_w%d" % w, "ref", "ref + A3"))
REGISTRY["a4"] = lambda: Candidate("a4", ART / "rondo/cand_a4/tcn.pt", hgb=ART / "rondo/cand_a3_w50/hgb.joblib",
                                   note="A4: twin fine-tuned TCN + A3 HGB")
REGISTRY["a4_blend_ref"] = lambda: _blend("a4_blend_ref", "a4", "ref", "ref + A4")
