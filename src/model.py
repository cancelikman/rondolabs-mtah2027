"""Portable frame classifier and duration-aware Potts/Viterbi decoder."""
import numpy as np
from .features import FEATURE_VERSION, blocks_of, build_features, swap_sides
from .symmetry import swap_feature_matrix


def viterbi(probabilities, switch_cost, dead_switch_cost=None):
    p = np.asarray(probabilities, dtype=float)
    if not len(p):
        return np.zeros(0, dtype=np.int8)
    if p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("expected finite H/A/D probabilities")
    dead_switch_cost = switch_cost if dead_switch_cost is None else dead_switch_cost
    costs = np.full((3, 3), float(switch_cost))
    costs[2, :2] = costs[:2, 2] = dead_switch_cost
    np.fill_diagonal(costs, 0)
    emit = -np.log(np.clip(p, 1e-6, 1))
    previous = np.zeros((len(p), 3), dtype=np.int8)
    score = emit[0]
    for i in range(1, len(p)):
        options = score[:, None] + costs
        best = np.argmin(options, axis=0)
        previous[i] = best
        score = options[best, np.arange(3)] + emit[i]
        score -= score.min()
    out = np.zeros(len(p), dtype=np.int8)
    out[-1] = np.argmin(score)
    for i in range(len(p)-1, 0, -1):
        out[i-1] = previous[i, out[i]]
    return out


def duration_viterbi(probabilities, switch_cost, dead_switch_cost, dead_bins):
    """Exact minimum interior-D duration via an expanded-state chain.

    H/A controls have no minimum duration. Head/tail D may be truncated by the
    recording boundary. No score or constants from the target game enter here.
    """
    if dead_bins <= 1:
        return viterbi(probabilities, switch_cost, dead_switch_cost)
    p = np.asarray(probabilities)
    if not len(p):
        return np.zeros(0, dtype=np.int8)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("expected finite H/A/D probabilities")
    n_states = 2+int(dead_bins)
    costs = np.full((n_states, n_states), np.inf)
    costs[0, 0] = costs[1, 1] = costs[-1, -1] = 0
    costs[0, 1] = costs[1, 0] = switch_cost
    costs[:2, 2] = dead_switch_cost
    costs[-1, :2] = dead_switch_cost
    for state in range(2, n_states-1):
        costs[state, state+1] = 0
    state_labels = np.r_[0, 1, np.repeat(2, dead_bins)]
    emit = -np.log(np.clip(p[:, state_labels], 1e-6, 1))
    score = emit[0].copy()
    score[2:-1] = np.inf
    previous = np.zeros((len(p), n_states), dtype=np.int16)
    for i in range(1, len(p)):
        options = score[:, None]+costs
        best = options.argmin(axis=0)
        previous[i] = best
        score = options[best, np.arange(n_states)] + emit[i]
        score -= score.min()
    path = np.zeros(len(p), dtype=np.int16)
    path[-1] = score.argmin()
    for i in range(len(p)-1, 0, -1):
        path[i-1] = previous[i, path[i]]
    return state_labels[path].astype(np.int8)


def predict_probabilities(model, data):
    if model.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ValueError("model/feature implementation version mismatch")
    if not np.array_equal(model["classifier"].classes_, [0, 1, 2]):
        raise ValueError("classifier must contain H/A/D in exactly that order")
    profile = model.get("feature_profile", "geometry")
    x, names = build_features(data, profile=profile)
    if names != model["feature_names"]:
        raise ValueError("model/feature schema mismatch")
    original = model["classifier"].predict_proba(x)
    # Team-exchange equivariance at inference, irrespective of estimator asymmetry.
    flipped = swap_feature_matrix(x, names)
    swapped = model["classifier"].predict_proba(flipped)[:, [1, 0, 2]]
    return (original + swapped) * .5


def decode(probabilities, data, config):
    out = np.full(len(probabilities), -1, dtype=np.int8)
    stride = int(config.get("stride", 5))
    if stride <= 0:
        raise ValueError("decoder stride must be positive")
    for a, b in blocks_of(data):
        # Average a short bin, then decode; each bin is expanded to its own frames.
        full = (b-a)//stride
        p = probabilities[a:a+full*stride].reshape(full, stride, 3).mean(axis=1)
        if a+full*stride < b:
            p = np.vstack([p, probabilities[a+full*stride:b].mean(axis=0)])
        minimum = float(config.get("min_dead_seconds", 0))
        if not np.isfinite(minimum) or minimum < 0:
            raise ValueError("minimum D duration must be finite and nonnegative")
        bins = max(1, int(np.ceil(minimum*float(data["fps"])/stride)))
        labels = duration_viterbi(p, config["switch_cost"], config["dead_switch_cost"], bins)
        out[a:b] = np.repeat(labels, stride)[:b-a]
    return out
