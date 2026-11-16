"""Exact team exchange of an already-built geometric feature matrix.

A schema permutation replaces the second feature-extraction pass. Team-named
columns exchange partners; signed home advantages negate. Shared coordinates,
ball dynamics and label-blind context values remain unchanged. No data fitting.
"""
from functools import lru_cache
import re
import numpy as np

_TEAM_TOKEN = re.compile(r"team([01])(?=_|$)")


@lru_cache(maxsize=32)
def _swap_plan(names):
    if any(not isinstance(name, str) for name in names):
        raise ValueError("feature names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    index = {name: i for i, name in enumerate(names)}
    permutation, odd = [], []
    for name in names:
        partner = _TEAM_TOKEN.sub(lambda match: "team"+str(1-int(match.group(1))), name)
        if partner not in index:
            raise ValueError("missing team-exchange feature partner: %s" % partner)
        permutation.append(index[partner])
        # All current antisymmetric scalars, including their context suffixes.
        is_distance_advantage = name == "distance_advantage_home" or name.startswith("distance_advantage_home_")
        is_reach_advantage = name.startswith("reach_") and "advantage_home" in name
        if "advantage_home" in name and not (is_distance_advantage or is_reach_advantage):
            raise ValueError("unrecognized home-advantage schema: %s" % name)
        odd.append(is_distance_advantage or is_reach_advantage)
    permutation = np.asarray(permutation, dtype=np.intp)
    odd = np.asarray(odd, dtype=bool)
    if not np.array_equal(permutation[permutation], np.arange(len(names))):
        raise ValueError("team feature permutation must be an involution")
    if not np.array_equal(odd[permutation], odd):
        raise ValueError("team partners disagree on sign semantics")
    return permutation, np.flatnonzero(odd)


def swap_feature_matrix(x, names):
    """Return the features for exchanged home/away membership, without rebuilding.

    Supports the current geometry/contact/contact_motion schemas and appended
    reachability features. Row order, shape, floating dtype and NaN payloads are
    retained. Only finite signed-advantage values negate; infinities and NaNs
    are preserved. The input is never mutated. Missing partners fail closed.
    """
    x = np.asarray(x)
    names = tuple(names)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("feature matrix/name dimensions disagree")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("feature matrix must have floating dtype")
    permutation, odd = _swap_plan(names)
    result = np.take(x, permutation, axis=1)
    if len(odd):
        values = result[:, odd]
        np.negative(values, out=values, where=np.isfinite(values))
        result[:, odd] = values
    return result
