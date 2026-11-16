"""Dead-ball ceiling on labelled external matches (diagnostic, no training, no choice).

    python -m tools.dead_report            (from the repository root)

For F6 on N10-A and the guard sets (clean and target-like):
  * oracle gain: spellF1 if every frame where truth or prediction is D were set to the truth;
  * recall of true stoppages (D runs >= 1 s, >= 50% of frames predicted D), split by where the
    stoppage starts: ball inside the pitch (fouls, offsides, injuries) or outside / not tracked
    (throw-ins, corners, goal kicks; on SkillCorner D = ball not detected);
  * how many true live frames F6 calls dead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def runs(lab, value):
    m = np.r_[False, lab == value, False]
    e = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(e[::2], e[1::2]))


def main():
    from rondo.r10 import N10A, decode, eval_sets, f6_estimator, item
    from rondo.r11 import f6_parts
    from rondo.phase_prior import gt25
    est = f6_estimator()
    out = {}
    print("| set | F6 spellF1 | oracle-D spellF1 | gain | stoppages in-pitch: n / recall | out or untracked: n / recall | live called dead |")
    print("|---|---|---|---|---|---|---|")
    for key, spec in eval_sets(N10A).items():
        it = item(*spec)
        S6, E = f6_parts(it, est)
        geo = it["geo"]
        pred = decode(S6, E, geo)
        base = it["mm"].score(pred)["spell"]
        y = gt25(it["mm"])
        orc = pred.copy()
        m = (y >= 0) & ((y == 2) | (pred == 2))
        orc[m] = y[m]
        oracle = it["mm"].score(orc)["spell"]
        fps = float(geo["fps"])
        ball = np.asarray(geo["ball"], float)
        L, W = [float(v) for v in geo["pitch"]][:2]
        stats = {"in": [0, 0], "out": [0, 0]}
        for a, b in runs(y, 2):
            if (b - a) < fps:
                continue
            look = ball[max(0, a - int(fps)):a + 1]
            ok = np.isfinite(look).all(1)
            inside = ok.any() and (0.5 < look[ok][-1][0] < L - .5) and (0.5 < look[ok][-1][1] < W - .5)
            k = "in" if inside else "out"
            stats[k][0] += 1
            stats[k][1] += int((pred[a:b] == 2).mean() >= .5)
        live = (y == 0) | (y == 1)
        fd = float((pred[live] == 2).mean())
        out[key] = dict(f6=base, oracle=oracle, gain=oracle - base, stoppages=stats, live_called_dead=fd)
        r = lambda s: "%d / %s" % (s[0], ("%.0f%%" % (100 * s[1] / s[0])) if s[0] else "-")
        print("| %s | %.1f | %.1f | +%.1f | %s | %s | %.1f%% |" % (key, 100 * base, 100 * oracle, 100 * (oracle - base),
                                                             r(stats["in"]), r(stats["out"]), 100 * fd), flush=True)
    (ROOT / "reports/rondo/dead_report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
