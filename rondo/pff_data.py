"""PFF FC / Gradient FIFA World Cup 2022 (64 matches) -> rondo Match objects.

External data about other games, obtained by the team from PFF FC under its data licence.
Nothing from it is committed: the zips and the 25 Hz cache live outside the repo
(RONDO_PFF_DIR, default ~/OneDrive/Desktop/pff). See docs for how to obtain it.

Steps (run from the repo root):

    python -m rondo.pff_data convert --workers 6          # zips -> <dir>/cache/PF<id>.npz (once, ~15 min)
    python -m rondo.pff_data fingerprint --out reports/rondo/pff_fingerprint.json
    python -m rondo.pff_data check PF3859                  # one match: coverage + label stats

Geometry: tracking frames (29.97 fps, video time = frameNum / fps) are resampled per period to
25 Hz by linear interpolation between neighbouring frames (gap <= 0.1 s, else NaN). Smoothed
player and ball positions; centre-origin metres shifted to [0, L] x [0, W] as for SkillCorner.

Labels are derived from the event JSON only (never from tracking):
  * possession: from each on-the-ball (OTB / kick-off) event's start, the team of that player
    (home -> 0, away -> 1), carried forward until the next on-the-ball event of any team;
  * dead (2): from a stop marker until the restart kick (start of the next set-piece OTB, or of the
    next OTB after an OUT). Stop markers: OUT, END, a foul on a possession event, and, when a set
    piece follows with no marker, `fallback` seconds after the previous event's release;
  * before the first kick-off of each period and after END: dead; video-missing spans: -1.
"""
from __future__ import annotations

import argparse
import bz2
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PFF_DIR = Path(os.environ.get("RONDO_PFF_DIR", str(Path.home() / "OneDrive" / "Desktop" / "pff")))
CACHE = Path(os.environ.get("RONDO_PFF_CACHE", str(PFF_DIR / "cache")))
FPS25 = 25.0
SET_PIECES = {"K", "T", "F", "G", "C", "P", "D"}
LABEL_VARIANTS = {"v1": dict(fallback=1.0), "v0": dict(fallback=0.0), "v2": dict(fallback=2.0)}
DEFAULT_VARIANT = "v1"

try:
    import orjson as _oj

    def _loads(s):
        return _oj.loads(s)
except Exception:  # pragma: no cover
    def _loads(s):
        return json.loads(s)


# ------------------------------------------------------------------ zip index

def zips():
    z = sorted(PFF_DIR.glob("*.zip"))
    if not z:
        raise FileNotFoundError("no PFF zips in %s (set RONDO_PFF_DIR)" % PFF_DIR)
    return z


def index():
    """game id -> {kind: (zip path, member)} for tracking / metadata / roster / events."""
    out = {}
    for zp in zips():
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                parts = n.split("/")
                if len(parts) < 3 or not parts[-1]:
                    continue
                leaf = parts[-1]
                gid = leaf.split(".")[0]
                if not gid.isdigit():
                    continue
                if parts[1] == "Tracking Data" and leaf.endswith(".jsonl.bz2"):
                    kind = "tracking"
                elif parts[1] == "Metadata":
                    kind = "metadata"
                elif parts[1] == "Rosters":
                    kind = "roster"
                elif parts[1] == "Event Data" and leaf.endswith(".json"):
                    # several dated re-releases exist: keep the newest (folder name sorts poorly, use length tie-break)
                    kind = "events"
                    sub = parts[2] if len(parts) == 4 else ""
                    prev = out.get(gid, {}).get("events_sub")
                    if prev is not None and _date_key(prev) >= _date_key(sub):
                        continue
                    out.setdefault(gid, {})["events_sub"] = sub
                else:
                    continue
                out.setdefault(gid, {})[kind] = (str(zp), n)
    return {g: v for g, v in out.items() if {"tracking", "metadata", "roster", "events"} <= set(v)}


def _date_key(sub):
    if not sub:
        return (0, 0, 0)
    import datetime
    try:
        d = datetime.datetime.strptime(sub, "%B %d, %Y")
        return (d.year, d.month, d.day)
    except ValueError:
        return (0, 0, 0)


def _read(entry):
    zp, member = entry
    with zipfile.ZipFile(zp) as z:
        return z.read(member)


# ------------------------------------------------------------------ convert

def _events_table(raw, fps):
    """Compact game-event table + foul times from the event JSON (video seconds)."""
    rows = _loads(raw)
    ge = {}
    fouls = []
    for r in rows:
        g = r.get("gameEvents") or {}
        gid = r.get("gameEventId")
        if gid is None:
            continue
        st, en = r.get("startTime"), r.get("endTime")
        if gid not in ge:
            ge[gid] = dict(type=g.get("gameEventType"), sp=g.get("setpieceType"), home=g.get("homeTeam"),
                           period=g.get("period"), start=st, end=en, vid=bool(g.get("videoMissing")))
        else:
            e = ge[gid]
            if st is not None and (e["start"] is None or st < e["start"]):
                e["start"] = st
            if en is not None and (e["end"] is None or en > e["end"]):
                e["end"] = en
        f = r.get("fouls") or {}
        if f.get("foulType") is not None and r.get("eventTime") is not None:
            fouls.append(float(r["eventTime"]))
    ev = sorted((e for e in ge.values() if e["start"] is not None), key=lambda e: e["start"])
    types = ["OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OUT", "END", "VID", "G",
             "SUB", "ON", "OFF"]
    code = {t: i for i, t in enumerate(types)}
    tab = np.array([[code.get(e["type"], -1), 1 if (e["sp"] in SET_PIECES) else 0,
                     -1 if e["home"] is None else (0 if e["home"] else 1), e["period"] or -1,
                     float(e["start"]), float(e["end"] if e["end"] is not None else e["start"]), int(e["vid"])]
                    for e in ev], dtype=np.float64).reshape(-1, 7)
    return tab, np.array(sorted(fouls), dtype=np.float64), np.array(types)


def convert_one(gid, entries, out_dir):
    t0 = time.time()
    meta = _loads(_read(entries["metadata"]))
    meta = meta[0] if isinstance(meta, list) else meta
    roster = _loads(_read(entries["roster"]))
    fps = float(meta["fps"])
    home_id = str(meta["homeTeam"]["id"])
    pitch = meta["stadium"]["pitches"][0]
    L, W = float(pitch["length"]), float(pitch["width"])
    slot, team = {}, []
    for p in roster:
        key = (0 if str(p["team"]["id"]) == home_id else 1, str(p["shirtNumber"]))
        if key not in slot:
            slot[key] = len(team)
            team.append(key[0])

    fn, per, ball, pls = [], [], [], []
    zp, member = entries["tracking"]
    with zipfile.ZipFile(zp) as z, z.open(member) as raw:
        stream = io.BufferedReader(bz2.BZ2File(raw), buffer_size=1 << 22)
        for line in stream:
            r = _loads(line)
            p = r.get("period")
            if p is None:
                continue
            fn.append(int(r["frameNum"]))
            per.append(int(p))
            b = r.get("ballsSmoothed") or {}
            ball.append((b.get("x"), b.get("y")))
            row = {}
            for side, k in ((0, "homePlayersSmoothed"), (1, "awayPlayersSmoothed")):
                for q in r.get(k) or []:
                    if q.get("x") is None:
                        continue
                    key = (side, str(q.get("jerseyNum")))
                    if key not in slot:
                        slot[key] = len(team)
                        team.append(side)
                    row[slot[key]] = (q["x"], q["y"], q.get("visibility") == "VISIBLE")
            pls.append(row)
    n, K = len(fn), len(team)
    fn = np.asarray(fn, np.int64)
    per = np.asarray(per, np.int16)
    B = np.array([[np.nan if v is None else v for v in xy] for xy in ball], np.float64).reshape(n, 2)
    P = np.full((n, K, 2), np.nan, np.float32)
    V = np.zeros((n, K), bool)
    for i, row in enumerate(pls):
        for j, (x, y, v) in row.items():
            P[i, j] = (x, y)
            V[i, j] = v
    del pls, ball
    order = np.lexsort((fn, per))
    fn, per, B, P, V = fn[order], per[order], B[order], P[order], V[order]
    tn = fn / fps

    balls, players, blocks, times, periods, visible = [], [], [], [], [], []
    cursor = 0
    for p in sorted(set(per.tolist())):
        m = per == p
        t = tn[m]
        keep = np.r_[True, np.diff(t) > 0]
        t, b, pp, vv = t[keep], B[m][keep], P[m][keep], V[m][keep]
        if len(t) < 10:
            continue
        g = np.arange(np.ceil(t[0] * FPS25), np.floor(t[-1] * FPS25) + 1) / FPS25
        hi = np.clip(np.searchsorted(t, g, side="left"), 1, len(t) - 1)
        lo = hi - 1
        dt = t[hi] - t[lo]
        w = np.where(dt > 0, (g - t[lo]) / np.maximum(dt, 1e-9), 0.)
        ok = dt <= 0.1 + 1e-6

        def interp(x):
            wl = w.reshape((-1,) + (1,) * (x.ndim - 1))
            y = x[lo] + (x[hi] - x[lo]) * wl
            y[~ok] = np.nan
            return y.astype(np.float32)

        bb = interp(b)
        bb[:, 0] += L / 2
        bb[:, 1] += W / 2
        pl = interp(pp)
        pl[..., 0] += L / 2
        pl[..., 1] += W / 2
        visible.append(vv[lo] & vv[hi] & ok[:, None])
        balls.append(bb)
        players.append(pl)
        times.append(g)
        periods.append(np.full(len(g), p, np.int16))
        blocks.append((cursor, cursor + len(g)))
        cursor += len(g)
    ev, fouls, types = _events_table(_read(entries["events"]), fps)
    out = dict(ball=np.concatenate(balls), players=np.concatenate(players), team_of_track=np.asarray(team, np.int8),
               fps=np.asarray(FPS25), pitch=np.asarray([L, W], np.float32), blocks=np.asarray(blocks, np.int64),
               t=np.concatenate(times), period=np.concatenate(periods), visible=np.concatenate(visible), events=ev, fouls=fouls, event_types=types,
               native_fps=np.asarray(fps), game_id=np.asarray(int(gid)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ("PF%s.tmp.npz" % gid)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, out_dir / ("PF%s.npz" % gid))
    return dict(game=gid, frames25=int(cursor), tracks=K, events=int(len(ev)), s=round(time.time() - t0, 1))


def _convert_task(args):
    gid, entries, out_dir = args
    try:
        return convert_one(gid, entries, Path(out_dir))
    except Exception as e:  # keep going, report at the end
        return dict(game=gid, error=repr(e))


def convert(workers=4, only=None, redo=False):
    from multiprocessing import Pool
    idx = index()
    todo = [(g, e, str(CACHE)) for g, e in sorted(idx.items(), key=lambda kv: int(kv[0]))
            if (only is None or g in only) and (redo or not (CACHE / ("PF%s.npz" % g)).exists())]
    print(json.dumps(dict(found=len(idx), todo=len(todo), cache=str(CACHE))), flush=True)
    done = []
    with Pool(max(1, workers)) as pool:
        for r in pool.imap_unordered(_convert_task, todo):
            done.append(r)
            print(json.dumps(r), flush=True)
    bad = [r for r in done if "error" in r]
    print(json.dumps(dict(converted=len(done) - len(bad), errors=len(bad))), flush=True)
    return done


# ------------------------------------------------------------------ labels

def match_ids():
    return sorted(p.stem[2:] for p in CACHE.glob("PF*.npz") if not p.stem.endswith(".tmp"))


def load_cached(gid):
    with np.load(CACHE / ("PF%s.npz" % gid), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def derive_labels(d, variant=DEFAULT_VARIANT):
    """25 Hz labels H=0 / A=1 / D=2 / ungraded=-1 from the event table (see module doc)."""
    fallback = LABEL_VARIANTS[variant]["fallback"]
    t = d["t"]
    types = list(d["event_types"])
    ev = d["events"]
    code = {name: i for i, name in enumerate(types)}
    OTB, OUT, END, VID = code["OTB"], code["OUT"], code["END"], code["VID"]
    KO = {code[k] for k in ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")}
    fouls = d["fouls"]
    lab = np.full(len(t), 2, np.int8)
    period = d["period"]
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        tp = t[rows]
        e = ev[ev[:, 3] == p]
        if not len(e):
            lab[rows] = -1
            continue
        out = np.full(len(rows), 2, np.int8)
        # possession carry from on-the-ball events
        onball = e[np.isin(e[:, 0], [OTB] + list(KO)) & (e[:, 2] >= 0)]
        if len(onball):
            k = np.searchsorted(onball[:, 4], tp, side="right") - 1
            has = k >= 0
            out[has] = onball[k[has], 2].astype(np.int8)
            out[~has] = 2                          # before the period's first kick-off
        # dead spans: stop marker -> restart
        last_release = None
        stop = None
        for row in e:
            typ, sp, start, end = int(row[0]), int(row[1]), row[4], row[5]
            if typ in (OUT, END):
                stop = start if stop is None else min(stop, start)
                if typ == END:
                    out[tp >= start] = 2
                continue
            if typ == OTB or typ in KO:
                restart = typ in KO or sp == 1 or stop is not None
                if restart and last_release is not None:
                    s = stop
                    if s is None:
                        f = fouls[(fouls > last_release - 1e-6) & (fouls < start)]
                        s = float(f[0]) if len(f) else (last_release + fallback if sp == 1 or typ in KO else None)
                    if s is not None and s < start:
                        out[(tp >= s) & (tp < start)] = 2
                stop = None
                last_release = end
            if typ == VID:
                out[(tp >= start) & (tp <= max(end, start))] = -1
        lab[rows] = out
    nobody = ~np.isfinite(d["players"][..., 0]).any(1)
    lab[nobody] = -1
    return lab


def fill_ball(ball, blocks):
    """PFF tracks the ball in ~91% of live frames but ~2% of dead frames, so a missing ball would
    leak the label. Fill every gap inside a period by linear interpolation (hold at the ends)."""
    out = np.array(ball, np.float32, copy=True)
    for a, b in np.asarray(blocks, int):
        seg = out[a:b]
        ok = np.isfinite(seg).all(1)
        if ok.sum() < 2:
            continue
        x = np.flatnonzero(ok)
        r = np.arange(len(seg))
        for c in (0, 1):
            seg[:, c] = np.interp(r, x, seg[ok, c])
    return out


def load_pff(name):
    """name: '3859' or '3859@v2' (label variant)."""
    from .matches import Match
    gid, _, variant = str(name).partition("@")
    d = load_cached(gid)
    lab = derive_labels(d, variant or DEFAULT_VARIANT)
    geometry = {k: d[k] for k in ("ball", "players", "team_of_track", "fps", "pitch", "blocks")}
    geometry["ball"] = fill_ball(d["ball"], d["blocks"])
    scorer = dict(geometry, labels=lab)
    m = Match("PF%s" % name, geometry, lab == 2, scorer, None)
    m.cache_name = "PF%s" % gid
    return m


# ------------------------------------------------------------------ checks

def stats(d, lab):
    from .d1_fingerprint import fingerprint
    blocks = [tuple(map(int, b)) for b in d["blocks"]]
    lab25 = np.concatenate([np.r_[lab[a:b], -1] for a, b in blocks]).astype(np.int8)
    f = fingerprint(lab25)
    graded = lab >= 0
    f.update(frames=int(len(lab)), H=float((lab == 0).mean()), A=float((lab == 1).mean()), D=float((lab == 2).mean()),
             ungraded=float((~graded).mean()), ball_visible=float(np.isfinite(d["ball"]).all(1).mean()),
             players_mean=float(np.isfinite(d["players"][..., 0]).sum(1).mean()))
    return f


ALIGN_OFFSETS = (-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8)


def align(d):
    """Median distance (m) from the tracked ball to the nearest player of the event's team at
    on-the-ball starts, for several time offsets. The minimum should sit near 0 s if events and
    tracking share a clock; a shifted minimum means a lag to correct."""
    t, ball, P, team = d["t"], d["ball"], d["players"], d["team_of_track"]
    types = list(d["event_types"])
    e = d["events"]
    e = e[(e[:, 0] == types.index("OTB")) & (e[:, 2] >= 0)]
    out = {}
    for off in ALIGN_OFFSETS:
        k = np.searchsorted(t, e[:, 4] + off)
        ok = (k > 0) & (k < len(t))
        k, side = k[ok], e[ok, 2].astype(int)
        good = np.abs(t[k] - (e[ok, 4] + off)) < 0.05
        k, side = k[good], side[good]
        dist = np.linalg.norm(P[k] - ball[k][:, None, :], axis=2)
        dist[team[None, :] != side[:, None]] = np.nan
        m = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        m = m[np.isfinite(m)]
        out[str(off)] = float(np.median(m)) if len(m) else float("nan")
    return out


def fingerprint_all(out, limit=None):
    from .d1_fingerprint import OFFICIAL, SHIFTS, average
    from .matches import merged_labels
    ids = match_ids()[:limit] if limit else match_ids()
    per = {}
    for gid in ids:
        d = load_cached(gid)
        blocks = [tuple(map(int, b)) for b in d["blocks"]]
        for v in LABEL_VARIANTS:
            lab = derive_labels(d, v)
            per.setdefault(v + "|raw", []).append(stats(d, lab))
            for s in (.5, 1., 2.):
                per.setdefault("%s|merge<%gs" % (v, s), []).append(stats(d, merged_labels(lab, blocks, s, FPS25)))
        print("done", gid, flush=True)
    keep = ("curve", "distance", "possessions_per_live_min", "mean_s", "median_s", "share_lt")
    rep = {k: average([{kk: f[kk] for kk in keep} for f in v]) for k, v in per.items()}
    al = [align(load_cached(g)) for g in ids]
    alignment = {o: float(np.nanmedian([a[o] for a in al])) for o in al[0]}
    print("event/tracking alignment (median ball-to-team distance, m, by offset s):", alignment, flush=True)
    for k, v in per.items():
        rep[k].update({x: float(np.mean([f[x] for f in v])) for x in ("H", "A", "D", "ungraded", "ball_visible", "players_mean")})
    best = min(rep, key=lambda k: rep[k]["distance"])
    res = dict(official=OFFICIAL, matches=len(ids), variants=rep, closest=best, alignment=alignment)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, f in sorted(rep.items(), key=lambda kv: kv[1]["distance"]):
        print("%-16s d=%.4f curve %s per_min %.2f median %.1f D %.3f ungraded %.3f" % (
            k, f["distance"], [round(f["curve"][s], 4) for s in SHIFTS], f["possessions_per_live_min"],
            f["median_s"], f["D"], f["ungraded"]))
    print("closest", best)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--only", nargs="*")
    c.add_argument("--redo", action="store_true", help="rebuild existing cache files (e.g. to add `visible`)")
    sub.add_parser("index")
    f = sub.add_parser("fingerprint")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--limit", type=int)
    k = sub.add_parser("check")
    k.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.workers, set(a.only) if a.only else None, a.redo)
    elif a.cmd == "index":
        idx = index()
        print(json.dumps(dict(dir=str(PFF_DIR), games=len(idx),
                              events_release={g: v.get("events_sub") for g, v in sorted(idx.items())}), indent=1))
    elif a.cmd == "fingerprint":
        fingerprint_all(a.out, a.limit)
    elif a.cmd == "check":
        gid, _, v = a.name.replace("PF", "").partition("@")
        d = load_cached(gid)
        r = stats(d, derive_labels(d, v or DEFAULT_VARIANT))
        r["alignment"] = align(d)
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
