"""G3: independent open test data (test only, never training or constant selection).

SkillCorner Open Data (github.com/SkillCorner/opendata, MIT; A-League 2024/25, 10 fps broadcast
tracking). Labels, written down before scoring:
  * dead (D) = every frame outside a `phases_of_play` interval;
  * inside a phase: the frame's `possession.group` (home team = H, away team = A); when it is
    null the last non-null group inside the same phase is carried, and before the first one the
    phase's `team_in_possession_id` is used.
Ball positions with is_detected = false become missing. Coordinates are centre-origin metres and
are shifted to corner-origin with the per-match pitch size.

Metrica sample data (github.com/metrica-sports/sample-data, games 1-2, 25 fps CSV). Labels from
the event data, written down before scoring:
  * (i)  `control`: possession changes only on a controlled action (PASS, CARRY, SHOT, RECOVERY,
          SET PIECE) of the other team;
  * (ii) `duel`: as (i), and also on a CHALLENGE with subtype WON by the other team;
  * dead: from a BALL OUT event, or a FAULT RECEIVED / fault event, to the next SET PIECE start.
Coordinates are pitch fractions; converted with a 105 x 68 m pitch (the documented size).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .matches import Match, ROOT

SC_DIR = ROOT / "data/open/skillcorner/data/matches"
MT_DIR = ROOT / "data/open/metrica/data"


# ------------------------------------------------------------------ SkillCorner

def skillcorner_ids():
    return sorted(p.name for p in SC_DIR.iterdir() if p.is_dir())


def load_skillcorner_native(match_id):
    folder = SC_DIR / str(match_id)
    meta = json.loads((folder / ("%s_match.json" % match_id)).read_text(encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    players = [p for p in meta["players"]]
    slot = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], dtype=np.int8)
    periods = [(p["start_frame"], p["end_frame"]) for p in meta["match_periods"]]
    phases = []
    with open(folder / ("%s_phases_of_play.csv" % match_id), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            phases.append((int(r["frame_start"]), int(r["frame_end"]),
                           0 if int(r["team_in_possession_id"]) == home else 1))
    rows = {}
    with open(folder / ("%s_tracking_extrapolated.jsonl" % match_id), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[int(r["frame"])] = r
    balls, plays, labels, blocks, frames = [], [], [], [], []
    cursor = 0
    for s, e in periods:
        n = e - s + 1
        ball = np.full((n, 2), np.nan, np.float32)
        pl = np.full((n, len(players), 2), np.nan, np.float32)
        grp = np.full(n, -1, np.int8)
        for i, fr in enumerate(range(s, e + 1)):
            r = rows.get(fr)
            if r is None:
                continue
            b = r.get("ball_data") or {}
            if b.get("x") is not None and b.get("is_detected"):
                ball[i] = (b["x"] + L / 2, b["y"] + W / 2)
            for p in r.get("player_data") or []:
                j = slot.get(p["player_id"])
                if j is not None and p.get("x") is not None:
                    pl[i, j] = (p["x"] + L / 2, p["y"] + W / 2)
            g = (r.get("possession") or {}).get("group")
            grp[i] = 0 if g == "home team" else 1 if g == "away team" else -1
        lab = np.full(n, 2, np.int8)          # dead outside phases
        for ps, pe, owner in phases:
            a, b2 = max(ps, s) - s, min(pe, e) - s
            if b2 < a:
                continue
            current = owner
            for i in range(a, b2 + 1):
                if grp[i] >= 0:
                    current = grp[i]
                lab[i] = current
        balls.append(ball); plays.append(pl); labels.append(lab)
        blocks.append((cursor, cursor + n)); cursor += n
    return dict(ball=np.concatenate(balls), players=np.concatenate(plays), labels=np.concatenate(labels),
                blocks=np.asarray(blocks), fps=10., pitch=np.array([L, W]), team_of_track=team,
                match_id="SC%s" % match_id, source="SkillCorner")


def load_skillcorner(match_id):
    from experiments.can_skillcorner_audit import resample_geometry
    from src.bundle import _geometry_only
    data = load_skillcorner_native(match_id)
    geometry, mapping = resample_geometry(_geometry_only(data))
    n25 = len(geometry["ball"])
    dead25 = np.interp(np.arange(n25), mapping[:, 1], (data["labels"] == 2).astype(float)) > .5
    m = Match("SC%s" % match_id, geometry, dead25, data, mapping)
    return m


def coverage(data):
    lab = np.asarray(data["labels"])
    fps = float(data["fps"])
    ball = np.asarray(data["ball"], float)
    graded = lab >= 0
    spells = 0
    for a, b in data["blocks"]:
        x = lab[a:b]
        spells += int(((np.r_[True, x[1:] != x[:-1]]) & np.isin(x, [0, 1])).sum())
    mins = graded.sum() / fps / 60
    return dict(frames=int(len(lab)), H=round(float((lab == 0).mean()), 4), A=round(float((lab == 1).mean()), 4),
                D=round(float((lab == 2).mean()), 4), possessions_per_min=round(spells / mins, 3),
                ball_visible=round(float(np.isfinite(ball).all(1).mean()), 4),
                ball_x_range=[round(float(np.nanmin(ball[:, 0])), 2), round(float(np.nanmax(ball[:, 0])), 2)],
                ball_y_range=[round(float(np.nanmin(ball[:, 1])), 2), round(float(np.nanmax(ball[:, 1])), 2)],
                pitch=[float(v) for v in data["pitch"]])


# ------------------------------------------------------------------ Metrica

CONTROLLED = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def _metrica_tracking(path):
    """Metrica raw CSV: 3 header rows; columns Period, Frame, Time [s], players (x,y)..., Ball (x,y)."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd); next(rd); header = next(rd)
        rows = [r for r in rd]
    arr = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r] for r in rows], dtype=np.float64)
    names = header
    player_cols = [i for i, h in enumerate(names) if h.startswith("Player")]
    ball_col = names.index("Ball")
    return arr, player_cols, ball_col


def load_metrica_native(game, definition="control"):
    g = MT_DIR / ("Sample_Game_%d" % game)
    home, hp, hb = _metrica_tracking(g / ("Sample_Game_%d_RawTrackingData_Home_Team.csv" % game))
    away, ap, ab = _metrica_tracking(g / ("Sample_Game_%d_RawTrackingData_Away_Team.csv" % game))
    n = min(len(home), len(away))
    home, away = home[:n], away[:n]
    L, W = 105., 68.
    pl = []
    for arr, cols in ((home, hp), (away, ap)):
        for c in cols:
            pl.append(np.stack([arr[:, c] * L, arr[:, c + 1] * W], axis=1))
    players = np.stack(pl, axis=1).astype(np.float32)
    team = np.array([0] * len(hp) + [1] * len(ap), dtype=np.int8)
    ball = np.stack([home[:, hb] * L, home[:, hb + 1] * W], axis=1).astype(np.float32)
    period = home[:, 0].astype(int)
    frames = home[:, 1].astype(int)
    index = {int(f): i for i, f in enumerate(frames)}
    lab = np.full(n, -1, np.int8)
    events = []
    with open(g / ("Sample_Game_%d_RawEventsData.csv" % game), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            events.append(r)
    owner, dead = None, False
    marks = []   # (row, owner or 'D')
    if definition == "challenge":
        # Round 8: owner changes when a team strings >= 3 consecutive controlled events (PASS,
        # CARRY, SHOT; RECOVERY alone does not count) without an opponent controlled event in
        # between, or carries the ball >= 3 s; restarts (SET PIECE) give a clear owner.
        run_team, run_start, run_n = None, None, 0
        for r in events:
            typ = r["Type"].strip().upper()
            team_ev = 0 if r["Team"].strip().lower() == "home" else 1
            start = index.get(int(r["Start Frame"]))
            end = index.get(int(r["End Frame"])) if r.get("End Frame") else start
            if start is None:
                continue
            if typ == "SET PIECE":
                dead = False
                owner = team_ev
                marks.append((start, owner))
                run_team, run_start, run_n = team_ev, start, 1
                continue
            if typ in ("BALL OUT", "FAULT RECEIVED"):
                dead = True
                marks.append((start, "D"))
                run_team, run_n = None, 0
                continue
            if dead or typ not in ("PASS", "CARRY", "SHOT"):
                continue
            if team_ev != run_team:
                run_team, run_start, run_n = team_ev, start, 0
            run_n += 1
            long_carry = typ == "CARRY" and end is not None and (end - start) >= 3 * 25
            if team_ev != owner and (run_n >= 3 or long_carry):
                owner = team_ev
                marks.append((run_start, owner))
        for (i, v), nxt in zip(marks, marks[1:] + [(n, None)]):
            lab[i:nxt[0]] = 2 if v == "D" else v
        events = []
    for r in events:
        typ, sub = r["Type"].strip().upper(), (r["Subtype"] or "").strip().upper()
        team_ev = 0 if r["Team"].strip().lower() == "home" else 1
        start = index.get(int(r["Start Frame"]))
        if start is None:
            continue
        if typ == "SET PIECE":
            dead = False
            owner = team_ev
            marks.append((start, owner))
            continue
        if typ in ("BALL OUT", "FAULT RECEIVED"):
            dead = True
            marks.append((start, "D"))
            continue
        if dead:
            continue
        change = typ in CONTROLLED or (definition == "duel" and typ == "CHALLENGE" and "WON" in sub)
        if change and team_ev != owner:
            owner = team_ev
            marks.append((start, owner))
    for (i, v), nxt in zip(marks, marks[1:] + [(n, None)]):
        lab[i:nxt[0]] = 2 if v == "D" else v
    blocks, cursor = [], 0
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(idx[0]), int(idx[-1]) + 1))
    ball_ok = np.isfinite(ball).all(1)
    return dict(ball=np.where(ball_ok[:, None], ball, np.nan).astype(np.float32), players=players,
                labels=lab, blocks=np.asarray(blocks), fps=25., pitch=np.array([L, W]), team_of_track=team,
                match_id="MT%d_%s" % (game, definition), source="Metrica")


def load_metrica(game, definition="control"):
    from src.bundle import _geometry_only
    data = load_metrica_native(game, definition)
    geo = _geometry_only(data)
    return Match(data["match_id"], geo, np.asarray(data["labels"]) == 2, data)
