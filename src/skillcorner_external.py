"""Pinned SkillCorner auxiliary ownership data; never manufacture dead-ball labels.

Official SkillCorner/opendata, MIT, copyright2020 SkillCorner. Automatic provider
ownership is weak supervision, not human possession ground truth. Only explicit
group AND covering half-open phase-owner agreement produces H/A. Everything else
is unknown. Tracking is native10Hz and detected observations only; no video input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from .idsse_external import _derivative_speed

SOURCE_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SOURCE_REPO = "https://github.com/SkillCorner/opendata"
DECLARED_AUXILIARY_TRAIN = (1874553, 1886347, 1899585, 1925299)
SCHEMA_VERSION = 1
FPS = 10.0
KNOWN_TRAILING_METADATA_OMISSIONS = {1899585: 1}
COMMON_SHA256 = {
    "LICENSE": "3384cc66ffd008209759af5e6328677eabded72756fb1f56f4f82e68e9b3c6eb",
    "matches.json": "4dd7a62f966046a8fde2563e69fd044997d471b213118aed436cb23ab1fdfad0",
}
# Immutable release-file receipts from the pinned Git/LFS source, never target data.
MATCH_SHA256 = {
    1874553: {
        "match.json": "e6974e402779d8be44e75bf88ef2a3acfd4d025b87b3fc57f518fa826a645ad8",
        "phases_of_play.csv": "e70054b64202a3f318b7208edec9a329da1f4cfcafeeb72ae197176c4a043a8c",
        "tracking_extrapolated.jsonl": "ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265",
    },
    1886347: {
        "match.json": "30c608e78663e9f490b3545479a3ce67b171f418c5133d5609cb249a2466abb2",
        "phases_of_play.csv": "d1c3066f7ed0598069d0d1b317bb30261eca3d7786fd9d540985f395e4c8aea9",
        "tracking_extrapolated.jsonl": "3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41",
    },
    1899585: {
        "match.json": "41de8cb74a0512fcf0cbaaa5c615ae3e7dcaaf6745b9c78a050b59b906b1cb06",
        "phases_of_play.csv": "e40562010c1b7c4aa6e0dbc17425dd236a1d82ce538ec03e77367a74e3f96b7e",
        "tracking_extrapolated.jsonl": "90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100",
    },
    1925299: {
        "match.json": "f4345b32c0fdd7c9990df70050b3a74f57b1992cf762e20f2a4d309bd2fba7d2",
        "phases_of_play.csv": "a1682a736a9cf1c311cb799f0a949f7e023dca4ebaa9dd294b138e8929271274",
        "tracking_extrapolated.jsonl": "48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _match_id(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("invalid auxiliary fixture id")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("auxiliary fixture id must be an integer") from exc
    if str(value) != str(result) or result not in DECLARED_AUXILIARY_TRAIN:
        raise ValueError("only the four declared SkillCorner auxiliary training fixtures are accepted")
    if result not in MATCH_SHA256:
        raise ValueError("fixture source receipts have not been registered")
    return result


def _source_url(name, match_id=None):
    raw = f"https://raw.githubusercontent.com/SkillCorner/opendata/{SOURCE_COMMIT}/"
    if match_id is None:
        return raw + ("LICENSE" if name == "LICENSE" else "data/matches.json")
    relative = f"data/matches/{match_id}/{match_id}_{name}"
    if name.endswith("jsonl"):
        return f"https://media.githubusercontent.com/media/SkillCorner/opendata/{SOURCE_COMMIT}/" + relative
    return raw + relative


def _checked_file(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"pinned external source SHA256 mismatch: {Path(path).name}")


def download_match(data_dir, match_id):
    match_id = _match_id(match_id)
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = [(name, value, None) for name, value in COMMON_SHA256.items()]
    files += [(name, value, match_id) for name, value in MATCH_SHA256[match_id].items()]
    for name, expected, mid in files:
        path = root / (name if mid is None else f"{mid}_{name}")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(_source_url(name, mid), timeout=90) as response, temporary.open("wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
            _checked_file(temporary, expected)
            temporary.replace(path)
        _checked_file(path, expected)
    return root / f"{match_id}_tracking_extrapolated.jsonl"


def _timestamp(value):
    if value is None:
        return np.nan
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid native SkillCorner timestamp") from exc


def _xy(value):
    try:
        xy = np.asarray([value["x"], value["y"]], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid tracking x/y schema") from exc
    return xy if np.isfinite(xy).all() else np.asarray([np.nan, np.nan])


def _detected(value):
    result = value.get("is_detected")
    if result is not True and result is not False and result is not None:
        raise ValueError("is_detected must be boolean or null")
    return result is True


def parse_records(metadata, phase_rows, records, match_id):
    """Pure parser for testing; production entry additionally pins every source hash."""
    match_id = _match_id(match_id)
    if metadata.get("id") != match_id:
        raise ValueError("metadata and declared fixture disagree")
    edition = metadata["competition_edition"]
    if edition["competition"]["id"] != 61 or edition["season"]["name"] != "2024/2025":
        raise ValueError("expected only declared other-game A-League2024/25 data")
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    if home == away:
        raise ValueError("distinct home/away teams required")
    team_codes = {home: 0, away: 1}
    pitch = np.asarray([metadata["pitch_length"], metadata["pitch_width"]], dtype=float)
    if not np.isfinite(pitch).all() or not (90 <= pitch[0] <= 120 and 45 <= pitch[1] <= 90):
        raise ValueError("invalid metric pitch dimensions")
    player_team = {}
    for player in metadata["players"]:
        ident = player["id"]
        if ident in player_team or player["team_id"] not in team_codes:
            raise ValueError("duplicate player id or unknown player team")
        player_team[ident] = team_codes[player["team_id"]]
    if set(player_team.values()) != {0, 1}:
        raise ValueError("both teams must have metadata players")
    tracks = sorted(player_team, key=lambda x: (player_team[x], x))
    track_index = {ident: i for i, ident in enumerate(tracks)}
    period_bounds = {}
    for row in metadata["match_periods"]:
        period, start, end = int(row["period"]), int(row["start_frame"]), int(row["end_frame"])
        if period not in (1, 2) or period in period_bounds or start < 0 or end < start:
            raise ValueError("invalid metadata period bounds")
        period_bounds[period] = (start, end + 1)  # records include end_frame
    if set(period_bounds) != {1, 2}:
        raise ValueError("both regulation periods required")
    n = max(end for start, end in period_bounds.values())
    expected_periods = np.zeros(n, dtype=np.int8)
    for period, (start, end) in sorted(period_bounds.items()):
        if np.any(expected_periods[start:end]):
            raise ValueError("overlapping metadata periods")
        expected_periods[start:end] = period
    phase_owner = np.full(n, -1, dtype=np.int8)
    phase_covered = np.zeros(n, dtype=bool)
    phase_statistics = Counter()
    for row in phase_rows:
        if int(row["match_id"]) != match_id:
            raise ValueError("phase belongs to a different fixture")
        start, end, period = int(row["frame_start"]), int(row["frame_end"]), int(row["period"])
        if not (0 <= start < end <= n) or period not in period_bounds:
            raise ValueError("phase bounds outside recording or invalid period")
        # Some official phase rows cross halftime despite declaring period1.
        # Accept only their intersection with that declared period; the remainder
        # is unknown, never carried into another period or interpreted as dead.
        lo, hi = period_bounds[period]
        clipped_start, clipped_end = max(start, lo), min(end, hi)
        discarded = (end - start) - max(0, clipped_end - clipped_start)
        phase_statistics["phase_frames_outside_declared_period_ignored"] += discarded
        if discarded:
            phase_statistics["phase_rows_crossing_declared_period"] += 1
        start, end = clipped_start, clipped_end
        if start >= end:
            continue
        if phase_covered[start:end].any():
            raise ValueError("overlapping phase intervals are ambiguous")
        phase_covered[start:end] = True
        owner_text = row["team_in_possession_id"]
        if owner_text not in (None, ""):
            owner_id = int(owner_text)
            if owner_id not in team_codes:
                raise ValueError("unknown phase-owner team")
            phase_owner[start:end] = team_codes[owner_id]
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(tracks), 2), np.nan, dtype=np.float32)
    timestamps = np.full(n, np.nan, dtype=np.float64)
    periods = np.zeros(n, dtype=np.int8)
    labels = np.full(n, -1, dtype=np.int8)
    groups = {None: -1, "home team": 0, "away team": 1}
    stats = phase_statistics.copy()
    count = 0
    for count, record in enumerate(records, start=1):
        i = count - 1
        if i >= n or type(record.get("frame")) is not int or record["frame"] != i:
            raise ValueError("native frames must be contiguous and ordered from0")
        period = record.get("period")
        expected = int(expected_periods[i])
        if period is not None and type(period) is not int:
            raise ValueError("native period must be integer or null")
        if (0 if period is None else period) != expected:
            raise ValueError("record period disagrees with metadata")
        groups_value = record["possession"]["group"]
        if groups_value not in groups:
            raise ValueError("unknown automatic possession-group enum")
        stats["group:" + str(groups_value)] += 1
        if period is None:
            continue  # no observations/ownership are asserted between periods
        periods[i] = period
        timestamps[i] = _timestamp(record["timestamp"])
        if not np.isfinite(timestamps[i]):
            raise ValueError("active period requires timestamp")
        if i and expected_periods[i - 1] == period:
            if not np.isclose(timestamps[i] - timestamps[i - 1], 1 / FPS, atol=1e-7, rtol=0):
                raise ValueError("source must remain native10Hz")
        bdata = record["ball_data"]
        if _detected(bdata):
            ball[i] = _xy(bdata) + pitch / 2
        seen = set()
        for player in record["player_data"]:
            ident = player["player_id"]
            if ident not in track_index or ident in seen:
                raise ValueError("unregistered or duplicate player id in frame")
            seen.add(ident)
            if _detected(player):
                players[i, track_index[ident]] = _xy(player) + pitch / 2
        group = groups[groups_value]
        if phase_covered[i]:
            stats["frames_inside_phase"] += 1
            if group >= 0 and phase_owner[i] == group:
                labels[i] = group
            elif group >= 0:
                stats["group_phase_disagreements"] += 1
        elif group >= 0:
            stats["explicit_group_outside_phase"] += 1
    if count != n:
        omitted = n - count
        if omitted != KNOWN_TRAILING_METADATA_OMISSIONS.get(match_id, 0):
            raise ValueError("tracking ends before declared metadata endpoint")
        # Pinned fixture1899585 ends one record before metadata end_frame. Keep
        # only actual native records; do not fabricate a label/observation.
        stats["known_metadata_trailing_record_absent"] = omitted
        n = count
        ball, players, labels = ball[:n], players[:n], labels[:n]
        timestamps, periods = timestamps[:n], periods[:n]
    blocks = np.asarray([(a, min(b, n)) for a, b in sorted(period_bounds.values()) if a < n], dtype=np.int64)
    result = {
        "source": np.asarray("SkillCorner"), "source_commit": np.asarray(SOURCE_COMMIT),
        "label_quality": np.asarray("weak_automatic_provider_ownership"),
        "match_id": np.asarray(str(match_id)), "fps": np.asarray(FPS),
        "ball": ball, "players": players,
        "team_of_track": np.asarray([player_team[p] for p in tracks], dtype=np.int8),
        "labels": labels, "timestamps": timestamps, "periods": periods,
        "frame_ids": np.arange(n, dtype=np.int64), "blocks": blocks,
        "pitch": pitch.astype(np.float32),
        "ball_speed": _derivative_speed(ball, FPS, blocks),
        "player_speed": _derivative_speed(players, FPS, blocks),
    }
    stats.update({"n_frames": n, "n_tracks": len(tracks),
                  "ball_detected_frames": int(np.isfinite(ball).all(axis=1).sum()),
                  "accepted_weak_home_frames": int((labels == 0).sum()),
                  "accepted_weak_away_frames": int((labels == 1).sum()),
                  "unknown_frames": int((labels < 0).sum())})
    return result, dict(stats)


def _validate_arrays(data, match_id):
    if str(data.get("source", "")) != "SkillCorner" or str(data.get("match_id", "")) != str(match_id):
        raise ValueError("SkillCorner auxiliary provenance mismatch")
    if str(data.get("source_commit", "")) != SOURCE_COMMIT or float(data["fps"]) != FPS:
        raise ValueError("source commit or native sampling mismatch")
    if str(data.get("label_quality", "")) != "weak_automatic_provider_ownership":
        raise ValueError("automatic labels must retain weak-quality tag")
    if not np.isin(data["labels"], [-1, 0, 1]).all():
        raise ValueError("SkillCorner auxiliary labels may never contain D=2")


def load_match(data_dir, match_id, *, cache=True, force=False):
    match_id = _match_id(match_id)
    root = Path(data_dir)
    for name, expected in COMMON_SHA256.items():
        _checked_file(root / name, expected)
    matches = json.loads((root / "matches.json").read_text())
    if len([m for m in matches if m["id"] == match_id and m["competition_id"] == 61 and m["season_id"] == 95]) != 1:
        raise ValueError("fixture not registered in pinned A-League match list")
    sources = {}
    for name, expected in MATCH_SHA256[match_id].items():
        path = root / f"{match_id}_{name}"
        _checked_file(path, expected)
        sources[name] = {"sha256": expected, "bytes": path.stat().st_size,
                         "url": _source_url(name, match_id)}
    path = root / f"SC{match_id}_external_v{SCHEMA_VERSION}.npz"
    if cache and not force and path.exists():
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("sources") != sources:
            raise ValueError("stale or altered external cache provenance")
        _checked_file(path, receipt["cache_sha256"])
        with np.load(path, allow_pickle=False) as saved:
            data = dict(saved)
        _validate_arrays(data, match_id)
        return data
    metadata = json.loads((root / f"{match_id}_match.json").read_text())
    with (root / f"{match_id}_phases_of_play.csv").open(newline="") as handle:
        phases = list(csv.DictReader(handle))
    with (root / f"{match_id}_tracking_extrapolated.jsonl").open() as handle:
        data, statistics = parse_records(metadata, phases, (json.loads(line) for line in handle), match_id)
    _validate_arrays(data, match_id)
    if cache:
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temporary.replace(path)
        receipt = {"schema_version": SCHEMA_VERSION, "source": "SkillCorner", "source_commit": SOURCE_COMMIT,
                   "match_id": str(match_id), "license": "MIT", "sources": sources,
                   "cache_sha256": sha256(path), "fps": FPS, "statistics": statistics,
                   "label_provenance": "weak automatic provider labels; explicit group agrees with covering half-open phase owner",
                   "unknown_policy": "outside-phase/null/disagreement/period-null remain-1; neverD", "detected_only": True,
                   "identity_policy": "persistent player IDs join observations only; never features", "units": "metres corner-origin; derivative m/s"}
        temporary_json = path.with_suffix(".json.part")
        temporary_json.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary_json.replace(path.with_suffix(".json"))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--match", nargs="+", type=int, choices=DECLARED_AUXILIARY_TRAIN, default=list(DECLARED_AUXILIARY_TRAIN))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for mid in args.match:
        if args.download:
            download_match(args.data_dir, mid)
        data = load_match(args.data_dir, mid)
        print(json.dumps({"match_id": mid, "n_frames": len(data["ball"]),
                          "known_weak_ownership_frames": int((data["labels"] >= 0).sum()),
                          "ball_detected_frames": int(np.isfinite(data["ball"]).all(axis=1).sum())}), flush=True)


if __name__ == "__main__":
    main()
