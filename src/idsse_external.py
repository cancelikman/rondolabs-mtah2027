"""Download and stream external IDSSE matches without any competition-match access.

Data: Bassek et al. (2025), doi:10.1038/s41597-025-04505-y,
doi:10.6084/m9.figshare.28196177. License: CC BY 4.0, DFL/Sportec.
The three-match protocol is declared before any model is fitted or scored.
Coordinates returned here are metres from the pitch's bottom-left corner.
Provider BallStatus/BallPossession are targets only, never input features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np


MATCH_COMPETITIONS = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
DECLARED_SPLIT = {"train": "J03WOH", "validation": "J03WOY", "test": "J03WPY"}
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
PERIODS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
SCHEMA_VERSION = 1


def source_paths(data_dir: str | Path, match_id: str) -> tuple[Path, Path]:
    competition = MATCH_COMPETITIONS[match_id]
    suffix = f"{competition}_DFL-MAT-{match_id}.xml"
    return (Path(data_dir) / f"DFL_02_01_matchinformation_{suffix}",
            Path(data_dir) / f"DFL_04_03_positions_raw_observed_{suffix}")


def download_match(data_dir: str | Path, match_id: str) -> tuple[Path, Path]:
    """Download publicly licensed other-game data; existing complete files are reused."""
    paths = source_paths(data_dir, match_id)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.is_file():
            continue
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(MIRROR + path.name, partial)
        partial.replace(path)
    return paths


def _metadata(path: Path, match_id: str) -> tuple[dict[str, int], np.ndarray]:
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    assert general is not None and general.attrib["MatchId"] == f"DFL-MAT-{match_id}"
    pitch = root.find(".//Environment")
    assert pitch is not None
    dimensions = np.asarray([float(pitch.attrib["PitchX"]), float(pitch.attrib["PitchY"])])
    player_team = {}
    for team in root.findall(".//Teams/Team"):
        role = team.attrib["Role"]
        if role not in ("home", "guest", "away"):
            raise ValueError(f"Unknown team role: {role}")
        for player in team.findall("./Players/Player"):
            player_team[player.attrib["PersonId"]] = 0 if role == "home" else 1
    if not player_team or set(player_team.values()) != {0, 1}:
        raise ValueError("Metadata must identify players from both teams")
    return player_team, dimensions


def _frames(path: Path):
    """Yield period/object/frame attributes with bounded XML memory."""
    period = obj = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            period = PERIODS.get(element.attrib.get("GameSection"))
            obj = "ball" if element.attrib.get("TeamId") == "BALL" else element.attrib.get("PersonId")
        elif event == "end" and element.tag == "Frame":
            if period is not None and obj is not None:
                yield period, obj, element.attrib
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            element.clear()
            period = obj = None


def _derivative_speed(xy: np.ndarray, fps: float, blocks: np.ndarray) -> np.ndarray:
    """Consecutive finite positions only: no velocity bridge across missing observations."""
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    for start, end in blocks:
        if end - start < 2:
            continue
        delta = np.linalg.norm(np.diff(xy[start:end], axis=0), axis=-1) * fps
        speed[start + 1:end] = delta
        speed[start] = speed[start + 1]
    return speed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_match(data_dir: str | Path, match_id: str, *, cache: bool = True,
               force: bool = False) -> dict[str, np.ndarray]:
    """Return arrays for a fixed external match and persist a provenance sidecar.

    labels: 0=home, 1=away, 2=dead, -1=unknown. Provider possession 0/unknown
    while live is left unknown, never silently converted to away. A period's
    frame-number gaps remain explicit padded NaN/-1 rows. blocks contains
    (start, end_exclusive) pairs; periods and frame_ids retain original keys.
    """
    metadata_path, tracking_path = source_paths(data_dir, match_id)
    cache_path = Path(data_dir) / f"{match_id}_external_v{SCHEMA_VERSION}.npz"
    if cache and not force and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    player_team, pitch = _metadata(metadata_path, match_id)
    ranges: dict[int, list[int]] = {}
    sample_times = []
    used_tracks = set()
    for period, obj, attrs in _frames(tracking_path):
        if obj != "ball" and obj not in player_team:
            continue  # Officials are excluded, as in the production tracking.
        frame_id = int(attrs["N"])
        bounds = ranges.setdefault(period, [frame_id, frame_id])
        bounds[0], bounds[1] = min(bounds[0], frame_id), max(bounds[1], frame_id)
        if obj != "ball":
            used_tracks.add(obj)
        if len(sample_times) < 100 and attrs.get("T"):
            sample_times.append(datetime.fromisoformat(attrs["T"]).timestamp())
    if not ranges or len(sample_times) < 2:
        raise ValueError("No usable external tracking frames")
    dt = np.diff(sample_times)
    dt = dt[dt > 0]
    fps = round(1.0 / float(np.median(dt)), 6)
    if not np.isclose(fps, 25.0, atol=0.01):
        raise ValueError(f"Unexpected sampling frequency: {fps}")
    fps = 25.0
    tracks = sorted(used_tracks, key=lambda track: (player_team[track], track))
    track_index = {track: i for i, track in enumerate(tracks)}
    offsets, blocks, cursor = {}, [], 0
    frame_ids, periods = [], []
    for period, (first, last) in sorted(ranges.items()):
        length = last - first + 1
        offsets[period] = (cursor, first)
        blocks.append((cursor, cursor + length))
        frame_ids.extend(range(first, last + 1))
        periods.extend([period] * length)
        cursor += length
    blocks_array = np.asarray(blocks, dtype=np.int64)
    players = np.full((cursor, len(tracks), 2), np.nan, dtype=np.float32)
    ball = np.full((cursor, 2), np.nan, dtype=np.float32)
    labels = np.full(cursor, -1, dtype=np.int8)
    raw_status, raw_possession = Counter(), Counter()
    for period, obj, attrs in _frames(tracking_path):
        if obj != "ball" and obj not in track_index:
            continue
        offset, first = offsets[period]
        idx = offset + int(attrs["N"]) - first
        try:
            x, y = float(attrs["X"]), float(attrs["Y"])
        except (KeyError, ValueError):
            x = y = np.nan
        xy = np.asarray([x, y], dtype=np.float32) + pitch / 2
        if obj == "ball":
            ball[idx] = xy
            status, possession = attrs.get("BallStatus", ""), attrs.get("BallPossession", "")
            raw_status[status] += 1
            raw_possession[possession] += 1
            if status == "0":
                labels[idx] = 2
            elif status == "1" and possession in ("1", "2"):
                labels[idx] = int(possession) - 1
        else:
            players[idx, track_index[obj]] = xy
    timestamps = np.arange(cursor, dtype=np.float64) / fps
    dataset = {
        "ball": ball, "players": players,
        "team_of_track": np.asarray([player_team[track] for track in tracks], dtype=np.int8),
        "player_speed": _derivative_speed(players, fps, blocks_array),
        "ball_speed": _derivative_speed(ball, fps, blocks_array),
        "labels": labels, "timestamps": timestamps, "fps": np.asarray(fps),
        "blocks": blocks_array, "pitch": pitch.astype(np.float32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "periods": np.asarray(periods, dtype=np.int8),
        "match_id": np.asarray(match_id),
        "source": np.asarray("IDSSE"),
    }
    sidecar = {
        "schema_version": SCHEMA_VERSION, "match_id": match_id,
        "declared_split": DECLARED_SPLIT, "license": "CC BY 4.0",
        "paper_doi": "10.1038/s41597-025-04505-y", "dataset_doi": "10.6084/m9.figshare.28196177",
        "metadata_url": MIRROR + metadata_path.name,
        "tracking_url": MIRROR + tracking_path.name,
        "metadata_sha256": _sha256(metadata_path), "tracking_sha256": _sha256(tracking_path),
        "label_provenance": "External DFL/Sportec provider BallStatus/BallPossession; no target-match data",
        "mapping": {"BallStatus=0": "D=2", "BallStatus=1,BallPossession=1": "H=0",
                    "BallStatus=1,BallPossession=2": "A=1", "otherwise": "unknown=-1"},
        "units": {"coordinates": "metres, shifted from centred DFL by pitch/2",
                  "speeds": "m/s, consecutive coordinate derivatives", "fps": fps},
        "raw_ball_status_counts": dict(raw_status), "raw_ball_possession_counts": dict(raw_possession),
        "n_frames": cursor, "n_tracks": len(tracks), "blocks": blocks,
        "label_counts": {str(int(value)): int(np.sum(labels == value)) for value in np.unique(labels)},
    }
    if cache:
        np.savez_compressed(cache_path, **dataset)
        cache_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--match-id", action="append", choices=sorted(MATCH_COMPETITIONS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for match_id in args.match_id or list(DECLARED_SPLIT.values()):
        if args.download:
            download_match(args.data_dir, match_id)
        data = load_match(args.data_dir, match_id, force=args.force)
        print(json.dumps({"match_id": match_id, "frames": len(data["labels"]),
                          "tracks": data["players"].shape[1], "blocks": data["blocks"].tolist(),
                          "cache": str(args.data_dir / f"{match_id}_external_v{SCHEMA_VERSION}.npz")}))


if __name__ == "__main__":
    main()
