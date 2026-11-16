"""Read target assets locally, with strict timeline checks and no row logging.

This is an inference-only reader. It never reads possession ground truth, fits a
transformer, or compares predictions with the video.
"""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from games import GAME, GAMES, asset_paths, half_video_blocks
from starter.baseline import team_of_each_track


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_target(data_dir):
    paths = asset_paths(data_dir=str(data_dir))
    teams = team_of_each_track(paths["atd_meta"])
    n = GAMES[GAME]["n_video_frames"]
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    pitch = np.array(GAMES[GAME]["pitch"])
    count = 0
    with open(paths["atd_raw"]) as f:
        for i, line in enumerate(f):
            if i >= n:
                raise ValueError("too many tracking rows; never trim/pad")
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != i+1:
                raise ValueError("tracking frame ids are not contiguous from one")
            fields = parts[1].rstrip(";").split(";")
            if len(fields) != len(teams):
                raise ValueError("track count differs from metadata")
            for j, value in enumerate(fields):
                xyz = value.split(",")
                if len(xyz) != 3:
                    raise ValueError("invalid player observation")
                players[i, j] = [float(xyz[0]), float(xyz[1])]
            xyz = parts[2].rstrip(";").split(",")
            if len(xyz) != 3:
                raise ValueError("invalid ball observation")
            ball[i] = [float(xyz[0]), float(xyz[1])]
            count += 1
    if count != n:
        raise ValueError("tracking row count mismatch; never trim/pad")
    players *= pitch
    ball *= pitch
    blocks = np.array([[a, b+1] for _, a, b in half_video_blocks(GAME)])
    return {"ball": ball, "players": players, "team_of_track": teams,
            "fps": GAMES[GAME]["fps"], "pitch": pitch, "blocks": blocks,
            "source": "competition-inference-only"}


def write_labels(path, labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if not np.isin(labels, [-1, 0, 1, 2, 3]).all():
        raise ValueError("invalid labels")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alphabet = np.array(["H", "A", "D", "X"])
    labels = np.where(labels < 0, 3, labels)
    if not np.isin(labels, np.arange(4)).all():
        raise ValueError("invalid labels")
    with path.open("w", newline="") as f:
        f.write("frame,label\n")
        for i, label in enumerate(labels):
            f.write("%d,%s\n" % (i, alphabet[label]))
