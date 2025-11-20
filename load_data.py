import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from typing import List, Tuple
import os

@dataclass
class Frame:
    frame: int
    t_s: float
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    speed_mps: float
    player_id: int = -1


def load_frames_from_csv(path: str, fps: float, player_name: str = None):
    """
    Load tracking data from comprehensive_data.csv.
    - Assigns unique player_id per player_name
    - Filters out 'Unknown'
    - Computes velocity, acceleration, and speed automatically
    """

    df = pd.read_csv(path)

    # --- Drop unknown players ---
    df = df[df["player_name"].astype(str).str.lower() != "unknown"]

    # --- Assign numeric player_id per unique player_name ---
    unique_players = sorted(df["player_name"].unique())
    player_id_map = {name: i for i, name in enumerate(unique_players, start=1)}
    df["player_id"] = df["player_name"].map(player_id_map)

    # --- Optional filter for a specific player ---
    if player_name is not None:
        df = df[df["player_name"].astype(str).str.lower() == player_name.lower()]
        if df.empty:
            available = df["player_name"].unique()
            raise ValueError(
                f"No data found for player_name='{player_name}'.\n"
                f"Available names: {list(available)}"
            )

    # Sort by frame
    df = df.sort_values(["player_id", "frame_number"]).reset_index(drop=True)

    # Compute time (s)
    df["t_s"] = df["frame_number"] / fps

    # --- Rename columns to engine format ---
    df = df.rename(columns={
        "court_x_normalized": "x",
        "court_y_normalized": "y",
    })

    # --- Compute velocities, accelerations, speed ---
    df["vx"] = df.groupby("player_id")["x"].diff().fillna(0) * fps
    df["vy"] = df.groupby("player_id")["y"].diff().fillna(0) * fps
    df["ax"] = df.groupby("player_id")["vx"].diff().fillna(0) * fps
    df["ay"] = df.groupby("player_id")["vy"].diff().fillna(0) * fps
    df["speed_mps"] = (df["vx"] ** 2 + df["vy"] ** 2) ** 0.5

    # --- Convert to Frame objects ---
    frames = [
        Frame(
            frame=int(row["frame_number"]),
            t_s=float(row["t_s"]),
            x=float(row["x"]),
            y=float(row["y"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            ax=float(row["ax"]),
            ay=float(row["ay"]),
            speed_mps=float(row["speed_mps"]),
            player_id=int(row["player_id"]),
        )
        for _, row in df.iterrows()
    ]

    print(f"[ok] Loaded {len(frames)} frames from {path} with {len(unique_players)} players")
    return frames

import json
from dataclasses import dataclass

@dataclass
class Bounce:
    frame: int
    t_s: float
    x: float
    y: float

def load_bounces_from_json(path: str, fps: float):
    """Load basketball bounce coordinates from JSON (frame→coords)."""
    with open(path, "r") as f:
        data = json.load(f)

    bounces = []
    for frame_str, v in data.items():
        frame = int(frame_str)
        t_s = frame / fps
        x, y = v["court_coord"]
        bounces.append(Bounce(frame, t_s, x, y))

    print(f"[ok] Loaded {len(bounces)} bounces from {path}")
    return bounces

def load_frames_from_detection_json(
    path: str,
    fps: float = 25.0,
):
    """
    Load tracking data from detection JSON.

    - Handles multiple players correctly.
    - Computes per-player velocity (vx, vy) and acceleration (ax, ay).
    - Supports pre-normalized or pixel-space coordinates.
    """
    with open(path, "r") as f:
        data = json.load(f)

    frames = []

    for frame_str, content in data.items():
        frame = int(frame_str)
        object_ids = content.get("object_id", [])
        coords = content.get("court_coord", None)

        if len(object_ids) == 0 or len(coords) == 0:
            continue

        # ensure matching length
        n = min(len(object_ids), len(coords))

        for i in range(n):
            oid = object_ids[i]
            cx, cy = coords[i]

            frames.append({
                "frame_number": frame,
                "player_id": int(oid),  # ← use actual object_id
                "x": float(cx),
                "y": float(cy),
            })

    if not frames:
        print(f"[warn] No valid detections in {path}")
        return []

    # --- Convert to DataFrame ---
    df = pd.DataFrame(frames).sort_values(["player_id", "frame_number"]).reset_index(drop=True)
    df["t_s"] = df["frame_number"] / fps

    # --- Compute per-player velocities and accelerations ---
    df["vx"] = df.groupby("player_id")["x"].diff().fillna(0) * fps
    df["vy"] = df.groupby("player_id")["y"].diff().fillna(0) * fps
    df["ax"] = df.groupby("player_id")["vx"].diff().fillna(0) * fps
    df["ay"] = df.groupby("player_id")["vy"].diff().fillna(0) * fps
    df["speed_mps"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)

    # --- Convert to Frame objects ---
    frames_out = [
        Frame(
            frame=int(row["frame_number"]),
            t_s=float(row["t_s"]),
            x=float(row["x"]),
            y=float(row["y"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            ax=float(row["ax"]),
            ay=float(row["ay"]),
            speed_mps=float(row["speed_mps"]),
            player_id=int(row["player_id"]),
        )
        for _, row in df.iterrows()
    ]

    print(f"[ok] Loaded {len(frames_out)} frames from {path} (players={df['player_id'].nunique()})")
    return frames_out

def load_frames(
    path: str,
    source_type: str = "csv",
    fps: float = 25.0,
    player_name: str = None,
):
    """
    Unified loader for either CSV (tracking) or JSON (detection) data.
    - source_type: 'csv' or 'json'
    - resolution: (width, height) only used for JSON scaling
    """
    ext = os.path.splitext(path)[1].lower()

    if source_type == "csv" or ext == ".csv":
        print(f"[data] Loading CSV: {path}")
        return load_frames_from_csv(path, fps=fps, player_name=player_name)

    elif source_type == "json" or ext == ".json":
        print(f"[data] Loading JSON detections: {path}")
        return load_frames_from_detection_json(path, fps=fps)

    else:
        raise ValueError(f"Unsupported source_type='{source_type}' or file type '{ext}'")
    
def load_score_events(path: str, fps: float):
    """
    Load shot-made events from CSV without headers.
    4th column: timestamp (HH:MM:SS)
    7th column: 'X Point Made'
    Returns list of dicts: [{'t_s': time_in_seconds, 'points': 2 or 3}, ...]
    """
    df = pd.read_csv(path, header=None)
    events = []

    for _, row in df.iterrows():
        time_str = str(row[3]).strip()
        event_str = str(row[6]).strip()

        # Parse timestamp to seconds
        try:
            h, m, s = map(int, time_str.split(":"))
            t_s = h * 3600 + m * 60 + s
        except Exception:
            continue

        # Extract number of points
        try:
            points = int(event_str.split()[0])
        except Exception:
            points = 0

        events.append({"t_s": t_s, "points": points})

    print(f"[ok] Loaded {len(events)} scoring events from {path}")
    return events