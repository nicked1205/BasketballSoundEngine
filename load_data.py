import pandas as pd
from dataclasses import dataclass

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


def load_frames_from_csv(path: str, fps: float = 30.0, player_name: str = None):
    """
    Load tracking data from comprehensive_data.csv.
    Filters by player_name (string).
    Computes velocity, acceleration, and speed automatically.
    """

    df = pd.read_csv(path)

    # --- Filter by player name ---
    if player_name is not None:
        df = df[df["player_name"].astype(str).str.lower() == player_name.lower()]
        if df.empty:
            available = df["player_name"].unique()
            raise ValueError(
                f"No data found for player_name='{player_name}'.\n"
                f"Available names: {list(available)}"
            )

    # Sort frames in chronological order
    df = df.sort_values("frame_number").reset_index(drop=True)

    # Compute time (s)
    df["t_s"] = df["frame_number"] / fps

    # --- Rename columns to engine format ---
    df = df.rename(columns={
        "court_x_normalized": "x",
        "court_y_normalized": "y",
    })

    # --- Compute velocities, accelerations, speed ---
    df["vx"] = df["x"].diff().fillna(0) * fps
    df["vy"] = df["y"].diff().fillna(0) * fps
    df["ax"] = df["vx"].diff().fillna(0) * fps
    df["ay"] = df["vy"].diff().fillna(0) * fps
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
        )
        for _, row in df.iterrows()
    ]

    print(f"[ok] Loaded {len(frames)} frames from {path} (player_name='{player_name or 'ALL'}')")
    return frames
