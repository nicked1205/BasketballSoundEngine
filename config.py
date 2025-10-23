from dataclasses import dataclass


@dataclass
class Court:
    width_m: float = 28.0
    depth_m: float = 15.0
    # normalized coords in [-1, 1]; map to meters via half-width/half-depth


@dataclass
class Camera:
    # Mounted 8 m behind the bottom wall and 8 m up
    x_m: float = 0.0
    y_m: float = - 8.0
    z_m: float = 8.0


@dataclass
class Synthesis:
    fps: int = 30
    n_frames: int = 1800 # ~60 s
    seed: int = 42


@dataclass
class Volumes:
    # Distance rolloff: overall_volume = distance_volume * intensity_volume
    # distance_volume = 1 / (1 + k * distance_m), clamped to [min_v, 1]
    distance_k: float = 0.12
    distance_min_v: float = 0.08


    # Intensity volume: piecewise-linear over speed (m/s).
    # Anchors: (0 → 0.05), walking 1.5→0.10, jogging 4.0→0.50, sprint 7.5→1.00
    anchors = [(0.0, 0.05), (1.5, 0.10), (4.0, 0.50), (7.5, 1.00)]


@dataclass
class Audio:
    sample_rate: int = 48000
    headroom_db: float = 6.0
    limiter_threshold_db: float = -1.0
    limiter_ratio: float = 6.0
    enable_doppler: bool = False # off for footsteps by default