import math
import random
from typing import List, Optional
from dataclasses import dataclass
from pydub import AudioSegment

from config import Court, Camera, Volumes, Audio

# ---------- DSP helpers ----------

def lin_to_db(g: float) -> float:
    return 20.0 * math.log10(max(g, 1e-6))


def constant_power_pan(pan: float) -> (float, float):
    pan = max(-1.0, min(1.0, pan))
    u = 0.5 * (pan + 1.0)
    theta = u * (math.pi/2)
    L = math.cos(theta)
    R = math.sin(theta)
    return L, R


def apply_pan(seg: AudioSegment, pan: float) -> AudioSegment:
    L, R = constant_power_pan(pan)
    return seg.apply_gain_stereo(lin_to_db(L), lin_to_db(R))

# ---------- Engine ----------

@dataclass
class Footstep:
    t_ms: int
    gain: float  # overall volume (0..1)
    pan: float   # -1..1

class Mixer:
    def __init__(self, cfg: Audio):
        self.cfg = cfg

    def make_timeline(self, duration_ms: int) -> AudioSegment:
        base = AudioSegment.silent(duration=duration_ms, frame_rate=self.cfg.sample_rate).set_channels(2)
        return base - self.cfg.headroom_db

    def limiter(self, seg: AudioSegment) -> AudioSegment:
        peak = seg.max_dBFS
        if peak <= self.cfg.limiter_threshold_db:
            return seg
        delta = peak - self.cfg.limiter_threshold_db
        gain_red = delta - delta/self.cfg.limiter_ratio
        return seg.apply_gain(-gain_red)

class Assets:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.foot = None

    def load(self, path: Optional[str]):
        from pydub import AudioSegment
        if path is None:
            # synthesize a thump if no asset provided
            self.foot = self._synth_click()
        else:
            self.foot = AudioSegment.from_file(path).set_frame_rate(self.sr).set_channels(2)
        # shape it: short decay
        self.foot = self.foot[:140].fade_out(90)

    def _synth_click(self) -> AudioSegment:
        import numpy as np

        dur_ms = 120
        n = int(self.sr * dur_ms / 1000)
        # random noise burst
        noise = (np.random.randn(n) * 0.15).astype("float32")

        # exponential decay envelope
        env = np.exp(-np.linspace(0, 8, n)).astype("float32")
        wave = (noise * env).clip(-1, 1)

        # convert to 16-bit PCM
        raw = (wave * 32767).astype("int16").tobytes()

        mono = AudioSegment(
            data=raw,  # use 'data' instead of 'raw_data'
            sample_width=2,
            frame_rate=self.sr,
            channels=1,
        )
        return mono.set_channels(2)

# Map speed (m/s) → steps per minute (very rough fit)
# 0 m/s → 0 spm, 1.5 → ~110 spm, 3.0 → ~150 spm, 6.0 → ~190 spm

def cadence_spm(speed_mps: float) -> float:
    if speed_mps <= 0.2:
        return 0.0
    return max(60.0, min(200.0, 80.0 + 20.0*speed_mps + 10.0*max(0.0, speed_mps-1.5)))

# Generate alternating L/R footsteps from per-frame telemetry

@dataclass
class Footstep:
    t_ms: int
    gain: float
    pan: float
    side: str  # "L" or "R"


def step_interval_ms(speed_mps: float) -> float:
    # Piecewise-linear model for human gait
    if speed_mps < 0.2:
        return 9999.0 / 1.5  # no steps if barely moving
    if speed_mps < 1.0:
        return (800.0 - 400.0 * (speed_mps - 0.3) / 0.7) / 1.5
    if speed_mps < 4.0:
        return (400.0 - 100.0 * (speed_mps - 1.0) / 3.0) / 1.5
    # sprinting region
    return max(180.0, 300.0 - 20.0 * (speed_mps - 4.0))


def footsteps_from_frames(frames, lead_ms: int = 0) -> List[Footstep]:
    events: List[Footstep] = []
    last_step_time = 0.0
    side = "L"

    for fr in frames:
        t_ms = fr.t_s * 1000.0 + lead_ms
        dt = t_ms - last_step_time
        si = step_interval_ms(fr.speed_mps)
        if dt >= si:
            # Alternate left/right
            side = "R" if side == "L" else "L"
            events.append(Footstep(
                t_ms=int(t_ms),
                gain=fr.total_v,
                pan=fr.pan,
                side=side
            ))
            last_step_time = t_ms
    return events

# Compute distance attenuation, intensity, and stereo pan from a frame containing (x, y, speed_mps).

def compute_audio_params(fr):
    court = Court()
    cam = Camera()
    vols = Volumes()

    # --- 3D distance from camera ---
    x_m = fr.x * (court.width_m / 2)
    y_m = fr.y * (court.depth_m / 2)
    dx = x_m - cam.x_m
    dy = y_m - cam.y_m
    dz = -cam.z_m
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    # --- Distance volume (attenuation) ---
    def distance_volume(d, d_ref=8.0, min_v=0.25, power=1.6):
        rel = (d_ref / max(d, d_ref)) ** power
        return max(min_v, rel)
    dvol = dvol = distance_volume(dist, d_ref=8.0, min_v=0.25)

    # --- Intensity volume from speed ---
    def intensity_from_speed(speed):
        v_max = 6.0
        base = 0.1
        p = 1.9
        norm = max(0.0, min(1.0, speed / v_max))
        return base + (1.0 - base) * (norm ** p)

    inten = intensity_from_speed(fr.speed_mps)

    # --- Combined volume (prioritize distance) ---
    w_d, w_i = 0.95, 0.05
    total_v = (w_d * dvol) + (w_i * (inten * dvol))

    # --- Stereo pan (narrower spread, scaled to ±0.4) ---
    pan = max(-1.0, min(1.0, fr.x * 0.4))

    print(f"[debug] t={fr.t_s:.2f}s y={fr.y:.2f} dist={dist:.2f} dvol={dvol:.2f} inten={inten:.2f} total_v={total_v:.2f} pan={pan:.2f}")

    return total_v, pan


# Render footsteps with alternating left/right samples

def render_footsteps(frames, foot_path: Optional[str], duration_ms: int, cfg: Audio) -> AudioSegment:
    assets = Assets(cfg.sample_rate)
    # load left/right variants (mono or stereo)
    if foot_path:
        import os
        base = os.path.splitext(foot_path)[0]
        left_file = base + "_L.wav"
        right_file = base + "_R.wav"
        if not os.path.exists(left_file):
            left_file = foot_path
        if not os.path.exists(right_file):
            right_file = foot_path
        assets.foot_L = AudioSegment.from_file(left_file).set_frame_rate(cfg.sample_rate).set_channels(2)
        assets.foot_R = AudioSegment.from_file(right_file).set_frame_rate(cfg.sample_rate).set_channels(2)
    else:
        # use synthetic thumps if none provided
        assets.foot_L = assets._synth_click()
        assets.foot_R = assets._synth_click().invert_phase()  # subtle difference

    mixer = Mixer(cfg)
    mix = mixer.make_timeline(duration_ms)

    for ev in footsteps_from_frames(frames):
        # Find the frame closest in time to this event
        nearest = min(frames, key=lambda f: abs(f.t_s * 1000 - ev.t_ms))

        # Compute gain & pan from physical data
        total_v, pan = compute_audio_params(nearest)

        base = assets.foot_L if ev.side == "L" else assets.foot_R

        # --- random pitch shift (±2%) ---
        semitones = random.uniform(-0.35, 0.35)
        if abs(semitones) > 1e-4:
            new_rate = int(base.frame_rate * (2 ** (semitones / 12.0)))
            seg = base._spawn(base.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(base.frame_rate)
        else:
            seg = base

        # --- random volume jitter (±1 dB) ---
        seg = seg.apply_gain(random.uniform(-1.0, 1.0))

        # --- slight stereo bias depending on foot ---
        pan_offset = -0.01 if ev.side == "L" else 0.01
        seg = apply_pan(seg, max(-1.0, min(1.0, pan + pan_offset)))

        # --- apply distance × intensity scaling ---
        seg = seg.apply_gain(lin_to_db(total_v))

        mix = mix.overlay(seg, position=ev.t_ms)


    mix = mixer.limiter(mix)
    return mix