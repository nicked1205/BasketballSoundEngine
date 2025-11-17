import math
import random
from typing import List, Optional
from dataclasses import dataclass
from pydub import AudioSegment
import pandas as pd
import numpy as np
from itertools import groupby

from config import Court, Camera, Volumes, Audio


MASTER_VOLUME = 10

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

def seg_to_np(seg: AudioSegment) -> np.ndarray:
    samples = np.array(seg.get_array_of_samples()).astype(np.int16)
    samples = samples.reshape((-1, seg.channels))
    return samples

def np_to_seg(arr: np.ndarray, sample_rate: int) -> AudioSegment:
    arr = np.clip(arr, -32768, 32767).astype(np.int16)
    raw = arr.tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sample_rate,
        channels=2
    )

# ---------- Engine ----------

@dataclass
class Footstep:
    t_ms: int
    gain: float  # overall volume (0..1)
    pan: float   # -1..1
@dataclass
class Squeak:
    t_ms: int
    gain: float
    pan: float

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
        self.foot_L = None
        self.foot_R = None
        self.squeak = None

    def load(self, foot_path: Optional[str] = None, squeak_path: Optional[str] = None):
        from pydub import AudioSegment

        # Footsteps (as before)
        if foot_path is None:
            self.foot_L = self._synth_click()
            self.foot_R = self._synth_click().invert_phase()
        else:
            import os
            base = os.path.splitext(foot_path)[0]
            left_file = base + "_L.wav"
            right_file = base + "_R.wav"
            if not os.path.exists(left_file): left_file = foot_path
            if not os.path.exists(right_file): right_file = foot_path
            self.foot_L = AudioSegment.from_file(left_file).set_frame_rate(self.sr).set_channels(2)
            self.foot_R = AudioSegment.from_file(right_file).set_frame_rate(self.sr).set_channels(2)

        if squeak_path is None:
            print("[warn] No squeak file specified — skipping squeak events.")
        else:
            self.squeak = AudioSegment.from_file(squeak_path).set_frame_rate(self.sr).set_channels(2)
            # Optionally shorten / fade
            self.squeak = self.squeak[:250].fade_out(100)

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
        return 9999.0  # no steps if barely moving
    if speed_mps < 1.0:
        return (800.0 - 400.0 * (speed_mps - 0.3) / 0.7)
    if speed_mps < 4.0:
        return (400.0 - 100.0 * (speed_mps - 1.0) / 3.0)
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
                gain=0.0,
                pan=0.0,
                side=side
            ))
            last_step_time = t_ms
    return events

def randomize_footstep(seg: AudioSegment, side: str) -> AudioSegment:
    """Randomize pitch, duration, brightness, and level of a footstep sound."""
    import random

    # Slight pitch variation (±3%)
    semitones = random.uniform(-1, 1)
    new_rate = int(seg.frame_rate * (2 ** (semitones / 12.0)))
    seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(seg.frame_rate)

    # Random length trimming (simulate varying pressure/time)
    length_factor = random.uniform(0.9, 1.0)
    seg = seg[:int(len(seg) * length_factor)].fade_out(int(50 * length_factor))

    # Brightness variation (EQ filtering)
    if random.random() < 0.9:
        cutoff = random.uniform(1800, 3500)
        seg = seg.high_pass_filter(cutoff)
    else:
        cutoff = random.uniform(5000, 8000)
        seg = seg.high_pass_filter(cutoff)

    # Micro timing offset (simulate imperfect stride timing)
    offset = random.uniform(-15, 15)  # ±15 ms offset
    seg = seg.fade_in(10).fade_out(80)

    return seg, offset

# Generate squeak events from frames based on deceleration and turning

def squeaks_from_frames(
    frames,
    accel_threshold=5.0,
    turn_threshold_deg=60.0,
    min_speed_for_event=6.0,
    cooldown=0.5,
):
    """
    Multi-player aware squeak detector.
    Keeps vectorized math per player for speed.
    """
    if len(frames) < 2:
        return []

    squeaks_all = []

    # --- group frames by player ---
    frames.sort(key=lambda f: getattr(f, "player_id", 0))
    for pid, group in groupby(frames, key=lambda f: getattr(f, "player_id", 0)):
        f_list = list(group)
        if len(f_list) < 2:
            continue

        # --- Extract arrays for this player ---
        vx = np.array([f.vx for f in f_list], dtype=np.float32)
        vy = np.array([f.vy for f in f_list], dtype=np.float32)
        speed = np.array([f.speed_mps for f in f_list], dtype=np.float32)
        t = np.array([f.t_s for f in f_list], dtype=np.float32)

        # --- Compute deceleration and turn angles ---
        dv = np.diff(speed)
        dt = np.diff(t)
        decel = np.maximum(0, -dv / np.maximum(dt, 1e-3))

        dot = vx[:-1]*vx[1:] + vy[:-1]*vy[1:]
        mag1 = np.hypot(vx[:-1], vy[:-1])
        mag2 = np.hypot(vx[1:], vy[1:])
        cos_angle = np.clip(dot / np.maximum(mag1*mag2, 1e-6), -1, 1)
        angles = np.degrees(np.arccos(cos_angle))

        # --- Boolean masks ---
        is_turn = angles > turn_threshold_deg
        is_brake = decel > accel_threshold
        is_moving = speed[1:] > min_speed_for_event
        candidates = np.where((is_turn | is_brake) & is_moving)[0] + 1

        # --- Apply cooldown ---
        squeak_indices = []
        last_time = -999.0
        for idx in candidates:
            if (t[idx] - last_time) > cooldown:
                squeak_indices.append(idx)
                last_time = t[idx]

        # --- Build results ---
        for idx in squeak_indices:
            fr = f_list[idx]
            gain, pan, dist = compute_audio_params(fr)
            squeaks_all.append(Footstep(
                t_ms=int(fr.t_s * 1000),
                gain=gain * 0.9,
                pan=pan,
                side="L"  # reused Footstep class for squeaks
            ))

    return squeaks_all


def randomize_squeak(seg: AudioSegment) -> AudioSegment:
    # pitch jitter
    semitones = random.uniform(-2.0, 2.0)
    if semitones > 0:
        seg = seg.high_pass_filter(1000)
    else:
        seg = seg.low_pass_filter(4000)
    new_rate = int(seg.frame_rate * (2 ** (semitones / 12.0)))
    seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(seg.frame_rate)

    # dynamic length (truncate tail randomly 80–100%)
    length_factor = random.uniform(0.8, 1.0)
    seg = seg[:int(len(seg) * length_factor)].fade_out(int(40 * length_factor))

    # gain variation (±2 dB)
    seg = seg.apply_gain(random.uniform(-2.0, 2.0))
    return seg

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

    # --- Stereo pan (narrower spread, scaled to ±0.7) ---
    pan = max(-1.0, min(1.0, fr.x * 0.7))
    
    return total_v, pan, dist

# Adds random echoes to simulate complex reflections in an indoor court.

def add_random_echoes(seg: AudioSegment, max_reflections: int = 3) -> AudioSegment:
    import random

    n_reflections = random.randint(1, max_reflections)
    wet = seg
    for _ in range(n_reflections):
        delay = random.uniform(20, 80)        # ms
        gain = random.uniform(-12, -4)        # dB quieter
        pan = random.uniform(-0.3, 0.3)       # small stereo offset

        echo = apply_pan(seg - abs(gain), pan)
        wet = wet.overlay(echo, position=delay)

    # optional slight tail fade
    return wet.fade_out(50)


# Adds subtle early reflections to simulate court acoustics.

def add_reflections(seg: AudioSegment, dist: float) -> AudioSegment:
    # Reflection timing and strength depend on distance
    floor_delay = 25  # ms
    wall_delay = 80   # ms
    base_gain = -9 if dist < 10 else -12  # closer sounds = stronger reflections

    # Create early reflections
    floor_echo = seg - (base_gain + random.uniform(0, 2))
    wall_echo = seg - (base_gain + 6 + random.uniform(0, 3))

    seg = seg.overlay(floor_echo, position=floor_delay)
    seg = seg.overlay(wall_echo, position=wall_delay)

    return seg

def extend_ambient(amb: AudioSegment, target_ms: int, crossfade_ms: int = 4000) -> AudioSegment:
    """
    Extends a short ambient AudioSegment to the desired duration using
    random slicing + crossfades to avoid repetition artifacts.
    """
    result = AudioSegment.silent(duration=0, frame_rate=amb.frame_rate)
    seg_len = len(amb)

    while len(result) < target_ms:
        # Pick a random start (avoid very end)
        start = random.randint(0, max(0, seg_len - 30000))  # random 0–30 s offset
        chunk = amb[start:start + random.randint(60000, 90000)]  # 1–1.5 min random section
        if len(result) == 0:
            result = chunk
        else:
            result = result.append(chunk, crossfade=crossfade_ms)

    # Trim to exact length
    result = result[:target_ms]

    # Optional: gentle fade in/out to hide start/stop points
    result = result.fade_in(5000).fade_out(5000)
    return result

# Render footsteps with alternating left/right samples

def render_footsteps(frames, foot_path: Optional[str], duration_ms: int, cfg: Audio):
    """
    Multi-player footsteps and squeaks rendered separately, 
    but still mixed with global NumPy accumulation for speed.
    """
    assets = Assets(cfg.sample_rate)
    assets.load(foot_path, squeak_path="./assets/squeak.wav")
    mixer = Mixer(cfg)

    mix_len = int(cfg.sample_rate * duration_ms / 1000)
    mix_foot_total = np.zeros((mix_len, 2), dtype=np.int32)
    mix_squeak_total = np.zeros((mix_len, 2), dtype=np.int32)

    all_foot_data = []
    all_squeak_data = []

    # --- Group frames per player ---
    frames.sort(key=lambda f: getattr(f, "player_id", 0))
    player_groups = {pid: list(g) for pid, g in groupby(frames, key=lambda f: getattr(f, "player_id", 0))}

    for pid, f_list in player_groups.items():
        print(f"[player {pid}] rendering {len(f_list)} frames")

        frame_times = [f.t_s * 1000 for f in f_list]

        def nearest_frame(t):
            import bisect
            i = bisect.bisect_left(frame_times, t)
            if i <= 0: return f_list[0]
            if i >= len(f_list): return f_list[-1]
            return f_list[i]

        # --- Footsteps for this player ---
        foot_events = footsteps_from_frames(f_list)
        mix_foot = np.zeros((mix_len, 2), dtype=np.int32)

        for ev in foot_events:
            fr = nearest_frame(ev.t_ms)
            total_v, pan, dist = compute_audio_params(fr)
            base = assets.foot_L if ev.side == "L" else assets.foot_R
            seg, offset = randomize_footstep(base, ev.side)
            seg = add_reflections(seg, dist)
            seg = apply_pan(seg, max(-1.0, min(1.0, pan + (-0.01 if ev.side == "L" else 0.01))))
            seg = seg.apply_gain(lin_to_db(total_v))
            seg = seg - 10 + MASTER_VOLUME
            s = seg_to_np(seg)
            start = int(ev.t_ms * cfg.sample_rate / 1000)
            if start >= mix_len:
                continue  # skip sounds beyond duration
            end = start + s.shape[0]
            if end > mix_len:
                s = s[:mix_len - start]
                end = mix_len
            if end > start:
                mix_foot[start:end] += s
            all_foot_data.append({
                "frame": ev.t_ms,
                "vol": total_v,
                "bal": pan,
                "length": len(seg)
            })
            print(f"  [footstep @ {ev.t_ms} ms] gain={total_v:.2f} pan={pan:.2f}")

        # --- Squeaks for this player ---
        squeaks = squeaks_from_frames(f_list)
        mix_squeak = np.zeros((mix_len, 2), dtype=np.int32)

        if assets.squeak is not None:
            for sq in squeaks:
                seg = randomize_squeak(assets.squeak)
                seg = apply_pan(seg, sq.pan)
                seg = seg.apply_gain(lin_to_db(sq.gain))
                seg = seg - 10 + MASTER_VOLUME
                if random.random() < max(0.2, 1.0 - sq.gain / 2.0):
                    seg = add_random_echoes(seg)
                s = seg_to_np(seg)
                start = int(sq.t_ms * cfg.sample_rate / 1000)
                if start >= mix_len:
                    continue
                end = start + s.shape[0]
                if end > mix_len:
                    s = s[:mix_len - start]
                    end = mix_len
                if end > start:
                    mix_squeak[start:end] += s
                all_squeak_data.append({
                    "frame": sq.t_ms,
                    "vol": sq.gain,
                    "bal": sq.pan,
                    "length": len(seg)
                })
                print(f"  [squeak @ {sq.t_ms} ms] gain={sq.gain:.2f} pan={sq.pan:.2f}")

        # --- Accumulate into global mix ---
        mix_foot_total += mix_foot
        mix_squeak_total += mix_squeak

    # --- Convert to segments and limit ---
    mix_foot_seg = mixer.limiter(np_to_seg(mix_foot_total, cfg.sample_rate))
    mix_squeak_seg = mixer.limiter(np_to_seg(mix_squeak_total, cfg.sample_rate))
    combined = mix_foot_seg.overlay(mix_squeak_seg)

    pd.DataFrame(all_foot_data).to_csv("footsteps_data.csv", index=False)
    pd.DataFrame(all_squeak_data).to_csv("squeaks_data.csv", index=False)

    # --- Optional ambient layer ---
    try:
        ambient = AudioSegment.from_file("./assets/ambient.wav").set_frame_rate(cfg.sample_rate).set_channels(2)
        ambient_full = extend_ambient(ambient, duration_ms)
        ambient_full = ambient_full + 10 + MASTER_VOLUME

        # Export full ambient track separately
        ambient_full.export("ambient_full.wav", format="wav")
        print("[ok] Exported full ambient track: ambient_full.wav")

        # Add it to the combined mix quietly
        combined = combined.overlay(ambient_full)

    except Exception as e:
        print(f"[warn] Could not load ambient.wav: {e}")

    return mix_foot_seg, mix_squeak_seg, combined

def render_bounces(bounces, bounce_path: str, duration_ms: int, cfg: Audio):
    from pydub import AudioSegment

    # --- Load and preprocess base sample ---
    base = AudioSegment.from_file(bounce_path).set_frame_rate(cfg.sample_rate).set_channels(2)
    base_np = seg_to_np(base)
    n_base = base_np.shape[0]
    base_len_ms = len(base)

    # --- Prepare master mix array ---
    mix_len = int(cfg.sample_rate * duration_ms / 1000)
    mix = np.zeros((mix_len, 2), dtype=np.int32)

    # --- Precompute attenuation and panning ---
    court = Court()
    cam = Camera()
    x = np.array([b.x for b in bounces], dtype=np.float32)
    y = np.array([b.y for b in bounces], dtype=np.float32)
    t_s = np.array([b.t_s for b in bounces], dtype=np.float32)

    # Distance attenuation
    x_m = x * (court.width_m / 2)
    y_m = y * (court.depth_m / 2)
    dx, dy, dz = x_m - cam.x_m, y_m - cam.y_m, -cam.z_m
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    d_ref = 8.0
    dvol = np.maximum(0.25, (d_ref / np.maximum(dist, d_ref)) ** 1.6)
    pan = np.clip(x * 0.7, -1.0, 1.0)

    # --- Apply each bounce ---
    bounce_data = []
    for i, b in enumerate(bounces):
        gain = dvol[i]

        # small random pitch
        rate_factor = random.uniform(0.9, 1.1)
        seg = base._spawn(base.raw_data, overrides={"frame_rate": int(base.frame_rate * rate_factor)})
        seg = seg.set_frame_rate(cfg.sample_rate)

        L, R = constant_power_pan(pan[i])
        gL, gR = lin_to_db(gain * L), lin_to_db(gain * R)
        seg = base.apply_gain_stereo(gL, gR)

        seg = seg + MASTER_VOLUME

        # simple echo: mix a quiet delayed copy
        if random.random() < 0.6:
            delay = random.randint(70, 150)
            echo = seg - random.uniform(15, 18)
            seg = seg.overlay(echo, position=delay)

        seg_np = seg_to_np(seg)
        start = int(t_s[i] * cfg.sample_rate)
        end = start + seg_np.shape[0]
        if start >= mix_len:
            continue
        if end > mix_len:
            seg_np = seg_np[:mix_len - start]
            end = mix_len
        mix[start:end] += seg_np

        bounce_data.append({
            "frame": b.frame,
            "vol": float(gain),
            "bal": float(pan[i]),
            "length": base_len_ms
        })

    # --- Save metadata CSV ---
    pd.DataFrame(bounce_data).to_csv("bounces_data.csv", index=False)

    # --- Convert back to AudioSegment and limit ---
    mix_bounce = np_to_seg(mix, cfg.sample_rate)
    mixer = Mixer(cfg)
    mix_bounce = mixer.limiter(mix_bounce)
    return mix_bounce

def render_claps(score_events, clap_path: str, duration_ms: int, cfg: Audio, fps: float):
    """
    For each scoring event, extract a random 4–6s segment of the clap sound
    with a fade-out and overlay it at the event time.
    """
    from pydub import AudioSegment

    clap_full = AudioSegment.from_file(clap_path).set_frame_rate(cfg.sample_rate).set_channels(2)
    clap_len = len(clap_full)

    mix_len = int(cfg.sample_rate * duration_ms / 1000)
    mix = np.zeros((mix_len, 2), dtype=np.int32)

    clap_data = []

    for ev in score_events:
        t_s = ev["t_s"]  # event time in seconds
        points = ev.get("points", 2)

        # choose random 4–6 s snippet
        seg_dur = random.randint(3000, 5000)
        if seg_dur >= clap_len:
            start_ms = 0
        else:
            start_ms = random.randint(0, clap_len - seg_dur)
        seg = clap_full[start_ms:start_ms + seg_dur].fade_out(1000)
        seg = seg.fade_in(500)

        # slight random gain and stereo variation
        seg = seg.apply_gain(random.uniform(-2, 2))
        seg = apply_pan(seg, 0)
        seg = seg + 5 + MASTER_VOLUME

        if random.random() < 0.7:
            seg = add_random_echoes(seg, max_reflections=2)

        # overlay into the mix
        s = seg_to_np(seg)
        start = int(t_s * cfg.sample_rate)
        end = start + s.shape[0]
        if start >= mix_len:
            continue
        if end > mix_len:
            s = s[:mix_len - start]
            end = mix_len
        mix[start:end] += s

        clap_data.append({
            "frame": int(t_s * fps),
            "vol": float(1.0) ,
            "pan": float(0.0),
            "length": seg_dur
        })

    # save metadata
    pd.DataFrame(clap_data).to_csv("claps_data.csv", index=False)

    mix_clap = np_to_seg(mix, cfg.sample_rate)
    mixer = Mixer(cfg)
    mix_clap = mixer.limiter(mix_clap)
    return mix_clap
