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
    accel_threshold=4.5,        # m/s² minimum decel to trigger
    turn_threshold_deg=45.0,    # must turn at least this angle
    min_speed_for_event=1.5,    # ignore slow walks
    cooldown=0.5,               # seconds between squeaks
):
    squeaks = []
    last_vx, last_vy = frames[0].vx, frames[0].vy
    last_speed = frames[0].speed_mps
    last_t = frames[0].t_s
    last_squeak_time = -999.0

    for fr in frames[1:]:
        dt = max(fr.t_s - last_t, 1e-3)
        dv = last_speed - fr.speed_mps
        decel = dv / dt if dv > 0 else 0.0

        # Direction change
        dot = last_vx * fr.vx + last_vy * fr.vy
        mag1 = math.hypot(last_vx, last_vy)
        mag2 = math.hypot(fr.vx, fr.vy)
        angle = 0.0
        if mag1 > 1e-3 and mag2 > 1e-3:
            cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle = math.degrees(math.acos(cos_angle))

        # Conditions for squeak
        is_turn = angle > turn_threshold_deg
        is_brake = decel > accel_threshold
        is_moving = fr.speed_mps > min_speed_for_event
        enough_time = (fr.t_s - last_squeak_time) > cooldown

        if (is_turn or is_brake) and is_moving and enough_time:
            gain, pan, dist = compute_audio_params(fr)
            squeaks.append(Footstep(
                t_ms=int(fr.t_s * 1000.0),
                gain=gain * 0.9,
                pan=pan,
                side="L"
            ))
            last_squeak_time = fr.t_s

        last_vx, last_vy, last_speed, last_t = fr.vx, fr.vy, fr.speed_mps, fr.t_s

    return squeaks

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

    # --- Stereo pan (narrower spread, scaled to ±0.4) ---
    pan = max(-1.0, min(1.0, fr.x * 0.4))

    print(f"[debug] t={fr.t_s:.2f}s y={fr.y:.2f} dist={dist:.2f} dvol={dvol:.2f} inten={inten:.2f} total_v={total_v:.2f} pan={pan:.2f}")

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

def generate_ambient_noise(duration_ms: int, sr: int) -> AudioSegment:
    import numpy as np
    n = int(sr * duration_ms / 1000)
    noise = (np.random.randn(n) * 0.03).astype("float32")

    # Band-limit the noise (simulate low-mid room hum)
    from scipy.signal import butter, lfilter
    b, a = butter(4, [100/(sr/2), 3000/(sr/2)], btype="band")
    filtered = lfilter(b, a, noise)

    raw = (filtered * 32767).astype("int16").tobytes()
    amb = AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sr,
        channels=2
    )
    return amb - 25  # make it subtle

# Render footsteps with alternating left/right samples

def render_footsteps(frames, foot_path: Optional[str], duration_ms: int, cfg: Audio) -> AudioSegment:
    assets = Assets(cfg.sample_rate)
    # load left/right variants (mono or stereo)
    assets.load(foot_path, squeak_path="./assets/squeak.wav")

    mixer = Mixer(cfg)
    mix = mixer.make_timeline(duration_ms)

    for ev in footsteps_from_frames(frames):
        # Find the frame closest in time to this event
        nearest = min(frames, key=lambda f: abs(f.t_s * 1000 - ev.t_ms))

        # Compute gain & pan from physical data
        total_v, pan, dist = compute_audio_params(nearest)

        base = assets.foot_L if ev.side == "L" else assets.foot_R

        # --- randomized footstep processing ---
        seg, offset = randomize_footstep(base, ev.side)

        # --- add early reflections ---
        seg = add_reflections(seg, dist)

        # --- slight stereo bias depending on foot ---
        pan_offset = -0.01 if ev.side == "L" else 0.01
        seg = apply_pan(seg, max(-1.0, min(1.0, pan + pan_offset)))

        # --- apply distance × intensity scaling ---
        seg = seg.apply_gain(lin_to_db(total_v))
        
        mix = mix.overlay(seg, position=ev.t_ms)
    
    squeaks = squeaks_from_frames(frames)
    print(f"[debug] {len(squeaks)} squeaks detected")

    if assets.squeak is not None and len(squeaks) > 0:
        for sq in squeaks:
            seg = randomize_squeak(assets.squeak)
            seg = assets.squeak.apply_gain(lin_to_db(sq.gain))
            seg = apply_pan(seg, sq.pan) - 20
            prob = max(0.2, 1.0 - dist / 20.0)  # drop-off with distance
            if random.random() < prob:
                seg = add_random_echoes(seg)
            mix = mix.overlay(seg, position=sq.t_ms)

    # --- add subtle ambient noise ---
    ambient = AudioSegment.from_file("ambient.wav")
    ambient = ambient - 55
    mix = mix.overlay(ambient)

    mix = mixer.limiter(mix)
    return mix