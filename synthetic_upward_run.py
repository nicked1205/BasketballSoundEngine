import math
from config import Court
from synth_data import Frame, write_csv

def generate_upward_run():
    """
    10-second upward run: bottom (y=-1) -> top (y=+1), x fixed at 0.
    Generates physical motion only (no audio params).
    """
    court = Court()
    fps = 30
    duration_s = 10.0
    n = int(duration_s * fps)

    x = 0.0         # constant (middle of court)
    y0, y1 = -1.0, 1.0

    frames = []
    for i in range(n):
        t = i / fps
        u = i / (n - 1)

        # --- nonlinear acceleration: smooth ease-in curve ---
        # starts slow, speeds up fast (like a real sprint)
        ease = u ** 2.2  # quadratic-like ease-in

        # --- speed profile (matches walk→jog→sprint) ---
        # walking: ~1.0 m/s → jogging: ~4.0 m/s → sprinting: ~6.0 m/s
        if t < 3:
            speed = 1.0 + (2.0 - 1.0) * (t / 3.0)
        elif t < 6:
            speed = 2.0 + (4.0 - 2.0) * ((t - 3.0) / 3.0)
        else:
            speed = 4.0 + (6.0 - 4.0) * ((t - 6.0) / 4.0)

        # --- position (normalized to -1..+1) ---
        y = y0 + (y1 - y0) * ease

        # --- velocity (along y only) ---
        vy = (y1 - y0) * (2.2 * u ** 1.2) * (1 / duration_s)
        vx = 0.0

        # --- acceleration (finite diff) ---
        if i == 0:
            ay = ax = 0.0
        else:
            ay = (vy - frames[-1].vy) * fps
            ax = 0.0

        frames.append(Frame(
            frame=i,
            t_s=t,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            speed_mps=speed,
            intensity_v=0.0,  # placeholder (computed later)
            distance_m=0.0,
            distance_v=0.0,
            total_v=0.0,
            pan=0.0,
        ))

    return frames


if __name__ == "__main__":
    frames = generate_upward_run()
    write_csv(frames, "upward_run_10s.csv")
    print("[ok] Wrote upward_run_10s.csv")
