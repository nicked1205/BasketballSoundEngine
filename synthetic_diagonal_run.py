import math
from config import Court, Synthesis
from synth_data import Frame, write_csv

def generate_diagonal_run():
    """
    10-second diagonal run (−1, −1) → (+1, +1)
    Generates only kinematic data: t, x, y, vx, vy, ax, ay, speed.
    """
    court = Court()
    fps = 30
    duration_s = 10.0
    n = int(duration_s * fps)

    x0, y0 = -1.0, -1.0
    x1, y1 = 1.0, 1.0

    frames = []
    for i in range(n):
        t = i / fps
        u = i / (n - 1)

        # Speed ramp (walk → jog → sprint)
        if t < 3:
            speed = 1.0 + (2.0 - 1.0) * (t / 3.0)
        elif t < 6:
            speed = 2.0 + (4.0 - 2.0) * ((t - 3.0) / 3.0)
        else:
            speed = 4.0 + (6.0 - 4.0) * ((t - 6.0) / 4.0)

        # Direction & velocity
        dirx, diry = (x1 - x0), (y1 - y0)
        norm = math.hypot(dirx, diry)
        dirx /= norm
        diry /= norm
        vx, vy = speed * dirx, speed * diry

        if i == 0:
            ax = ay = 0.0
        else:
            ax = (vx - frames[-1].vx) * fps
            ay = (vy - frames[-1].vy) * fps

        frames.append(Frame(
            frame=i,
            t_s=t,
            x=x0 + (x1 - x0) * u,
            y=y0 + (y1 - y0) * u,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            speed_mps=speed,
            intensity_v=0.0,     # placeholders
            distance_m=0.0,
            distance_v=0.0,
            total_v=0.0,
            pan=0.0,
        ))

    return frames

if __name__ == "__main__":
    frames = generate_diagonal_run()
    write_csv(frames, "diagonal_run_10s.csv")
    print("[ok] Wrote diagonal_run_10s.csv")
