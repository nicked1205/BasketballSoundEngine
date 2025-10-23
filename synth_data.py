import math, random
from dataclasses import dataclass
from typing import List, Tuple
import csv

from config import Court, Camera, Synthesis, Volumes

@dataclass
class Vec2:
    x: float
    y: float

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
    intensity_v: float
    distance_m: float
    distance_v: float
    total_v: float
    pan: float  # -1..1 (constant-power pan to be applied later)

# ---------- Mapping helpers ----------

def norm_to_meters(xn: float, yn: float, court: Court) -> Tuple[float, float]:
    x_m = xn * (court.width_m / 2.0)
    y_m = yn * (court.depth_m / 2.0)
    return x_m, y_m


def meters_to_norm(xm: float, ym: float, court: Court) -> Tuple[float, float]:
    return xm / (court.width_m / 2.0), ym / (court.depth_m / 2.0)


def distance_from_camera(xn: float, yn: float, court: Court, cam: Camera) -> float:
    x_m, y_m = norm_to_meters(xn, yn, court)
    dx = x_m - cam.x_m
    dy = y_m - cam.y_m
    dz = 0.0 - cam.z_m
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def intensity_from_speed(speed_mps: float, anchors) -> float:
    # piecewise-linear interpolation between anchor points
    if speed_mps <= anchors[0][0]:
        return anchors[0][1]
    for (x0,y0), (x1,y1) in zip(anchors, anchors[1:]):
        if speed_mps <= x1:
            t = (speed_mps - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return anchors[-1][1]


def distance_volume(d_m: float, k: float, min_v: float) -> float:
    return max(min_v, 1.0/(1.0 + k * d_m))


def pan_from_x(xn: float) -> float:
    # Map normalized x in [-1,1] directly to pan in [-1,1]
    return max(-1.0, min(1.0, xn))

# ---------- Trajectory synthesis ----------

def ease_in_out(t: float) -> float:
    # cosine ease
    return 0.5 - 0.5*math.cos(math.pi*t)


def make_motion_profile(fps: int) -> List[Tuple[int, float]]:
    """
    Returns a list of (duration_frames, target_speed_mps) phases.
    The player: walk → jog → turn → sprint → slow → pause → jog, etc.
    Durations sum to ~1800, we'll trim/pad.
    """
    phases = [
        (180, 1.2),  # walk ~6s
        (240, 2.5),  # easy jog ~8s
        (120, 0.8),  # slow to turn ~4s
        (300, 6.0),  # sprint line ~10s
        (120, 0.5),  # recover ~4s
        (240, 3.5),  # tempo run ~8s
        (180, 0.0),  # brief stop ~6s
        (240, 4.5),  # fast jog ~8s
        (180, 1.5),  # cool down ~6s
    ]
    total = sum(d for d,_ in phases)
    scale = 1800/total
    phases = [(max(1,int(d*scale)), s) for d,s in phases]
    # Adjust to exact 1800
    diff = 1800 - sum(d for d,_ in phases)
    if diff != 0:
        d0, s0 = phases[0]
        phases[0] = (d0+diff, s0)
    return phases


def random_waypoints() -> List[Vec2]:
    # Choose 6-7 interior waypoints to make realistic, smooth path
    rng = random.Random(123)
    wps = [Vec2(-0.8, -0.7)]
    for _ in range(6):
        wps.append(Vec2(rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.8)))
    wps.append(Vec2(0.7, -0.5))
    return wps


def catmull_rom(P0, P1, P2, P3, t):
    # t in [0,1]
    t2 = t*t; t3 = t2*t
    x = 0.5*((2*P1.x) + (-P0.x + P2.x)*t + (2*P0.x - 5*P1.x + 4*P2.x - P3.x)*t2 + (-P0.x + 3*P1.x - 3*P2.x + P3.x)*t3)
    y = 0.5*((2*P1.y) + (-P0.y + P2.y)*t + (2*P0.y - 5*P1.y + 4*P2.y - P3.y)*t2 + (-P0.y + 3*P1.y - 3*P2.y + P3.y)*t3)
    return Vec2(x,y)


def generate_path_samples(n: int, wps: List[Vec2]) -> List[Vec2]:
    # Sample a smooth path through waypoints via Catmull-Rom
    pts = []
    # Pad endpoints
    P = [wps[0]] + wps + [wps[-1]]
    segs = len(P) - 3
    per_seg = max(2, n // segs)
    for i in range(segs):
        P0,P1,P2,P3 = P[i],P[i+1],P[i+2],P[i+3]
        for k in range(per_seg):
            t = k / per_seg
            pts.append(catmull_rom(P0,P1,P2,P3,t))
    # Trim/pad to n
    if len(pts) < n:
        pts += [pts[-1]]*(n-len(pts))
    else:
        pts = pts[:n]
    # clamp to court bounds
    for p in pts:
        p.x = max(-1, min(1, p.x))
        p.y = max(-1, min(1, p.y))
    return pts


def synthesize_frames() -> List[Frame]:
    court = Court(); cam = Camera(); syn = Synthesis(); vols = Volumes()
    random.seed(syn.seed)

    # Build a long smooth spatial path we can traverse at varying speeds
    wps = random_waypoints()
    path_pts = generate_path_samples(6000, wps)  # oversample spatially

    # Motion phases define speed targets; we walk the path accordingly
    phases = make_motion_profile(syn.fps)

    frames: List[Frame] = []
    idx = 0  # index into path_pts
    last_pos = path_pts[idx]
    last_vx = 0.0; last_vy = 0.0

    # Parameters for acceleration realism
    max_acc = 3.0  # m/s^2 cap in meters space

    for (dur, target_speed) in phases:
        for i in range(dur):
            t = len(frames)/syn.fps
            # Current position in normalized coords
            pos = path_pts[idx]
            # Direction to next point
            nxt = path_pts[min(idx+1, len(path_pts)-1)]
            dx = nxt.x - pos.x; dy = nxt.y - pos.y
            # Convert direction to meters
            # Compute desired velocity magnitude (target_speed), ease in/out within phase
            # phase position u in [0,1]
            u = i/max(1,(dur-1))
            sm = target_speed * (0.6 + 0.4*ease_in_out(u))  # ease factor
            # Normalize direction
            mag = math.hypot(dx, dy) + 1e-6
            dirx = dx/mag; diry = dy/mag
            # Convert desired velocity to normalized units per second
            # vx_mps, vy_mps in meters; map to normalized units
            half_w = Court().width_m/2.0; half_d = Court().depth_m/2.0
            vxn = (sm * dirx) / half_w
            vyn = (sm * diry) / half_d
            # Integrate position with dt
            dt = 1.0/syn.fps
            new_x = pos.x + vxn*dt
            new_y = pos.y + vyn*dt
            # Clamp
            new_x = max(-1, min(1, new_x))
            new_y = max(-1, min(1, new_y))

            # Compute velocities/accelerations in meters for realism
            x_m, y_m = norm_to_meters(new_x, new_y, court)
            lx_m, ly_m = norm_to_meters(last_pos.x, last_pos.y, court)
            vx_mps = (x_m - lx_m)/dt
            vy_mps = (y_m - ly_m)/dt
            # Cap acceleration
            ax = (vx_mps - last_vx)/dt
            ay = (vy_mps - last_vy)/dt
            acc_mag = math.hypot(ax, ay)
            if acc_mag > max_acc:
                scale = max_acc/acc_mag
                ax *= scale; ay *= scale
                vx_mps = last_vx + ax*dt
                vy_mps = last_vy + ay*dt
                # Recompute position from capped velocity
                x_m = lx_m + vx_mps*dt
                y_m = ly_m + vy_mps*dt
                new_x, new_y = meters_to_norm(x_m, y_m, court)

            speed = math.hypot(vx_mps, vy_mps)
            inten = intensity_from_speed(speed, Volumes.anchors)
            dist = distance_from_camera(new_x, new_y, court, cam)
            dvol = distance_volume(dist, vols.distance_k, vols.distance_min_v)
            total_v = inten * dvol
            pan = pan_from_x(new_x)

            frames.append(Frame(
                frame=len(frames), t_s=t,
                x=new_x, y=new_y,
                vx=vx_mps, vy=vy_mps,
                ax=ax, ay=ay,
                speed_mps=speed,
                intensity_v=inten,
                distance_m=dist,
                distance_v=dvol,
                total_v=total_v,
                pan=pan,
            ))

            last_pos = Vec2(new_x, new_y)
            last_vx, last_vy = vx_mps, vy_mps
            idx = min(idx+1, len(path_pts)-1)

            if len(frames) >= Synthesis().n_frames:
                return frames
    return frames


def write_csv(frames: List[Frame], path: str):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame','t_s','x','y','vx_mps','vy_mps','ax_mps2','ay_mps2','speed_mps',
                    'intensity_v','distance_m','distance_v','total_v','pan'])
        for fr in frames:
            w.writerow([fr.frame, fr.t_s, fr.x, fr.y, fr.vx, fr.vy, fr.ax, fr.ay, fr.speed_mps,
                        fr.intensity_v, fr.distance_m, fr.distance_v, fr.total_v, fr.pan])

if __name__ == '__main__':
    frames = synthesize_frames()
    write_csv(frames, 'player_1_1800.csv')
    print('Wrote player_1_1800.csv')