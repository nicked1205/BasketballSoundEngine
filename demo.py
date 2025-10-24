import argparse
from pydub import AudioSegment
from audio_engine import render_footsteps
from config import Audio

# --- optional imports for modes ---
from synthetic_diagonal_run import generate_diagonal_run, write_csv
from load_data import load_frames_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Sound Engine Demo")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic",
                        help="Choose between synthetic or real tracking data")
    parser.add_argument("--csv", type=str, default="comprehensive_data.csv",
                        help="CSV file for real data mode")
    parser.add_argument("--player-name", type=str, default=None,
                    help="Player name to render (case-insensitive, matches player_name column)")
    parser.add_argument("--foot", type=str, default="./assets/footsteps.mp3",
                        help="Path to footstep sound")
    parser.add_argument("--out", type=str, default="footsteps_demo.wav",
                        help="Output WAV file")
    parser.add_argument("--sr", type=int, default=48000,
                        help="Audio sample rate")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate for real tracking data")
    args = parser.parse_args()

    # --- choose data source ---
    if args.mode == "synthetic":
        print("[mode] Using synthetic data generator...")
        frames = generate_diagonal_run()
        write_csv(frames, "synthetic_data.csv")
    else:
        print(f"[mode] Using real data from {args.csv}")
        frames = load_frames_from_csv(args.csv, fps=args.fps, player_name=args.player_name)

    # --- compute duration ---
    duration_ms = int((frames[-1].t_s + 2.0) * 1000)

    # --- render audio ---
    audio_cfg = Audio(sample_rate=args.sr)
    mix = render_footsteps(frames, args.foot, duration_ms, audio_cfg)
    mix.export(args.out, format="wav")
    print(f"[ok] Exported stereo output → {args.out}")
