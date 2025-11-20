import argparse
from pydub import AudioSegment
from audio_engine import render_footsteps, render_bounces, render_claps
from config import Audio
from load_data import load_frames, load_bounces_from_json, load_score_events

# --- optional imports for modes ---
from synthetic_diagonal_run import generate_diagonal_run, write_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Sound Engine Demo")
    parser.add_argument("--source-type", type=str, choices=["csv", "json"], default="json",
                        help="Select input format: 'csv' for legacy or 'json' for detection data")
    parser.add_argument("--data", type=str, default="./data/people_tracks_5.json",
                        help="Path to input data file (CSV or JSON)")
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=25.00195)
    parser.add_argument("--foot", type=str, default="./assets/footsteps.mp3")
    parser.add_argument("--out", type=str, default="res.mp3")
    parser.add_argument("--bounces", type=str, default="./data/bounce_5_test.json")
    parser.add_argument("--bounce-sound", type=str, default="./assets/bounce.mp3")
    parser.add_argument("--game", type=int, default=5, help="Game ID")
    parser.add_argument("--sr", type=int, default=48000,
                        help="Audio sample rate")
    args = parser.parse_args()

    frames = load_frames(
        path=args.data,
        source_type=args.source_type,
        fps=args.fps,
        player_name=args.player_name,
    )

    # --- compute duration ---
    duration_ms = int((frames[-1].t_s + 2.0) * 1000)

    # --- render audio ---
    audio_cfg = Audio(sample_rate=args.sr)
    mix_foot, mix_squeak, combined = render_footsteps(frames, args.foot, duration_ms, audio_cfg)
    mix_foot.export(f"footsteps_{args.game}.mp3", format="mp3")
    mix_squeak.export(f"squeaks_{args.game}.mp3", format="mp3")

    bounces = load_bounces_from_json(
        path=args.bounces,
        fps=args.fps,
    )
    mix_bounce = render_bounces(bounces, args.bounce_sound, duration_ms, audio_cfg)
    mix_bounce.export(f"bounces_{args.game}_test.mp3", format="mp3")

    # --- Load and render crowd claps ---
    score_events = load_score_events("./data/shots_5.csv", fps=args.fps)
    mix_clap = render_claps(score_events, "./assets/claps.wav", duration_ms, audio_cfg, fps=args.fps)
    mix_clap.export(f"claps_{args.game}.mp3", format="mp3")

    final_mix = combined.overlay(mix_bounce).overlay(mix_clap)
    final_mix.export("res.mp3", format="mp3")
    print("[ok] Exported tracks: footsteps.mp3, squeaks.mp3, bounces.mp3 and full mix →", args.out)
