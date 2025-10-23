import argparse
from pydub import AudioSegment
# from synth_data import synthesize_frames, write_csv
from audio_engine import render_footsteps
from synthetic_diagonal_run import generate_diagonal_run, write_csv
# from synthetic_upward_run import generate_upward_run
from config import Audio, Synthesis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basketball Sound Engine demo')
    parser.add_argument('--foot', type=str, default=None, help='Path to a footstep WAV/MP3 (optional)')
    parser.add_argument('--out', type=str, default='footsteps_demo.wav', help='Output WAV')
    parser.add_argument('--csv', type=str, default='player_1_1800.csv', help='CSV out for telemetry')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()


    # Generate synthetic data
    # frames = synthesize_frames()
    frames = generate_diagonal_run()


    # Duration from frames
    duration_ms = int((frames[-1].t_s + 2.0)*1000) # tail


    # Render audio
    audio_cfg = Audio(sample_rate=args.sr)
    mix = render_footsteps(frames, args.foot, duration_ms, audio_cfg)
    mix.export(args.out, format='wav')
    print(f'[ok] Wrote stereo demo → {args.out}')