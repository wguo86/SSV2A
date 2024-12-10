import sys
sys.path.insert(0, './SSV2A')
import argparse
import glob

from ssv2a.model.pipeline import Pipeline, image_to_audio, video_to_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSV2A')
    parser.add_argument('--cfg', type=str, help='Model Config File')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained Checkpoint')
    parser.add_argument('--agg_ckpt', type=str, default=None, help='Pretrained Aggregator Checkpoint')
    parser.add_argument('--vid_dir', type=str, default=None, help='Path to the video files')
    parser.add_argument('--frames', type=int, default=64, help='Total frames to pass to Aggregator per video')
    parser.add_argument('--out_dir', type=str, default='./output', help='Path to save the output audios to')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--var_samples', type=int, default=64, help='variational samples')
    parser.add_argument('--cycle_its', type=int, default=64, help='number of Cycle Mix iterations')
    parser.add_argument('--cycle_samples', type=int, default=64, help='number of Cycle Mix samples')
    parser.add_argument('--duration', type=int, default=10, help='generation duration in seconds')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Computation Device')
    args = parser.parse_args()

    pipe = Pipeline(config=args.cfg, pretrained=args.ckpt, device=args.device)
    vids = glob.glob(f'{args.vid_dir}/*')
    video_to_audio(args.cfg, args.ckpt, vids, args.agg_ckpt, args.out_dir,
                   agg_var_samples=1, frames=args.frames,
                   batch_size=args.bs, var_samples=args.var_samples,
                   cycle_its=args.cycle_its, cycle_samples=args.cycle_samples,
                   duration=args.duration, seed=args.seed, device=args.device)

