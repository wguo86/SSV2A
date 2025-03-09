import sys
sys.path.insert(0, './SSV2A')
import argparse

from ssv2a.model.pipeline import Pipeline, srcs_to_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSV2A')
    parser.add_argument('--cfg', type=str, help='Model Config File')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained Checkpoint')
    parser.add_argument('--dalle2_cfg', type=str, default=None, help='DALLE2 Prior Config File')
    parser.add_argument('--dalle2_ckpt', type=str, default=None, help='DALLE2 Prior Pretrained Checkpoint')
    parser.add_argument('--images', nargs='+', type=str, default=None, help='Image Conditions')
    parser.add_argument('--texts', nargs='+', type=str, default=None, help='Text Conditions')
    parser.add_argument('--audios', nargs='+', type=str, default=None, help='Image Conditions')
    parser.add_argument('--out_dir', type=str, default='./output', help='Path to save the output audio to')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--var_samples', type=int, default=64, help='variational samples')
    parser.add_argument('--cycle_its', type=int, default=64, help='number of Cycle Mix iterations')
    parser.add_argument('--cycle_samples', type=int, default=64, help='number of Cycle Mix samples')
    parser.add_argument('--duration', type=int, default=10, help='generation duration in seconds')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Computation Device')
    args = parser.parse_args()

    pipe = Pipeline(config=args.cfg, pretrained=args.ckpt, device=args.device)
    srcs = {
        'image': [] if args.images is None else args.images,
        'text': [] if args.texts is None else args.texts,
        'audio': [] if args.audios is None else args.audios,
    }
    srcs_to_audio(srcs, args.out_dir,
                  config=args.cfg, pretrained=args.ckpt,
                  dalle2_cfg=args.dalle2_cfg, dalle2_ckpt=args.dalle2_ckpt,
                  shuffle_remix=True, cycle_its=args.cycle_its, cycle_samples=args.cycle_samples,
                  var_samples=args.var_samples, batch_size=args.bs, seed=args.seed,
                  duration=args.duration, device=args.device)

