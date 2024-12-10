"""
Matching Score (MS) measures generation relevance from multiple audio source conditions.
Suppose we obtain top M detected ground-truth labels, and top N detected labels from a classifier on a generated audio,
Then we would have 3 kinds of matchings:
1. True Positive: label is present in both ground truth and generated sets.
2. False Positive: label is not present in ground truth, but present in generation.
3. False Negative: label is present in ground truth, but not in generation.

Notice we don't consider True Negatives here because we are not interested in them for a generation task.

We then compute the following sub-metrics in MS:
1. Precision
2. Recall
3. F1 Score
"""

import numpy as np

from ssv2a.data.utils import read_wav_file
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm
from BEATs import BEATs, BEATsConfig


class AudioLabelDataset(Dataset):
    def __init__(self, meta_csv, aud_folder):
        super().__init__()
        self.folder = Path(aud_folder)
        self.df = pd.read_csv(meta_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        aud_f = self.folder / f"{row['id']}.wav"
        waveform = read_wav_file(aud_f, segment_length=163840, sampling_rate=16000)
        return waveform, row['labels'], row['id']


def collate_audiolabels(data):
    return torch.stack([torch.from_numpy(d[0]) for d in data]).squeeze().float(), [d[1] for d in data], [d[2] for d in data]


def get_ms(gt_aud_dir, gen_aud_dir, beats_ckpt, M=10, N=50, bs=64, device='cuda'):
    ckpt = torch.load(beats_ckpt)

    cfg = BEATsConfig(ckpt['cfg'])
    label_dict = ckpt['label_dict']

    beats = BEATs(cfg)
    beats.load_state_dict(ckpt['model'])
    beats.to(device)
    beats.eval()

    with torch.no_grad():
        tps, fps, fns = [], [], []

        gt_aud_fs = [str(p) for p in Path(gt_aud_dir).glob('*.wav')]
        ids = [p.name.replace('.wav', '') for p in Path(gt_aud_dir).glob('*.wav')]

        for s in tqdm(range(0, len(gt_aud_fs), bs)):
            e = min(len(gt_aud_fs), s + bs)
            wave = []
            for aud_f in gt_aud_fs[s:e]:
                wave.append(torch.from_numpy(read_wav_file(str(aud_f), segment_length=163840, sampling_rate=16000)))
            wave = torch.stack(wave).squeeze().float().to(device)

            B = wave.shape[0]

            # prepare ground truth labels
            labels = []
            padding_mask = torch.zeros(wave.shape).bool().to(device)
            gt_pred = beats.extract_features(wave.to(device), padding_mask=padding_mask)[0]
            for i, (label_prob, label_idx) in enumerate(zip(*gt_pred.topk(k=M))):
                lbs = [label_dict[idx.item()] for idx in label_idx]
                labels.append(lbs)

            # predict labels for generated audios
            gen_wave = []
            gen_aud_dir = Path(gen_aud_dir)
            for i, vid in enumerate(ids[s:e]):
                try:
                    aud_f = next(gen_aud_dir.glob(f'*{vid}*'))
                    gen_wave.append(torch.from_numpy(read_wav_file(str(aud_f), segment_length=163840, sampling_rate=16000)))
                except Exception as e:
                    del labels[i]
                    print(f'Ignored non-present audio {vid}.wav')
            gen_wave = torch.stack(gen_wave).squeeze().float().to(device)
            padding_mask = torch.zeros(gen_wave.shape).bool().to(device)
            gen_pred = beats.extract_features(gen_wave.to(device), padding_mask=padding_mask)[0]
            gen_labels = []
            for i, (label_prob, label_idx) in enumerate(zip(*gen_pred.topk(k=N))):
                lbs = [label_dict[idx.item()] for idx in label_idx]
                gen_labels.append(lbs)

            # matching
            tp, fp, fn = [], [], []
            for i in range(len(labels)):
                gt_set = set(labels[i])
                gen_set = set(gen_labels[i])
                true_pos = gt_set.intersection(gen_set)
                tp.append(len(true_pos))
                fp.append(len(gen_set.difference(true_pos)))
                fn.append(len(gt_set.difference(true_pos)))
            tps += tp
            fps += fp
            fns += fn

        # compute metrics
        tps = np.array(tps, dtype=np.float64)
        fps = np.array(fps, dtype=np.float64)
        fns = np.array(fns, dtype=np.float64)

        precision = np.mean(tps / (tps + fps)).item()
        recall = np.mean(tps / (tps + fns)).item()
        f1_score = (2 * precision * recall / (precision + recall))

        return precision, recall, f1_score

