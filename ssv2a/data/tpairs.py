"""
This module defines the data structure of tags-audio pairs. This is a many-to-one matching.
"""
import os
import pickle
import random
from pathlib import Path

import numpy as np
import clip
import pandas as pd
import torch
from tqdm.auto import tqdm

from ssv2a.data.pairs import PairDataset, load_pair
from ssv2a.data.utils import normalize_wav
from ssv2a.model.clap import CLAP
from ssv2a.model.dalle2_prior import Dalle2Prior


class TagPair:
    def __init__(self, pid, caption, tags, aud_wave):
        self.pid = pid

        self.aud_wave = aud_wave
        self.aud_clap = None

        self.caption = caption
        self.caption_clip = None
        self.caption_clip_prior = None

        self.tags = tags
        self.tag_clips = None
        self.tag_clips_prior = None
        self.tag_claps = None

        self.aug_img_id = None
        self.aug_clip = None

        self.aug_tag_img_ids = None
        self.aug_tag_clips = None

    def save(self, folder):
        with open(f"{folder}/{self.pid}.pkl", 'wb') as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_tpair(pdir, pid=None):
    if pid is not None:
        pdir = Path(pdir) / f'{pid}.pkl'
    with open(pdir, 'rb') as fp:
        return pickle.load(fp)


def tpairs2aclaps(pairs, device='cuda'):
    return torch.cat([torch.from_numpy(p.aud_clap).unsqueeze(0) for p in pairs]).to(device)


def tpairs2cclips(pairs, device='cuda'):
    return torch.cat([torch.from_numpy(p.caption_clip).unsqueeze(0) for p in pairs]).to(device)


def tpairs2tclips(pairs, max_length=0, use_prior=False, device='cuda'):
    if max_length == 0:
        if use_prior:
            return torch.cat([torch.from_numpy(p.tag_clips_prior) for p in pairs]).to(device)
        else:
            return torch.cat([torch.from_numpy(p.tag_clips) for p in pairs]).to(device)
    else:
        tclips = []
        for p in pairs:
            L, E = p.tag_clips.shape

            if use_prior:
                seq = torch.from_numpy(p.tag_clips_prior)
            else:
                seq = torch.from_numpy(p.tag_clips)

            # truncate and random sample (or shuffle if short)
            idx = random.sample(range(L), min(L, max_length))
            seq = seq[idx]

            if L < max_length:  # pad if short
                seq = torch.cat([seq, torch.zeros(max_length - L, E)])

            tclips.append(seq)
        return torch.stack(tclips).to(device)


class TagPairDataset(PairDataset):
    def __getitem__(self, idx):
        for df_idx, df in enumerate(self.pairs_dfs):
            if idx < len(df):
                entry = df.iloc[idx]
                return load_tpair(self.pairs_roots[df_idx], entry['pid']), entry['category']
            idx -= len(df)

    def merge(self, other_set):
        return TagPairDataset(split=self.split,
                              pairs_dfs=self.pairs_dfs + other_set.pairs_dfs,
                              pairs_roots=self.pairs_roots + other_set.pairs_roots)


class MixedPairDataset:
    def __init__(self, pairs_meta_fs=None, tpairs_meta_fs=None, split='train', pairs_dfs=None, pairs_roots=None):
        if pairs_meta_fs is None:
            pairs_meta_fs = []
        self.split = split

        self.pairs_roots = []
        self.pairs_dfs = []

        for f in pairs_meta_fs:
            self.pairs_roots.append(Path(f).parent / split)
            df = pd.read_csv(f)
            df = df[df['split'] == split]
            self.pairs_dfs.append((df, 'pairs'))

        for f in tpairs_meta_fs:
            self.pairs_roots.append(Path(f).parent / split)
            df = pd.read_csv(f)
            df = df[df['split'] == split]
            self.pairs_dfs.append((df, 'tpairs'))

    def __len__(self):
        return sum([len(df) for df, _ in self.pairs_dfs])

    def __getitem__(self, idx):
        for df_idx, (df, dft) in enumerate(self.pairs_dfs):
            if idx < len(df):
                entry = df.iloc[idx]
                if dft == 'pairs':
                    return load_pair(self.pairs_roots[df_idx], entry['pid']), entry['category']
                else:
                    return load_tpair(self.pairs_roots[df_idx], entry['pid']), entry['category']
            idx -= len(df)


def clip_embed_tpairs(pids, pdir, bs=64, clip_version='ViT-L/14', device='cuda'):
    with (torch.no_grad()):
        model, preprocess = clip.load(clip_version, device=device)

        # embed caption
        print(f'Embedding {len(pids)} captions into CLIP:')
        for s in tqdm(range(0, len(pids), bs)):
            e = min(len(pids), s + bs)
            pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
            # pairs = [p for p in pairs if p.caption_clip is None]

            if len(pairs) == 0:
                continue

            ts = [p.caption for p in pairs]
            ts = clip.tokenize(ts, truncate=True).to(device)
            ts = model.encode_text(ts).float()
            ts = ts / ts.norm(p=2, dim=-1, keepdim=True)
            ts = ts.detach().cpu().numpy()
            ts = np.nan_to_num(ts)

            for i, p in enumerate(pairs):
                p.caption_clip = ts[i]
                p.save(pdir)

        # embed tags
        print(f'Embedding {len(pids)} bags of tags into CLIP:')
        for s in tqdm(range(0, len(pids), bs)):
            e = min(len(pids), s + bs)
            pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
            # pairs = [p for p in pairs if p.tag_clips is None]

            if len(pairs) == 0:
                continue

            ts = []
            for p in pairs:
                ts += p.tags
            if len(ts) == 0:
                continue
            ts = clip.tokenize(ts, truncate=True).to(device)
            ts = model.encode_text(ts).float()
            ts = ts / ts.norm(p=2, dim=-1, keepdim=True)
            ts = ts.detach().cpu().numpy()
            ts = np.nan_to_num(ts)

            step = 0
            jumps = [len(p.tags) for p in pairs]
            for i, p in enumerate(pairs):
                p.tag_clips = ts[step:step + jumps[i]]
                step += jumps[i]
                p.save(pdir)


def clap_embed_tpairs(pids, pdir, bs=64,
                      clap_version='audioldm-s-full-v2', device='cuda'):
    with torch.no_grad():
        # seg_length = int(duration * 102.4) * 160
        clap = CLAP(clap_version=clap_version, embed_mode='audio', device=device)
        del_pids = []

        # embed source audios
        print(f'Embedding {len(pids)} source audios into CLAP:')
        for s in tqdm(range(0, len(pids), bs)):
            e = min(len(pids), s + bs)
            pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
            pairs = [p for p in pairs if p.aud_clap is None]

            if len(pairs) == 0:
                continue

            for p in pairs:
                if p.aud_wave.shape[0] > 100:
                    waveform = normalize_wav(p.aud_wave)
                    waveform = waveform[None, ...]
                    waveform = waveform / np.max(np.abs(waveform))
                    waveform = np.nan_to_num(0.5 * waveform)
                else:
                    print(f'Delete Short Audio: {p.pid}')
                    os.remove(pdir / f'{p.pid}.pkl')
                    del_pids.append(p.pid)
                    continue

                embeds = clap.model(torch.from_numpy(waveform).float()).detach().cpu().numpy()
                embeds = np.squeeze(np.nan_to_num(embeds))

                p.aud_clap = embeds
                p.save(pdir)

        # embed tags
        print(f'Embedding {len(pids)} bags of tags into CLAP:')
        for pid in del_pids:
            pids.remove(pid)
        clap = CLAP(clap_version=clap_version, embed_mode='text', device=device)
        for s in tqdm(range(0, len(pids), bs)):
            e = min(len(pids), s + bs)
            pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
            pairs = [p for p in pairs if p.tag_claps is None]

            if len(pairs) == 0:
                continue

            ts = []
            for p in pairs:
                ts += p.tags
            rts = np.empty((len(ts), 512))
            for s1 in range(0, len(ts), bs):
                e1 = min(len(ts), s1 + bs)
                rts[s1:e1] = clap.model(ts[s1:e1]).squeeze().float().detach().cpu().numpy()

            step = 0
            jumps = [len(p.tags) for p in pairs]
            for i, p in enumerate(pairs):
                p.tag_claps = rts[step:step + jumps[i]]
                step += jumps[i]
                p.save(pdir)


# translate tpairs from text-audio data to image-audio data
def prior_embed_tpairs(pids, pdir, cfg, ckpt, bs=64, n_samples_per_batch=2, cond_scale=1, device='cuda'):
    model = Dalle2Prior(cfg, ckpt, device=device)

    print(f'Translating {len(pids)} captions from CLIP text space to image space:')
    for s in tqdm(range(0, len(pids), bs)):
        e = min(len(pids), s + bs)
        pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
        pairs = [p for p in pairs if (not hasattr(p, 'caption_clip_prior')) or p.caption_clip_prior is None]

        if len(pairs) == 0:
            continue

        caps = [p.caption for p in pairs]
        cap_clips = model.sample(caps, n_samples_per_batch=n_samples_per_batch, cond_scale=cond_scale)
        cap_clips = np.nan_to_num(cap_clips.detach().cpu().float().numpy())

        for i, p in enumerate(pairs):
            p.caption_clip_prior = cap_clips[i]
            p.save(pdir)

    print(f'Translating {len(pids)} bags of tags from CLIP text space to image space:')
    for s in tqdm(range(0, len(pids), bs)):
        e = min(len(pids), s + bs)
        pairs = [load_tpair(pdir, pid) for pid in pids[s:e]]
        pairs = [p for p in pairs if (not hasattr(p, 'tag_clips_prior')) or p.tag_clips_prior is None]

        if len(pairs) == 0:
            continue

        tags = []
        for p in pairs:
            tags += p.tags
        if len(tags) == 0:
            continue

        tag_clips = []
        for s1 in range(0, len(tags), bs):
            e1 = min(len(tags), s1 + bs)
            tag_clips.append(model.sample(tags[s1:e1], n_samples_per_batch=n_samples_per_batch, cond_scale=cond_scale))
        tag_clips = np.nan_to_num(torch.cat(tag_clips, dim=0).detach().cpu().float().numpy())

        step = 0
        jumps = [len(p.tags) for p in pairs]
        for i, p in enumerate(pairs):
            p.tag_clips_prior = tag_clips[step:step + jumps[i]]
            step += jumps[i]
            p.save(pdir)

