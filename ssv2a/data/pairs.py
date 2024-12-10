import pickle
import json
from enum import Enum
from pathlib import Path

import numpy as np
import clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ssv2a.data.utils import read_wav_file
from ssv2a.model.clap import CLAP


class Mode(Enum):
    NA = -1
    AUDIO = 0
    IMAGE = 1
    VIDEO = 2
    TEXT = 3
    LABEL = 4


# batch embed the sources of a list of pairs with CLIP
def clip_embed_pairs(pids, pairs_dir, model=None, preprocess=None, version='ViT-L/14', batch_size=64, device='cuda'):
    with torch.no_grad():
        if model is None:
            model, preprocess = clip.load(version, device=device)
        print(f'Embedding {len(pids)} pairs into CLIP:')
        for locality in ['local', 'global', 'context']:
            for s in tqdm(range(0, len(pids), batch_size)):
                e = min(len(pids), s + batch_size)
                pairs = {pid: load_pair(pairs_dir, pid) for pid in pids[s:e]}
                texts = [(pid, p.get_sources(f'{locality}_srcs')) for pid, p in pairs.items()
                         if p.mode == Mode.TEXT or p.mode == Mode.LABEL]
                images = [(pid, p.get_sources(f'{locality}_srcs')) for pid, p in pairs.items()
                          if (p.mode == Mode.IMAGE or p.mode == Mode.VIDEO) and p.data[f'{locality}_clips'] is None]
                # embed images
                if len(images) > 0:
                    imgs = []
                    for _, image_set in images:
                        imgs += image_set
                    if len(imgs) == 0:
                        continue
                    img_arr = []
                    img_sz = None
                    for img in imgs:
                        try:
                            processed_img = preprocess(img).unsqueeze(0)
                            if img_sz is None:
                                img_sz = processed_img.shape
                            img_arr.append(processed_img)
                        except Exception as e:
                            print(f'Illegal image {img}.')
                            img_arr.append(torch.zeros(img_sz))
                    imgs = torch.cat(img_arr).to(device)
                    img_embeds = model.encode_image(imgs)
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    img_embeds = img_embeds.detach().cpu().numpy()
                    img_embeds = np.nan_to_num(img_embeds)
                    idx = 0
                    for pid, image_set in images:
                        step = len(image_set)
                        pairs[pid].data[f'{locality}_clips'] = img_embeds[idx:idx + step]
                        pairs[pid].save(pairs_dir)
                        idx += step
                # embed texts
                if len(texts) > 0:
                    ts = []
                    for _, text_set in texts:
                        ts += text_set
                    ts = clip.tokenize(ts).to(device)
                    ts_embeds = model.encode_text(ts)
                    ts_embeds = ts_embeds / ts_embeds.norm(dim=-1, keepdim=True)
                    ts_embeds = ts_embeds.detach().cpu().numpy()
                    ts_embeds = np.nan_to_num(ts_embeds)
                    idx = 0
                    for pid, text_set in texts:
                        step = len(text_set)
                        pairs[pid].data[f'{locality}_clips'] = ts_embeds[idx:idx + step]
                        pairs[pid].save(pairs_dir)
                        idx += step


# batch embed the audios of a list of pairs with CLAP (AudioLDM2 flavor)
def clap_embed_pairs(pids, pairs_dir, model=None, clap_version='audioldm-s-full-v2',
                     duration=10, batch_size=256, sampling_rate=16000, device='cuda'):
    with torch.no_grad():
        seg_length = int(duration * 102.4) * (sampling_rate // 100)
        if model is None:
            clap = CLAP(clap_version=clap_version, embed_mode='audio', sample_rate=sampling_rate, device=device)
        else:
            clap = model

        # embed
        print(f'Embedding {len(pids)} pairs into CLAP:')
        for s in tqdm(range(0, len(pids), batch_size)):
            e = min(len(pids), s + batch_size)
            pairs = [load_pair(pairs_dir, pid) for pid in pids[s:e]]
            wavs = np.concatenate([read_wav_file(p.data['audio'], seg_length, sampling_rate=sampling_rate)
                                   for p in pairs])
            embeds = clap.model(torch.from_numpy(wavs).float()).detach().cpu().numpy()
            embeds = np.nan_to_num(embeds)
            # update pairs and save
            for i, p in enumerate(pairs):
                p.data['clap'] = embeds[i]
                p.save(pairs_dir)


def pairs2clips(pairs, clips_type, max_length=0, device='cuda'):  # local clips or global clips
    if max_length == 0:
        clips = []
        for p in pairs:
            clips.append(torch.from_numpy(p.data[clips_type]))
        return torch.cat(clips).float().to(device)
    else:
        clips = torch.zeros(len(pairs), max_length, pairs[0].data[clips_type].shape[-1])
        for i, p in enumerate(pairs):
            emb = torch.from_numpy(p.data[clips_type])
            emb = emb.reshape(-1, emb.shape[-1])
            clips[i, :emb.shape[0], :] = emb
        return clips.float().to(device)


def pairs2claps(pairs, align_clips=None, device='cuda'):  # optionally, align with multiple clips by duplication
    if align_clips:
        claps = []
        for p in pairs:
            claps.append(
                torch.from_numpy(np.repeat(p.data['clap'], p.data[align_clips].shape[0], axis=0)))
        return torch.cat(claps).float().to(device)
    else:
        return torch.cat([torch.from_numpy(p.data['clap']) for p in pairs]).float().to(device)


"""
A pair can be image-audio, text-audio, video-audio.
Notice that this can be a many-to-one pair because a video has multiple frames.
"""


class Pair:
    def __init__(self, global_srcs, context_srcs, local_srcs, localities, aud, mode, pid):
        self.mode = mode
        self.data = {
            'pid': pid,  # unique id of this pair
            'mode': mode.value,
            # if text/label, a list of strings, otherwise, a list of image files (for videos, extracted visual frames)
            'global_srcs': global_srcs,
            'context_srcs': context_srcs,
            'local_srcs': local_srcs,
            'localities': localities,
            'audio': aud,
            'local_clips': None,  # clip embeddings of single sources
            'global_clips': None,  # clip embeddings of the original data
            'context_clips': None,
            'clap': None,  # clap embedding of audio
        }

    def get_sources(self, src_type):  # return preprocessed source data
        src = self.data[src_type]
        if src is None:
            raise NotImplementedError('Source data corrupted for this pair!')

        # if label, preprocess by prompt augmentation
        if self.mode == Mode.LABEL:
            prompts = []
            for s in src:
                prompts += [f"the sound of {s}",
                            f"the sound {s} makes",
                            f"the audio of {s}"]
            return prompts
        elif self.mode == Mode.TEXT:
            return src

        images = []
        if self.mode == Mode.IMAGE or self.mode == Mode.VIDEO:
            images += [Image.open(s) for s in src]
        return images

    def save(self, folder):
        with open(f"{folder}/{self.data['pid']}.pickle", 'wb') as fp:
            pickle.dump(self.data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        content = {
            'pid': self.data['pid'],  # unique id of this pair
            'mode': self.mode.name,
            'global_srcs': self.data['global_srcs'],
            'context_srcs': self.data['context_srcs'],
            'local_srcs': self.data['local_srcs'],
            'audio': self.data['audio']
        }
        return json.dumps(content, indent=4)


# load a pair by id and return it
def load_pair(pdir, pid=None):
    pair = Pair(None, None, None, None, None, Mode.NA, pid)
    if pid is not None:
        pdir = Path(pdir) / f'{pid}.pickle'
    with open(pdir, 'rb') as fp:
        pair.data = pickle.load(fp)
        pair.mode = Mode(pair.data['mode'])
    return pair


def collate_pairs(data):
    return [d[0] for d in data], [d[1] for d in data]


class PairDataset(Dataset):
    def __init__(self, pairs_meta_file=None, split='train', pairs_dfs=None, pairs_roots=None):
        self.split = split

        if pairs_meta_file is not None:
            self.pairs_roots = [Path(pairs_meta_file).parent / split]
            df = pd.read_csv(pairs_meta_file)
            df = df[df['split'] == split]
            self.pairs_dfs = [df]

        elif pairs_dfs is not None and pairs_roots is not None:
            self.pairs_dfs = pairs_dfs
            self.pairs_roots = pairs_roots

        else:
            raise NotImplementedError('Illegal dataset parameters.')

    def __len__(self):
        return sum([len(df) for df in self.pairs_dfs])

    def __getitem__(self, idx):
        for df_idx, df in enumerate(self.pairs_dfs):
            if idx < len(df):
                entry = df.iloc[idx]
                return load_pair(self.pairs_roots[df_idx], entry['pid']), entry['category']
            idx -= len(df)

    # merge with another PairDataset, non-destructively
    def merge(self, other_set):
        return PairDataset(split=self.split,
                           pairs_dfs=self.pairs_dfs + other_set.pairs_dfs,
                           pairs_roots=self.pairs_roots + other_set.pairs_roots)

