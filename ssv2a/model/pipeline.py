import copy
import gc
import json
import os.path
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import soundfile as sf

from ssv2a.data.detect import detect
from ssv2a.data.tpairs import tpairs2tclips
from ssv2a.data.utils import clip_embed_images, get_timestamp, save_wave, set_seed, emb2seq, batch_extract_frames, \
    prior_embed_texts
from ssv2a.model.aggregator import Aggregator
from ssv2a.model.clap import clap_embed_auds
from ssv2a.model.aldm import build_audioldm, emb_to_audio
from ssv2a.model.generator import Generator
from ssv2a.model.manifold import Manifold
from ssv2a.model.remixer import Remixer


class Pipeline(nn.Module):
    def __init__(self, config, pretrained=None, device='cuda'):
        super().__init__()
        if not isinstance(config, dict):
            with open(config, 'r') as fp:
                config = json.load(fp)
        self.ckpt_path = Path(config['checkpoints'])
        self.config = config
        self.device = device
        config['device'] = device
        self.clip_dim = config['clip_dim']
        self.clap_dim = config['clap_dim']
        self.fold_dim = config['manifold_dim']

        # SDV2A Manifold
        self.manifold = Manifold(**config)
        # if the generator is just a linear operation, ignore modelling
        if config['generator']['disabled']:
            self.generator = None
            self.skip_gen_model = True
        elif config['generator']['arch'] == 'linear':
            self.generator = self.linear_generator
            self.skip_gen_model = True
        else:
            self.generator = Generator(**config)
            self.skip_gen_model = False

        # SDV2A Remixer
        self.remixer = Remixer(**config)

        # if there is any pretrained, load
        if pretrained:
            self.load(pretrained)

        # timestamp
        self.timestamp = get_timestamp()

    def linear_generator(self, fold_clips):
        gen_claps = self.manifold.clap_encoder.model.solve(fold_clips)
        return gen_claps, 0  # add a fake kl loss term to align with other generator models

    def save(self, filepath):
        state = {
            'timestamp': get_timestamp(),
            'manifold_state': self.manifold.state_dict(),
            'generator_state': None if self.skip_gen_model else self.generator.state_dict(),
            'remixer_state': self.remixer.state_dict()
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location='cpu')
        mia_states = []

        if 'manifold_state' in state:
            self.manifold.load_state_dict(state['manifold_state'])
        else:
            mia_states.append('manifold')

        if 'generator_state' in state and not self.skip_gen_model:
            self.generator.load_state_dict(state['generator_state'])
        else:
            mia_states.append('generator')

        if 'remixer_state' in state:
            self.remixer.load_state_dict(state['remixer_state'])
        else:
            mia_states.append('remixer')

        if len(mia_states) > 0:
            print(f"These states are missing in the model checkpoint supplied:\n"
                  f"{' '.join(mia_states)}\n"
                  f"Inference will be funky if these modules are involved without training!")

        self.timestamp = state['timestamp']

    def __str__(self):
        return (f"SDV2A@{self.timestamp}"
                f"{json.dumps(self.config, sort_keys=True, indent=4)}")

    # postprocessing: cycle generation
    def cycle_mix(self, clips, fixed_src=None, its=1, var_samples=64, samples=16, shuffle=True):
        """
        clips: [B, slot, E]  (global clip injected at first token)
        """
        B = clips.shape[0]
        rt_claps = torch.empty(B, 512).to(self.device)  # [B, E]
        rt_scores = torch.ones(B).to(self.device)  # [B]
        src = self.manifold.fold_clips(clips, var_samples=var_samples, normalize=False)
        for i in range(its + 1):  # one more round to include its = 0 (no recursion)
            if fixed_src is not None:  # reinject audio sources
                src_mask = torch.sum(clips.bool(), dim=-1).bool().logical_not()  # zero slots
                src[src_mask, ...] = fixed_src[src_mask, ...]

            src_claps = self.generator.fold2claps(src, var_samples=var_samples)

            src_claps[:, 0, :] = 0  # suppress global clap in clap score later

            if i > 0:  # recursion, inject the best clap as the global source code
                src[:, 0, :] = self.manifold.fold_claps(rt_claps, var_samples=var_samples, normalize=False)

            # sample remixed claps
            remix_claps = torch.empty(src_claps.shape[0], samples, 512).to(self.device)  # [B, S, E]
            for j in range(samples):
                if self.remixer.guidance == 'generator':
                    remix_claps[:, j, :] = self.remixer.sample(self.generator.fold2claps(src, var_samples=var_samples),
                                                               clips,
                                                               var_samples=var_samples, normalize=True, shuffle=shuffle)
                elif self.remixer.guidance == 'manifold+generator':
                    src_code = torch.cat([src, self.generator.fold2claps(src, var_samples=var_samples)], dim=-1)
                    remix_claps[:, j, :] = self.remixer.sample(src_code, clips,
                                                               var_samples=var_samples, normalize=True, shuffle=shuffle)
                else:
                    remix_claps[:, j, :] = self.remixer.sample(src, clips,
                                                               var_samples=var_samples, normalize=True, shuffle=shuffle)

            # select the remixed clap with highest CLAP-Score, can use std or mean
            clap_score = torch.einsum('bmi,bni->bmn', remix_claps, src_claps)  # [B, S, slot]
            clap_score /= (torch.einsum('bmi,bni->bmn',
                                        remix_claps.norm(dim=-1, keepdim=True),
                                        src_claps.norm(dim=-1, keepdim=True)) + 1e-6)
            clap_score = torch.std(clap_score, dim=-1)  # [B, S, slot] -> [B, S]
            best_remix_claps = torch.argmin(clap_score, dim=-1)
            clap_score = clap_score[torch.arange(B), best_remix_claps]
            updates = clap_score < rt_scores
            rt_scores[updates] = clap_score[updates]
            best_remix_claps = remix_claps[torch.arange(B), best_remix_claps]  # [B, S, E] -> [B, E]
            rt_claps[updates] = best_remix_claps[updates]

        return rt_claps

    # reconstruct an audio's clap
    def recon_claps(self, claps, var_samples=64):
        fold_claps = self.manifold.fold_claps(claps, var_samples=var_samples)
        gen_claps = self.generator.fold2claps(fold_claps, var_samples=var_samples)
        return gen_claps

    def clips2foldclaps(self, clips, var_samples=64):
        fold_clips = self.manifold.fold_clips(clips, var_samples=var_samples, normalize=False)
        gen_claps = self.generator.fold2claps(fold_clips, var_samples=var_samples)
        return gen_claps

    def clips2folds(self, clips, var_samples=64, normalize=False):
        fold_clips = self.manifold.fold_clips(clips, var_samples=var_samples, normalize=normalize)
        return fold_clips

    def claps2folds(self, claps, var_samples=64, normalize=False):
        fold_claps = self.manifold.fold_claps(claps, var_samples=var_samples, normalize=normalize)
        return fold_claps

    def clips2clap(self, clips, var_samples=64, normalize=False):
        src = self.clips2folds(clips, var_samples=var_samples, normalize=False)
        if self.remixer.guidance == 'generator':
            src = self.generator.fold2claps(src, var_samples=var_samples)
        elif self.remixer.guidance == 'manifold+generator':
            fold_gen_claps = self.generator.fold2claps(src, var_samples=var_samples)
            src = torch.cat([src, fold_gen_claps], dim=-1)
        clap = self.remixer.sample(src, clips, var_samples=var_samples, normalize=normalize)
        return clap

    def tpairs2clap(self, pairs, var_samples=64, normalize=False):
        clips = tpairs2tclips(pairs, max_length=self.remixer.slot, device=self.device)
        clap = self.clips2clap(clips, var_samples, normalize)
        return clap


# in this application we recycle models to save memory, the intermediate products are saved to disk under data_cache
@torch.no_grad()
def image_to_audio(images, text="", transcription="", save_dir="", config=None,
                   gen_remix=True, gen_tracks=False, emb_only=False,
                   pretrained=None, batch_size=64, var_samples=1,
                   shuffle_remix=True, cycle_its=3, cycle_samples=16, keep_data_cache=False,
                   duration=10, seed=42, device='cuda'):
    set_seed(seed)
    # revert to default model config if not supplied
    if not os.path.exists(config):
        config = Path().resolve() / 'configs' / 'model.json'
    with open(config, 'r') as fp:
        config = json.load(fp)

    if not save_dir:
        save_dir = Path().resolve() / 'output'  # default saving folder
    else:
        save_dir = Path(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if gen_tracks:
            os.makedirs(save_dir / 'tracks')
    cache_dir = save_dir / 'data_cache'

    # segmentation proposal
    if not isinstance(images, dict):
        local_imgs = detect(images, config['detector'],
                            save_dir=cache_dir / 'masked_images', batch_size=batch_size, device=device)
    else:
        local_imgs = copy.deepcopy(images)
        images = [k for k in images]
        keep_data_cache = True  # prevent deleting nonexistent folder

    # clip embed
    global_clips = clip_embed_images(images, batch_size=batch_size, device=device)
    imgs = []
    for img in images:
        imgs += [li for li, _ in local_imgs[img]]
    local_clips = clip_embed_images(imgs, batch_size=batch_size, device=device)

    jumps = [len(local_imgs[img]) for img in local_imgs]

    # SDV2A
    model = Pipeline(copy.deepcopy(config), pretrained, device)
    model.eval()
    with torch.no_grad():
        # clips to claps
        local_claps = model.clips2foldclaps(local_clips, var_samples=var_samples)

        if gen_remix:
            # remix
            remix_clips = emb2seq(jumps, local_clips, max_length=model.remixer.slot, delay=1, device=model.device)
            remix_clips[:, 0, :] = global_clips  # blend in global clip
            remix_clap = model.cycle_mix(remix_clips, its=cycle_its, var_samples=var_samples,
                                         samples=cycle_samples, shuffle=shuffle_remix)

            del remix_clips

    if emb_only:
        if not keep_data_cache:
            rmtree(cache_dir)
        return remix_clap.detach().cpu().numpy()

    # clean up gpu
    # del global_clips, local_clips, remix_clips
    del local_clips

    audioldm_v = config['audioldm_version']
    # AudioLDM
    model = build_audioldm(model_name=audioldm_v, device=device)
    if gen_tracks:
        local_wave = emb_to_audio(model, local_claps, batchsize=batch_size, duration=duration())
    if gen_remix:
        waveform = emb_to_audio(model, remix_clap, batchsize=batch_size, duration=duration)

    # I/O
    if gen_tracks:
        local_names = [Path(img).name.replace('.png', '') for img in imgs]
        save_wave(local_wave, save_dir / 'tracks', name=local_names)
    if gen_remix:
        save_wave(waveform, save_dir,
                  name=[os.path.basename(img).replace('.png', '') for img in images])
    if not keep_data_cache:
        rmtree(cache_dir)


@torch.no_grad()
def video_to_claps(config, pretrained, videos, save_dir, frames=64, batch_size=256, var_samples=64,
                   shuffle_remix=True, cycle_its=4, cycle_samples=64, seed=42, device='cuda'):
    cache_dir = Path(save_dir) / 'cache'

    print('Extracting frames and generate high-level audios:')
    result_claps = []
    for s in tqdm(range(0, len(videos), batch_size)):
        # extract frames
        os.makedirs(cache_dir, exist_ok=True)
        e = min(len(videos), s + batch_size)
        batch_extract_frames(videos[s:e], cache_dir, size=(512, 512), frames=frames, num_workers=8)

        # get generated claps
        imgs = [str(p) for p in cache_dir.glob('*.png')]
        gen_claps = image_to_audio(imgs, save_dir=str(cache_dir), config=config,
                                   gen_remix=True, gen_tracks=False, emb_only=True,
                                   pretrained=pretrained, batch_size=64, var_samples=var_samples,
                                   shuffle_remix=shuffle_remix, cycle_its=cycle_its, cycle_samples=cycle_samples,
                                   keep_data_cache=False, seed=seed, device=device)

        # map to output
        for video_f in videos[s:e]:
            vid = str(os.path.basename(video_f).replace('.mp4', ''))
            gen_clap = [None] * frames
            for i, img in enumerate(imgs):
                if vid in img:
                    img_idx = int(img[img.rfind('_') + 1:img.rfind('.')])
                    gen_clap[img_idx] = gen_claps[i]
            result_claps.append(np.stack(gen_clap))

        rmtree(cache_dir)

    gc.collect()
    torch.cuda.empty_cache()

    return np.stack(result_claps)


@torch.no_grad()
def video_to_audio(config, pretrained, videos, agg_ckpt, save_dir,
                   agg_var_samples=1, frames=64, batch_size=256,
                   var_samples=64, cycle_its=4, cycle_samples=64,
                   duration=10, seed=42, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    claps = video_to_claps(config, pretrained, videos, save_dir,
                           frames=frames, batch_size=batch_size,
                           var_samples=var_samples, shuffle_remix=True, cycle_its=cycle_its,
                           cycle_samples=cycle_samples, seed=seed, device=device)

    # Temporal Aggregation
    model = Aggregator(emb_dim=512, device=device)
    model.load_state_dict(torch.load(agg_ckpt))
    agg_claps = []
    for s in range(0, len(videos), batch_size):
        e = min(len(videos), s + batch_size)
        agg_claps.append(model.sample(torch.from_numpy(claps[s:e]), var_samples=agg_var_samples))
    agg_claps = torch.cat(agg_claps, dim=0)

    # AudioLDM
    print('Low level generation with AudioLDM:')
    with open(config, 'r') as fp:
        m_config = json.load(fp)

    audioldm_v = m_config['audioldm_version']
    # AudioLDM
    model = build_audioldm(model_name=audioldm_v, device=device)
    waveform = emb_to_audio(model, agg_claps, batchsize=batch_size, duration=duration)

    # I/O
    save_wave(waveform, save_dir, name=[os.path.basename(v).replace('.mp4', '') for v in videos])


# generate oracle audio from audioldm
@torch.no_grad()
def audio_to_audio(aud_dir, save_dir, aldm_version='audioldm-s-full-v2', batchsize=16, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    auds = [str(aud) for aud in Path(aud_dir).glob('*.wav')]
    claps = clap_embed_auds(auds, clap_version=aldm_version, device=device)
    model = build_audioldm(model_name=aldm_version, device=device)
    waveform = emb_to_audio(model, claps, batchsize=batchsize)
    fns = [os.path.basename(f).replace('*.wav', '') for f in auds]
    save_wave(waveform, save_dir, name=fns)


# generate audio from multimodal conditions
@torch.no_grad()
def srcs_to_audio(srcs, save_dir,
                  config=None, pretrained=None,
                  dalle2_cfg='', dalle2_ckpt='',
                  shuffle_remix=True, cycle_its=3, cycle_samples=16,
                  var_samples=1, batch_size=64, seed=42, duration=10, device='cuda'):
    set_seed(seed)
    with open(config, 'r') as fp:
        config = json.load(fp)

    # CLIP embeds
    img_ks = list(srcs['image'].keys())
    if img_ks:
        embs = clip_embed_images(img_ks, version='ViT-L/14', batch_size=batch_size, device=device).detach().cpu().numpy()
        for k, img_k in enumerate(img_ks):
            srcs['image'][img_k] = [None, embs[k]]

    # DALLE2 Prior embeds
    text_ks = list(srcs['text'].keys())
    if text_ks:
        embs = prior_embed_texts(text_ks, cfg=dalle2_cfg, ckpt=dalle2_ckpt, bs=batch_size,
                                 n_samples_per_batch=2, cond_scale=1, device=device)
        embs = embs.detach().cpu().numpy()
        for k, text_k in enumerate(text_ks):
            srcs['text'][text_k] = [None, embs[k]]

    # CLAP embeds
    aud_ks = list(srcs['audio'].keys())
    if aud_ks:
        embs = clap_embed_auds(aud_ks, clap_version='audioldm-s-full-v2', device=device).detach().cpu().numpy()
        for k, aud_k in enumerate(aud_ks):
            srcs['audio'][aud_k] = [None, embs[k]]

    model = Pipeline(copy.deepcopy(config), pretrained, device)
    model.eval()
    # manifold embeds
    for mod in ['image', 'text', 'audio']:
        ks = list(srcs[mod].keys())
        if ks:
            embs = np.stack([srcs[mod][k][1] for k in ks])
            for ks_s in range(0, len(ks), batch_size):
                ks_e = min(len(ks), ks_s + batch_size)
                bembs = torch.from_numpy(embs[ks_s:ks_e]).to(device)
                if mod == 'audio':
                    bembs = model.manifold.fold_claps(bembs, var_samples=var_samples, normalize=False)
                else:
                    bembs = model.manifold.fold_clips(bembs, var_samples=var_samples, normalize=False)
                bembs = bembs.detach().cpu().numpy()
                for z, k in enumerate(ks[ks_s:ks_e]):
                    srcs[mod][k][0] = bembs[z]

    # assemble remixer input
    rm_src = torch.zeros(model.remixer.slot, model.fold_dim)
    rm_clip = torch.zeros(model.remixer.slot, model.clip_dim)

    stepper = 1  # reserve first row for global condition (empty)
    for mod in srcs:
        for k in srcs[mod]:
            rm_src[stepper, ...] = torch.from_numpy(srcs[mod][k][0])
            if mod == 'audio':
                continue
            else:
                rm_clip[stepper, ...] = torch.from_numpy(srcs[mod][k][1])
            stepper += 1

    rm_src = rm_src.unsqueeze(0).float().to(device)
    rm_clip = rm_clip.unsqueeze(0).float().to(device)

    # remix!
    remix_clap = model.cycle_mix(rm_clip, fixed_src=rm_src,
                                 its=cycle_its, var_samples=var_samples,
                                 samples=cycle_samples, shuffle=shuffle_remix)

    del model, rm_src, rm_clip, embs

    # AudioLDM
    audioldm_v = config['audioldm_version']
    model = build_audioldm(model_name=audioldm_v, device=device)
    waveform = emb_to_audio(model, remix_clap, batchsize=batch_size, duration=duration)

    # I/O
    sf.write(save_dir, waveform[0, 0], samplerate=16000)

