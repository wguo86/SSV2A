"""
These helper functions mainly deal with jumping between audio representations.
The wav manipulations are adapted from https://github.com/haoheliu/AudioLDM2
-- danke
"""
import os
import random
import urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import clip
import torch
import torchaudio
import skvideo.io
from PIL import Image, ImageDraw, ImageFilter
from audioldm import get_metadata
from audioldm.utils import MyProgressBar
from kneed import KneeLocator
from textblob import TextBlob
import soundfile as sf
from tqdm import tqdm

from ssv2a.model.dalle2_prior import Dalle2Prior


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# find elbow point of a given score list automatically with Kneedle algorithm
# as we always operate in batches, the sensitivity is fixed at 1 in hope of an optimal online Kneedle
def elbow(scores, sensitivity=1, curve='convex', return_idx=True):
    x = np.arange(len(scores))
    kneedle = KneeLocator(x, scores, S=sensitivity, curve=curve, direction='decreasing')
    if return_idx:
        if kneedle.knee is None:
            return len(x)
        return min(len(x), round(kneedle.knee))
    else:
        if kneedle.knee_y is None:
            return x[-1]
        else:
            return kneedle.knee_y


def random_mute(a, p=.5):  # randomly mute a tensor with probability, used for classifier free guidance
    """
    a: [*, d1, d2]
    """
    d1 = a.shape[-2]
    idx = torch.rand(d1) < p
    rt = a.clone()
    rt[..., idx, :] = 0
    return rt, idx


def get_noun_phrases(text):
    blob = TextBlob(text)
    return list(blob.noun_phrases)


def read_classes(classes_file):
    with open(classes_file, 'r') as fp:
        lines = fp.readlines()
    classes = list([c.strip('\n').lower() for c in lines if c.strip('\n').lower() != ''])
    return classes


def extract_central_frame(video, save_dir, size=None):
    videodata = skvideo.io.vread(str(video))
    cf = Image.fromarray(videodata[videodata.shape[0] // 2 + videodata.shape[0] % 2, :, :, :])
    if size is not None:
        cf = cf.resize(size, resample=Image.Resampling.BICUBIC)
    cf.save(Path(save_dir) / Path(video).name.replace('.mp4', '.png'), 'PNG')


# given a list of video files, batch extract the central frames and save to a directory
def batch_extract_central_frame(video_fs, save_dir, size=None, bs=32, num_workers=8):
    os.makedirs(save_dir, exist_ok=True)
    for s in tqdm(range(0, len(video_fs), bs)):
        e = min(len(video_fs), s + bs)
        pool = ThreadPoolExecutor(max_workers=num_workers)
        for f in video_fs[s:e]:
            pool.submit(extract_central_frame, f, save_dir, size=size)
        pool.shutdown(wait=True)

# evenly extract n frames from a given video
def extract_frames(video_fs, save_dir, size=None, frames=64):
    for vid in tqdm(video_fs):
        videodata = skvideo.io.vread(str(vid))
        L = videodata.shape[0]
        for i, f in enumerate(np.round(np.linspace(0, L - 1, frames)).astype(int).tolist()):
            cf = Image.fromarray(videodata[f, :, :, :])
            if size is not None:
                cf = cf.resize(size, resample=Image.Resampling.BICUBIC)
            cf.save(Path(save_dir) / Path(vid).name.replace('.mp4', f'_{i}.png'), 'PNG')

# given a list of video files, batch extract the central frames and save to a directory
def batch_extract_frames(video_fs, save_dir, size=None, frames=64, num_workers=8):
    os.makedirs(save_dir, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=num_workers)
    if len(video_fs) >= num_workers:
        workload = len(video_fs) // num_workers
        for s in range(0, len(video_fs), workload):
            e = min(len(video_fs), s + workload)
            pool.submit(extract_frames, video_fs[s:e], save_dir, size, frames)
        pool.shutdown(wait=True)
    else:
        extract_frames(video_fs, save_dir, size, frames)


def get_fps(s):
    if s.isdigit():
        return float(s)
    else:
        num, denom = s.split('/')
        return float(num) / float(denom)


def video2images(video, fps=4):  # video to image sequence, sampled by fps
    try:
        video_name = os.path.basename(video).replace('.mp4', '')
        videodata = skvideo.io.vread(video)
        videometadata = skvideo.io.ffprobe(video)
        frame_rate = videometadata['video']['@avg_frame_rate']
        frame_num = videodata.shape[0]
        frames_in_sec = get_fps(frame_rate)
        length_in_secs = frame_num / frames_in_sec

        return [videodata[::int(round(frames_in_sec)/fps), :, :, :], length_in_secs, frame_num, video_name]

    except Exception as e:
        return None


def clip_embed_images(images, version='ViT-L/14', batch_size=256, device='cuda'):
    with torch.no_grad():
        model, preprocess = clip.load(version, device=device)
        embeds = []
        for i in tqdm(range(0, len(images), batch_size)):
            e = min(len(images), i + batch_size)
            imgs = torch.cat([preprocess(Image.open(img)).unsqueeze(0).to(device) for img in images[i:e]])
            embs = model.encode_image(imgs)
            embeds.append(embs)
        embeds = torch.cat(embeds).float()
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)


def clip_embed_texts(texts, bs=256, version='ViT-L/14', device='cuda'):
    with torch.no_grad():
        model, preprocess = clip.load(version, device=device)
        embeds = torch.empty(len(texts), 512)
        for s in tqdm(range(0, len(texts), bs)):
            e = min(len(texts), s + bs)
            ts = clip.tokenize(texts[s:e]).to(device)
            ts = model.encode_text(ts, normalize=True)
            embeds[s:e, :] = ts.detach().cpu()
        embeds = embeds.float()
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)


def prior_embed_texts(texts, cfg, ckpt, bs=64, n_samples_per_batch=2, cond_scale=1, device='cuda'):
    with torch.no_grad():
        model = Dalle2Prior(cfg, ckpt, device=device)

        prior_clips = torch.empty(len(texts), 768)
        for s in tqdm(range(0, len(texts), bs)):
            e = min(len(texts), s + bs)
            prior_clips[s:e, :] = model.sample(texts[s:e], n_samples_per_batch=n_samples_per_batch, cond_scale=cond_scale).detach().cpu()
        return prior_clips


# given a data length dictionary and its flattened clip embeds,
# unflatten it to batched sequences with sampling and padding, delay specifies the placeholder tokens in first k rows
def emb2seq(jumps, emb, max_length=0, delay=0, device='cuda'):
    step = 0
    rt_emb = []
    for j in jumps:
        rt_emb.append(emb[step:step+j])
        step += j

    if max_length == 0:
        return rt_emb

    for i, e in enumerate(rt_emb):
        L, E = e.shape
        if L > max_length:
            idx = random.sample(range(L), max_length - delay)
            rt_emb[i] = e[idx]
        else:
            rt_emb[i] = torch.cat([e, torch.zeros(max_length - L - delay, E).to(device)])
        rt_emb[i] = torch.cat([torch.zeros(delay, E).to(device), rt_emb[i]])
    return torch.stack(rt_emb)


# given two batches of embeddings, find the top-k similar ids in emb2 for ech entry of emb1
def topk_sim(emb1, emb2, topk=10, bs=512, normalize=False, device='cuda'):
    with torch.no_grad():
        emb1 = emb1.to(device)
        emb2 = emb2.to(device)
        if normalize:
            emb1 = emb1 / emb1.norm(p=2, dim=-1, keepdim=True)
            emb2 = emb2 / emb2.norm(p=2, dim=-1, keepdim=True)

        B = emb1.shape[0]
        topk_sims = torch.ones(B, topk).to(device) * -2
        topk_idx = torch.zeros(B, topk, dtype=torch.int64).to(device)

        for s in range(0, len(emb2), bs):
            e = min(len(emb2), s + bs)
            sims = torch.einsum('ai,bi->ab', emb1, emb2[s:e])
            sims, idx = torch.topk(sims, k=topk, dim=-1)
            idx += s
            topk_sims = torch.cat([topk_sims, sims], dim=-1)
            topk_idx = torch.cat([topk_idx, idx], dim=-1)
            topk_sims, idx = torch.topk(topk_sims, k=topk, dim=-1)
            topk_idx = topk_idx.gather(-1, idx)

        return topk_sims, topk_idx


def mask2bbox(mask, normalize=False):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if normalize:
        return cmin / mask.shape[0], rmin / mask.shape[1], cmax / mask.shape[0], rmax / mask.shape[1]
    return cmin, rmin, cmax, rmax


# blur the image content with ellipses filling the bounding boxes
def blur_image_bbox(img, bboxes, blur_radius=15, mask_blur_radius=15):
    # ellipse masks
    mask = Image.new('L', img.size, color=0)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        draw.ellipse(bbox, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))
    # blur and overlay with masks
    overlay = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(overlay, img, mask)


# crop a PIL image with a normalized [x1, y1, x2, y2] bbox
def crop_image_bbox(img, bbox, keep_size=False):
    width, height = img.size
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    x1, y1, x2, y2 = min(1, x1), min(1, y1), min(1, x2), min(1, y2)
    oimg = img.crop([round(min(x1, x2) * width), round(min(y1, y2) * height),
                     round(max(x1, x2) * width), round(max(y1, y2) * height)])
    if keep_size:  # keep original size by upsampling
        oimg = oimg.resize((width, height), Image.Resampling.BICUBIC)
    return oimg


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:, :segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
        return temp_wav


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def read_wav_file(filename, segment_length=0, sampling_rate=16000):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sampling_rate)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    if segment_length != 0:
        waveform = pad_wav(waveform, segment_length)

    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform

    waveform = np.nan_to_num(waveform)

    return waveform


def extract_kaldi_fbank_feature(waveform, sampling_rate):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    target_len = waveform.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = target_len - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:target_len, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]


def save_wave(waveform, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1 :
            fname = "%s_%s.wav" % (
                    os.path.basename(name[i])
                    if (not ".wav" in name[i])
                    else os.path.basename(name[i]).split(".")[0],
                    i,
                )
        else:
            fname = "%s.wav" % os.path.basename(name[i]) if (not ".wav" in name[i]) else os.path.basename(name[i]).split(".")[0]
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        path = os.path.join(
            savepath, fname
        )
        # print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=samplerate)


def download_audioldm_checkpoint(checkpoint_name):
    meta = get_metadata()
    if(checkpoint_name not in meta.keys()):
        print("The model name you provided is not supported. Please use one of the following: ", meta.keys())

    if not os.path.exists(meta[checkpoint_name]["path"]) or os.path.getsize(meta[checkpoint_name]["path"]) < 2*10**9:
        os.makedirs(os.path.dirname(meta[checkpoint_name]["path"]), exist_ok=True)
        print(f"Downloading the main structure of {checkpoint_name} into {os.path.dirname(meta[checkpoint_name]['path'])}")

        urllib.request.urlretrieve(meta[checkpoint_name]["url"], meta[checkpoint_name]["path"], MyProgressBar())
        print(
            "Weights downloaded in: {} Size: {}".format(
                meta[checkpoint_name]["path"],
                os.path.getsize(meta[checkpoint_name]["path"]),
            )
        )

    return meta[checkpoint_name]["path"]

def image2video(img_fs, out_dir, duration=10, fps=24):  # pad an image to a video
    for img_f in tqdm(img_fs):
        img = np.asarray(Image.open(img_f))[None, ...]
        img = np.repeat(img, duration * fps, axis=0)
        writer = skvideo.io.FFmpegWriter(os.path.join(out_dir, os.path.basename(img_f).replace('.png', '.mp4')))
        for i in range(img.shape[0]):
            writer.writeFrame(img[i, ...])
        writer.close()

def batch_image2video(img_fs, out_dir, duration=10, fps=24, num_workers=16):
    os.makedirs(out_dir, exist_ok=True)

    worker_fs = [[] for _ in range(num_workers)]
    for i, j in enumerate(img_fs):
        worker_fs[i % num_workers].append(j)

    pool = ThreadPoolExecutor(max_workers=num_workers)
    for i in range(num_workers):
        pool.submit(image2video, worker_fs[i], out_dir, duration, fps)
    pool.shutdown(wait=True)

