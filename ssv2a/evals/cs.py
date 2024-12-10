import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wav2clip
import clip

from ssv2a.data.utils import pad_wav


class WaveImageDataset(Dataset):
    def __init__(self, aud_imgs, sr=16000, duration=10, limit_num=None):
        self.data = aud_imgs
        self.data_idx = sorted(list(self.data.keys()))
        if limit_num is not None:
            self.data_idx = self.data_idx[:limit_num]
        self.sr = sr
        self.seg_len = int(duration * 102.4) * (sr // 100)

    def __getitem__(self, index):
        while True:
            try:
                filename = self.data_idx[index]
                waveform = self.read_from_file(filename)
                if waveform.shape[-1] < 1:
                    raise ValueError("empty file %s" % filename)
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.data_idx)
        return waveform, self.data[filename], os.path.basename(filename)

    def __len__(self):
        return len(self.data_idx)

    def read_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # Only use the first channel
        audio = audio[0:1, ...]
        audio = audio - audio.mean()

        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=file_sr, new_freq=self.sr,  # rolloff=0.95, lowpass_filter_width=16
            )
            # audio = torch.FloatTensor(librosa.resample(audio.numpy(), file_sr, self.sr))

        audio = pad_wav(audio.numpy(), self.seg_len)
        return audio


def collate_waveimage(data):
    return np.concatenate([d[0] for d in data], dtype=np.float32), [d[1] for d in data], [d[2] for d in data]


# clip score between image and audio, image can be multiple patches
def get_cs(img_d, aud_d, sr=16000, duration=10, batch_size=64, device='cuda'):
    with torch.no_grad():
        model = wav2clip.get_model().to(device)
        clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
        auds = [str(i) for i in Path(aud_d).rglob('*.wav')]
        aud_imgs = {}
        preimg = [str(i) for i in Path(img_d).rglob('*.png')]
        for p in auds:
            pat = Path(p).name.replace('.wav', '')
            aud_imgs[p] = [i for i in preimg if pat in Path(i).name]

        loader = DataLoader(
            WaveImageDataset(aud_imgs, sr=sr, duration=duration),
            batch_size=batch_size,
            sampler=None,
            num_workers=0,
            collate_fn=collate_waveimage
        )

        ret_score = None
        total_score = 0
        n = 0
        for audio, imgs, audio_path in tqdm(loader):
            embedding = torch.from_numpy(wav2clip.embed_audio(audio, model))
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

            # embed images
            jumps = [len(img) for img in imgs]
            pimgs = []
            for img in imgs:
                pimgs += img
            pimgs = torch.cat([clip_preprocess(Image.open(img)).unsqueeze(0).to(device) for img in pimgs])
            img_emb = clip_model.encode_image(pimgs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

            # scoring (cosine similarity)
            idx = 0
            for i, j in enumerate(jumps):
                ae = embedding[i].detach().cpu().float()
                ie = img_emb[idx:idx+j].detach().cpu().float()
                sims = torch.einsum('i,bi->b', ae, ie)
                total_score += torch.mean(sims).numpy()
                idx += j
                n += 1

        return total_score / n

