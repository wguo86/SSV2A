import numpy as np
import torch
import torchaudio
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2 as CLAP_ALD
from tqdm.auto import tqdm

from ssv2a.data.utils import download_audioldm_checkpoint, normalize_wav


# a wrapper to build the clap model (AudioLDM2 flavor)
class CLAP:
    def __init__(self, clap_version='audioldm2-full', embed_mode='audio', sample_rate=16000, device='cuda'):
        self.model = None
        ckpt_path = download_audioldm_checkpoint(clap_version)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model = CLAP_ALD(
            key="waveform",
            sampling_rate=sample_rate,
            embed_mode=embed_mode,
            unconditional_prob=0
        )
        clap_ckpt = {}
        for k, v in ckpt["state_dict"].items():
            if k.split('.')[0] == 'cond_stage_model':
                clap_ckpt[k.split('cond_stage_model.')[-1]] = v
        self.model.load_state_dict(clap_ckpt)
        self.model.eval()
        self.model.to(device)


def clap_embed_texts(texts, version='audioldm-s-full-v2', bs=256, device='cuda'):
    clap = CLAP(clap_version=version, embed_mode='text', device=device)
    embeds = torch.zeros((len(texts) + 1, 512))
    for s in tqdm(range(0, len(texts), bs)):
        e = min(len(texts), s + bs)
        emb = clap.model(texts[s:e]).squeeze().float().detach().cpu()
        embeds[s+1:e+1] = emb
    embeds = torch.nan_to_num(embeds)
    return embeds


def clap_embed_auds(auds, clap_version='audioldm-s-full-v2', device='cuda'):
    with torch.no_grad():
        clap = CLAP(clap_version=clap_version, embed_mode='audio', device=device)

        embeds = torch.empty(len(auds), 512)
        for i, aud in enumerate(auds):
            waveform, sr = torchaudio.load(aud)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            waveform = waveform.numpy()[0, ...]
            waveform = normalize_wav(waveform)
            waveform = waveform[None, ...]
            waveform = waveform / np.max(np.abs(waveform))
            waveform = np.nan_to_num(0.5 * waveform)

            embeds[i, :] = clap.model(torch.from_numpy(waveform).float().to(device)).detach().cpu().squeeze()

        return embeds

