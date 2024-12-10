"""
Reimplementation of AudioLDM and AudioLDM2 to accept CLAP embeddings instead of texts as condition.
Adapted from https://github.com/haoheliu/AudioLDM2
-- danke
"""
import os

import numpy as np
import torch
import yaml
from transformers import logging

from audioldm import get_metadata, download_checkpoint, default_audioldm_config, LatentDiffusion, seed_everything, \
    read_wav_file, duration_to_latent_t_size, set_cond_audio
from audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution


def ddpm_get_input(batch, k):
    fbank, log_magnitudes_stft, label_indices, fname, waveform, text, image, emb = batch
    ret = {}

    ret["fbank"] = (
        fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
    )
    ret["stft"] = log_magnitudes_stft.to(
        memory_format=torch.contiguous_format
    ).float()
    # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
    ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
    ret["text"] = list(text)
    ret['image'] = list(image)
    ret["fname"] = fname
    ret['emb'] = emb
    return ret[k]


class EmbAudioLDM(LatentDiffusion):
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:  # true
            if hasattr(self.cond_stage_model, "encode") and callable(
                    self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:  # true
                if self.cond_stage_key == 'emb':
                    if len(c.shape) == 2:
                        c = c[:, None, :]
                else:
                    # Text input is list
                    if type(c) == list and len(c) == 1:  # true
                        c = self.cond_stage_model([c[0], c[0]])  # clap/encoders.py
                        c = c[0:1]  # [1, 1, 512])
                    else:
                        c = self.cond_stage_model(c)  # torch.Size([1, 1, 512]) torch.cuda.FloatTensor float32
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_input(
            self,
            batch,
            k,
            return_first_stage_encode=True,
            return_first_stage_outputs=False,
            force_c_encode=False,
            cond_key=None,
            return_original_cond=False,
            bs=None,
    ):

        x = ddpm_get_input(batch, k)

        if bs is not None:  # false
            x = x[:bs]

        x = x.to(self.device)

        if return_first_stage_encode:  # true
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()  # torch.Size([10, 8, 256, 16])
        else:
            z = None

        if self.model.conditioning_key is not None:  # film
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:  # true
                if cond_key in ["caption", "coordinates_bbox"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    # [bs, 1, 527]
                    xc = ddpm_get_input(batch, cond_key)  # 10,512
                    if type(xc) == torch.Tensor:  # false
                        xc = xc.to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:  # true, true
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))  # 10,1,512
            else:
                c = xc

            if bs is not None:  # false
                c = c[:bs]

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def generate_sample(
            self,
            batchs,
            ddim_steps=200,
            ddim_eta=1.0,
            x_T=None,
            n_candidate_gen_per_text=1,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            name="waveform",
            use_plms=False,
            save=False,
            **kwargs,
    ):
        # Generate n_candidate_gen_per_text times and select the best
        # Batch: audio, text, fnames
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None
        use_ddim = ddim_steps is not None
        # waveform_save_path = os.path.join(self.get_log_dir(), name)
        # os.makedirs(waveform_save_path, exist_ok=True)
        # print("Waveform save path: ", waveform_save_path)

        with self.ema_scope("Generate"):
            waves = []
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    cond_key=self.cond_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = ddpm_get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_candidate_gen_per_text
                c = torch.cat([c] * n_candidate_gen_per_text, dim=0)
                text = text * n_candidate_gen_per_text

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )

                if (torch.max(torch.abs(samples)) > 1e2):
                    samples = torch.clip(samples, min=-10, max=10)

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(mel)

                if waveform.shape[0] > 1:
                    similarity = self.cond_stage_model.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )

                    best_index = []
                    for i in range(z.shape[0]):
                        candidates = similarity[i:: z.shape[0]]
                        max_index = torch.argmax(candidates).item()
                        best_index.append(i + max_index * z.shape[0])

                    waveform = waveform[best_index]
                    # print("Similarity between generated audio and text", similarity)
                    # print("Choose the following indexes:", best_index)

                    waves.append(waveform)

        return np.concatenate(waves)


def make_batch_for_emb_to_audio(emb, waveform=None, fbank=None, batchsize=1):
    batches = []
    B = emb.shape[0]

    for s in range(0, B, batchsize):
        e = min(B, s + batchsize)
        bs = e - s

        text = [''] * bs
        image = [''] * bs

        if bs < 1:
            print("Warning: Batchsize must be at least 1. Batchsize is set to .")

        if fbank is None:  # true
            fb = torch.zeros((bs, 1024, 64))  # Not used, here to keep the code format
        else:
            fb = torch.FloatTensor(fbank[s:e])
            fb = fb.expand(bs, 1024, 64)
            assert fb.size(0) == bs

        stft = torch.zeros((bs, 1024, 512))  # Not used

        if waveform is None:
            wave = torch.zeros((bs, 160000))  # Not used 16kHz*10s
        else:
            wave = torch.FloatTensor(waveform[s:e])
            wave = wave.expand(bs, -1)
            assert wave.size(0) == bs

        fname = [""] * bs  # Not used

        batch = (
            fb,
            stft,
            None,
            fname,
            wave,
            text,
            image,
            emb[s:e]
        )
        batches.append(batch)

    return batches


def build_audioldm(  # only model_name
        ckpt_path=None,
        config=None,
        model_name="audioldm-s-full",
        device=None
):
    # print(f"Load AudioLDM: {model_name}")

    if (ckpt_path is None):
        ckpt_path = get_metadata()[model_name]["path"]

    if (not os.path.exists(ckpt_path)):
        download_checkpoint(model_name)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = EmbAudioLDM(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"
    return latent_diffusion


def set_cond_emb(latent_diffusion):
    latent_diffusion.cond_stage_key = "emb"
    latent_diffusion.cond_stage_model.embed_mode = None
    return latent_diffusion


def emb_to_audio(
        latent_diffusion,
        emb,
        original_audio_file_path=None,
        seed=42,
        ddim_steps=200,
        duration=10,
        batchsize=1,
        guidance_scale=2.5,
        n_candidate_gen_per_text=3
):
    seed_everything(int(seed))
    logging.set_verbosity_error()
    waveform = None
    if (original_audio_file_path is not None):
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)

    batchs = make_batch_for_emb_to_audio(emb, waveform=waveform, batchsize=batchsize)

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)

    if (waveform is not None):
        print("Generate audio that has similar content as %s" % original_audio_file_path)
        latent_diffusion = set_cond_audio(latent_diffusion)
    else:
        # print("Generate audio using embedding", emb.shape)
        latent_diffusion = set_cond_emb(latent_diffusion)

    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            batchs,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform

