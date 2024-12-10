import torch

import clip
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig


class Dalle2Prior:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        prior_config = TrainDiffusionPriorConfig.from_json_path(config_path).prior
        self.prior = prior_config.create().to(device)
        self.device = device

        states = torch.load(ckpt_path)
        if 'model' in states:
            states = states['model']

        self.prior.load_state_dict(states, strict=True)

    def sample(self, texts, n_samples_per_batch=2, cond_scale=1):
        texts = clip.tokenize(texts, truncate=True).to(self.device)
        clips = self.prior.sample(texts, num_samples_per_batch=n_samples_per_batch, cond_scale=cond_scale)
        return clips / clips.norm(p=2, dim=-1, keepdim=True)

