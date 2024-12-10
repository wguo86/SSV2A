"""
Measures the F-Ratio and Partition Coefficient of the learned manifold.
"""
import random

import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA

from ssv2a.data.pairs import PairDataset, pairs2clips, pairs2claps


def get_cluster(pipeline, pairs_meta, cats, samples_per_cat=20, var_samples=1, device='cuda'):
    pairs = PairDataset(pairs_meta, split='test')

    # collect samples
    samples = {}
    for i in random.sample(range(len(pairs)), len(pairs)):
        pair, cat = pairs[i]
        if cat in cats:
            if cat not in samples:
                samples[cat] = [pair]
            elif cat in samples and len(samples[cat]) < samples_per_cat:
                samples[cat].append(pair)

    # get manifold embeddings
    raw_clips = {}
    raw_claps = {}
    fold_clips = {}
    fold_claps = {}
    gen_clips = {}
    gen_claps = {}
    pipeline.eval()
    with torch.no_grad():
        for cat in cats:
                clips = pairs2clips(samples[cat], 'local_clips', device=device)
                claps = pairs2claps(samples[cat], align_clips='local_clips', device=device)
                raw_clips[cat] = clips.detach().cpu().numpy()
                raw_claps[cat] = claps.detach().cpu().numpy()
                clip = pipeline.manifold.fold_clips(clips, var_samples=var_samples, normalize=False)
                clap = pipeline.manifold.fold_claps(claps, var_samples=var_samples, normalize=False)
                fold_clips[cat] = clip.detach().cpu().numpy()
                fold_claps[cat] = clap.detach().cpu().numpy()
                gen_clips[cat] = pipeline.generator.fold2claps(clip, var_samples=var_samples).detach().cpu().numpy()
                gen_claps[cat] = pipeline.generator.fold2claps(clap, var_samples=var_samples).detach().cpu().numpy()
    return raw_clips, raw_claps, fold_clips, fold_claps, gen_clips, gen_claps


def clusters2arr(clusters, cats):
    rt = []
    for cat in cats:
        rt.append(clusters[cat])
    return np.stack(rt)


def clusters2csv(clusters, save_dir):  # save the clustering to csv
    df = []
    for k in clusters:
        embs = clusters[k]
        for i in range(len(embs)):
            entry = {'class': k}
            if i < len(embs) // 2:
                entry['mode'] = 'visual'
            else:
                entry['mode'] = 'audio'
            emb = embs[i]
            for j in range(len(emb)):
                entry[f'dim_{j}'] = emb[j]
            df.append(entry)
    df = pd.DataFrame(df)
    df.to_csv(save_dir, index=False)
    return df


def pca_fit(pairs_meta, modality='local_clips', pipeline=None, n_components=512, device='cuda'):  # use https://github.com/valentingol/torch_pca, faster than sklearn
    pairs = PairDataset(pairs_meta, split='test')
    embs = []
    clap_embs = []
    for i in random.sample(range(len(pairs)), len(pairs)):
        pair, _ = pairs[i]
        if pipeline is None:
            embs.append(np.squeeze(pair.data[modality]))
        else:
            embs.append(np.squeeze(pair.data['local_clips']))
            clap_embs.append(np.squeeze(pair.data['clap']))

    embs = np.stack(embs)
    if pipeline is not None:
        clap_embs = np.stack(clap_embs)
        pipeline.manifold.eval()
        with torch.no_grad():
            fold_embs = []
            for s in range(0, len(embs), 64):
                e = min(len(embs), s + 64)
                fold_embs.append(
                    pipeline.manifold.fold_clips(torch.from_numpy(embs[s:e]).float().to(device),
                                                 var_samples=1, normalize=False).detach().cpu().numpy())
            fold_embs = np.concatenate(fold_embs, axis=0)

            fold_clap_embs = []
            for s in range(0, len(clap_embs), 64):
                e = min(len(clap_embs), s + 64)
                fold_clap_embs.append(
                    pipeline.manifold.fold_claps(torch.from_numpy(clap_embs).float().to(device),
                                                 var_samples=1, normalize=False).detach().cpu().numpy())
            fold_clap_embs = np.concatenate(fold_clap_embs, axis=0)
        embs = np.concatenate([fold_embs, fold_clap_embs], axis=0)

    pca = PCA(n_components=n_components, svd_solver='auto')
    pca.fit(embs)
    return pca


def pca_reduce(clusters, pca):
    cats = list(clusters.keys())
    cluster_arr = clusters2arr(clusters, cats)
    _, S, _ = cluster_arr.shape
    cluster_arr = np.concatenate(cluster_arr, axis=0)

    cluster_arr = pca.transform(cluster_arr)

    new_clusters = {}
    for i, cat in enumerate(cats):
        new_clusters[cat] = cluster_arr[i*S:i*S+S]
    return new_clusters


def get_pc(clusters):
    """
    :param clusters (K, S, E): a numpy array containing a cluster of samples
    :return: the Partition Coefficient
    """
    # find centroids
    centroids = np.mean(clusters, axis=1)  # (K, E)

    # compute membership (cosine similarity)
    member = np.einsum('kse,ke->ks', clusters, centroids)
    norms = np.linalg.norm(clusters, axis=-1) * np.expand_dims(np.linalg.norm(centroids, axis=-1), axis=-1)
    member /= norms  # (K, S)
    member = member.T / 2 + 0.5  # (S, K), rescale, don't allow negative member resulted from cosine similarity

    # compute partition coefficient
    pc = np.sum(member ** 2, axis=-1)
    pc = np.mean(pc)
    return pc

