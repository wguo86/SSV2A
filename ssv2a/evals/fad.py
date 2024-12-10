from frechet_audio_distance import FrechetAudioDistance


# use https://github.com/gudgud96/frechet-audio-distance
def get_fad(pred, target, sr=16000, model='pann', device='cuda'):
    if model == 'pann' or model == 'vggish':
        frechet = FrechetAudioDistance(model_name=model, sample_rate=sr,
                                       use_pca=False, use_activation=False, verbose=False)
    elif model == 'clap':
        frechet = FrechetAudioDistance(
            model_name="clap",
            sample_rate=48000,
            submodel_name="630k-audioset",  # for CLAP only
            verbose=False,
            enable_fusion=False  # for CLAP only
        )
    elif model == 'encodec':
        frechet = FrechetAudioDistance(
            model_name="encodec",
            sample_rate=48000,
            channels=2,
            verbose=False,
        )
    else:
        raise NotImplementedError('Model is not supported.')
    score = frechet.score(background_dir=target, eval_dir=pred)
    return score

