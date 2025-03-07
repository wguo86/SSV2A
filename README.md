# Gotta Hear Them All: Sound Source-Aware Vision to Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2411.15447-brightgreen?logo=arxiv&logoColor=white&style=flat-square)](https://arxiv.org/abs/2411.15447) [![githubio](https://img.shields.io/badge/GitHub.io-Demo_Website-blue?logo=Github&logoColor=white&style=flat-square)](https://ssv2a.github.io/SSV2A-demo/) [![Hugging Face Spaces](https://img.shields.io/badge/Gradio-Interactive_Demo-orange?logo=huggingface&logoColor=white&style=flat-square)](https://ssv2a.ngrok.io/) 

**Flexibly generate sounds by composing visual, text, and audio sound source prompts.**

In order to run our code, please clone the repository and follow these instructions to set up a virtual environment:

1. `conda create -n SSV2A python==3.10`
2. `pip install -r requirements.txt`

The `ssv2a` module provides implementations for SSV2A. We also provide scripts for major functions below.

## TODO
- [ ] Distribute the VGG Sound Single Source (VGGS3) dataset.
- [ ] Upload code for training.
- [ ] Upload code for multimodal inference.
- [x] Upload code for vision-to-audio inference. (2024-12-10)

## Pretrained Weights
We provide pretrained weights of SSV2A modules at [this google drive link](https://drive.google.com/drive/folders/17SAuZ2sZrTYf21BiNKhRsEfdj-fbeQQN?usp=sharing), 
which has the following contents:

| Files      | Comment                                                                              |
|------------|--------------------------------------------------------------------------------------|
| ssv2a.json | Configuration File of SSV2A                                                          |
| ssv2a.pth  | Pretrained Checkpoint of SSV2A                                                       |
| agg.pth    | Pretrained Checkpoint of Temporal Aggregation Module (for video-to-audio generation) |

Please download them according to your usage cases.

As SSV2A works with [YOLOv8](https://docs.ultralytics.com/models/yolov8/) for visual sound source detection, 
it also needs to include a pretrained YOLO checkpoint for inference. We recommend using [yolov8x-oi7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) 
pretrained on the OpenImagesV7 dataset. After downloading this model, paste its path in the `"detection-model"` field in `ssv2a.json`.

## Inference
There are several hyperparameters you can adjust to control the generation fidelity/diversity/relevance. We list them here:

| Parameter         | Default Value | Comment                                                                                                                       |
|-------------------|----|-------------------------------------------------------------------------------------------------------------------------------|
| `--var_samples`   | 64 | Number of variational samples drawn in each generation and averaged. Higher number increases fidelity and decreases diversity. |
| `--cycle_its`     | 64 | Number of Cycle Mix iterations. Higher number increases generation relevance to given conditions.                             |
| `--cycle_samples` | 64 | Number of variational samples drawn in each Cycle Mix iteration. Higher number increases fidelity and decreases diversity.    |
| `--duration` | 10 | Length of generated audio in seconds.                                                                                         |
| `--seed`          | 42 | Random seed for generation.                                                                                                   |

### Image to Audio Generation
Navigate to the root directory of this repo and execute the following script:

```shell
python infer_i2a.py \
--cfg "ssv2a.json" \
--ckpt "ssv2a.pth" \
--image_dir "./images" \
--out_dir "./output"
```
Replace the arguments with the actual path names on your machine.

### Video to Audio Generation
Navigate to the root directory of this repo and execute the following script:

```shell
python infer_v2a.py \
--cfg "ssv2a.json" \
--ckpt "ssv2a.pth" \
--agg_ckpt "agg.pth" \
--image_dir "/images" \
--out_dir "./output"
```
Replace the arguments with the actual path names on your machine.


## Cite this work
If you find our work useful, please consider citing

```bibtex
@article{SSV2A,
  title={Gotta Hear Them All: Sound Source Aware Vision to Audio Generation},
  author={Guo, Wei and Wang, Heng and Ma, Jianbo and Cai, Weidong},
  journal={arXiv preprint arXiv:2411.15447},
  year={2024}
}
```

## References
We list major references in our code here:

1. [AudioLDM](https://github.com/haoheliu/AudioLDM), by Haohe Liu
2. [AudioLDM2](https://github.com/haoheliu/AudioLDM2), by Haohe Liu
3. [LAION-Audio-630K](https://github.com/LAION-AI/audio-dataset), by LAION
3. [frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance), by Haohao Tan
4. [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch), by Phil Wang

Thank you for the excellent works! Other references are commented inline.

