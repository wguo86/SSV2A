import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from PIL import Image
from tqdm.auto import tqdm

# please link to the CaR modules from https://github.com/google-research/google-research/tree/master/clip_as_rnn
from ssv2a.data.utils import read_classes, video2images, mask2bbox, elbow


# detect and segment images, return all or top k segment masks > conf, optionally, save masked images to disk
# default to cropping instead of segmentation if crop=True
def yolo_detect(images, detection_model='yolov8x-worldv2.pt', segment_model="sam_b.pt", resize=None, crop=True,
                classes=None, batch_size=64, conf=.5, iou=0.5, max_det=64, top_k=None, save_dir="", device='cuda', **_):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = YOLO(detection_model)
    model.to(device)
    if 'world' in detection_model and classes is not None:
        classes = read_classes(classes)
        model.set_classes(classes)

    # automatically determine image size, assuming all images are the same size as image[0]
    if resize is not None:
        imgsz = resize
    else:
        sample_img = Image.open(images[0])
        imgsz = sample_img.size
    img_area = imgsz[0] * imgsz[1]

    print(f"Detecting objects with {detection_model}:")
    segments = {}
    for img in images:
        segments[img] = []

    for i in tqdm(range(0, len(images), batch_size)):
        e = min(len(images), i + batch_size)
        for img in images[i:e]:
            oimg = Image.open(img)
            if resize is not None and oimg.size != resize:
                oimg = oimg.resize(resize, resample=Image.Resampling.BICUBIC)
                oimg.save(img, 'PNG')
        detect_results = model.predict(images[i:e], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                       augment=True, verbose=False)

        if crop:
            # print("Cropping objects:")
            for j, img in enumerate(images[i:e]):
                oimg = Image.open(img)
                rs = detect_results[j][:top_k]
                for z, r in enumerate(rs):
                    box = r.boxes.xyxy.cpu().tolist()[0]
                    cimg = oimg.crop(box).resize(imgsz, Image.Resampling.BICUBIC)
                    cimg_file = Path(save_dir) / Path(images[i:e][j]).name.replace('.png', f'_{z}.png')
                    cimg.save(cimg_file, 'PNG')
                    locality = abs(box[2] - box[0]) * abs(box[3] - box[1]) / img_area  # locality ratio
                    segments[img].append((str(cimg_file), locality))

        else:
            # print(f"Segmenting objects with {segment_model}:")
            overrides = dict(conf=.25, retina_masks=True, task="segment", mode="predict",
                             imgsz=imgsz, model=segment_model, save=False, verbose=False, device=device)
            model = SAMPredictor(overrides=overrides)
            for j in range(len(images[i:e])):
                model.set_image(images[i:e][j])
                img = np.array(Image.open(images[i:e][j]))
                rs = detect_results[j][:top_k]
                for z, r in enumerate(rs):
                    mask = model(bboxes=r.boxes.xyxy)[0].masks.data.cpu().numpy()
                    mask = np.squeeze(mask, axis=0).astype(int)
                    mimg_file = Path(save_dir) / Path(images[i:e][j]).name.replace('.png', f'_{z}.png')
                    Image.fromarray((img * np.expand_dims(mask, axis=2)).astype(np.uint8)).save(mimg_file, 'PNG')
                    locality = float(np.sum(mask.astype(int))) / img_area
                    segments[images[i]].append((str(mimg_file), locality))
                model.reset_image()

    return segments


# filter a list of single-source or multi-source videos for true positive visual frames
def yolo_detect_videos(videos, signatures, save_dir, imgsz=(512, 512), fps=4, conf=.5,
                       detection_model='yolov8x-worldv2.pt',
                       classes=None, batchsize=64, device='cuda'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    positives = dict([(vid, []) for vid in videos])

    model = YOLO(detection_model)
    if 'world' in detection_model and classes is not None:
        model.set_classes(read_classes(classes))
    signatures = set(signatures)

    for s in tqdm(range(0, len(videos), batchsize)):
        e = min(s + batchsize, len(videos))
        for vid in videos[s:e]:
            frames, _, _, video_name = video2images(vid, fps=fps)
            frames = [Image.fromarray(frames[i]).resize(imgsz, Image.Resampling.BICUBIC) for i in range(len(frames))]

            rs = model.predict(frames, conf=conf, imgsz=imgsz, augment=True, verbose=False, device=device)
            for j, r in enumerate(rs):
                cls = set([model.names[c] for c in r.boxes.cls.cpu().tolist()])
                if cls == signatures:  # bingo
                    fimg_file = Path(save_dir) / f'{video_name}_{j}.png'
                    frames[j].save(fimg_file, 'PNG')
                    positives[vid].append(fimg_file)

    # remove video if it doesn't contain any qualified visual frames
    false_pos = []
    for pos in positives:
        if len(positives[pos]) == 0:
            false_pos.append(pos)
    for pos in false_pos:
        del positives[pos]

    return positives


def detect(images, detector_cfg, save_dir='masked_images', batch_size=64, device='cuda'):
    detector_cfg['save_dir'] = save_dir
    detector_cfg['batch_size'] = batch_size
    detector_cfg['device'] = device

    if 'yolo' in detector_cfg['detection_model']:
        return yolo_detect(images, **detector_cfg)

    else:
        raise NotImplementedError('Detection model is unsupported.')

