"""
Preprocessing code is mainly based on Nvidia's FFHQ preprocessing code
https://github.com/NVlabs/ffhq-dataset/blob/bb67086731d3bd70bc58ebee243880403726197a/download_ffhq.py#L259-L349).
"""


import os, logging
import PIL
import dlib
import random
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from skimage import io

import torch


logger = logging.getLogger(__name__)


def preprocess(pretrained_face_landmark_model, in_dir, out_dir=None, out_size=256):
    """If out_dir contains a file with the same name as the output name, the
    input image is not processed.
    
    Args:
        in_dir (TYPE): Description
        out_dir (None, optional): Description
        out_size (int, optional): Description
    """
    transform_size=4096
    enable_padding=True
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(pretrained_face_landmark_model)

    torch.backends.cudnn.benchmark = False

    os.makedirs(out_dir, exist_ok=True)
    img_files = [os.path.join(path, filename) 
        for path, dirs, files in os.walk(in_dir)
        for filename in files
        if filename.endswith(".png") or filename.endswith(".jpg") 
            or filename.endswith(".jpeg")
        ]
    img_files.sort()

    cnt = 0
    preprocessed_col = []
    for img_file in tqdm(img_files):
        if out_dir is not None:
            output_img = os.path.join(out_dir, f"{cnt:08}.png")
            if os.path.isfile(output_img):
                cnt += 1
                continue
        img = dlib.load_rgb_image(img_file)
        dets = detector(img, 1)
        if len(dets) == 0:
            logger.debug("no face landmark detected")
            continue
        else:
            shape = sp(img, dets[0]) # a class that return face landmark
            points = np.empty([68, 2], dtype=int)

            # arrange landmarks in shape object into an array
            for b in range(68):
                points[b, 0] = shape.part(b).x
                points[b, 1] = shape.part(b).y
            lm = points 

        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # To process image using other more friendly library like skimage, the below code must be changed accordingly because img is not of type numpy.  
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')

        # Shrink.
        shrink = int(np.floor(qsize / out_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if out_size < transform_size:
            img = img.resize((out_size, out_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        if out_dir is not None:
            img.save(output_img)
        cnt += 1 
        preprocessed_col.append(img)
    return preprocessed_col


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True, 
        help="Directory of the output, which is assumed to be different from \
            that of the input")
    parser.add_argument("--out_size", type=int, default=256)
    args = parser.parse_args()

    pretrained_face_landmark_model =  "experiments/preprocess/shape_predictor_68_face_landmarks.dat"
    preprocess(pretrained_face_landmark_model, args.in_dir, args.out_dir, args.out_size)
