import os
import argparse
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt

from emosaic.utils.indexing import index_images
from emosaic.utils.normalize import normalize_images
from emosaic import mosaicify

"""
Example usage:

    $ python mosaic.py \
        --target "media/example/beach.jpg" \
        --savepath "media/output/%s-mosaic-scale-%d.jpg" \
        --codebook-dir media/pics/ \
        --scale 12 \
        --height-aspect 4 \
        --width-aspect 3 \
        --opacity 0.0 \
        --detect-faces
"""
parser = argparse.ArgumentParser()

# required
parser.add_argument("--codebook-dir", dest='codebook_dir', type=str, required=True, help="Source folder of images")
parser.add_argument("--normalize-dir", dest='normalize_dir', type=str, required=False, help="Normalized folder of images")

parser.add_argument("--savepath", dest='savepath', type=str, required=True, help="Where to save image to. Scale/filename is used in formatting.")
parser.add_argument("--target", dest='target', type=str, required=True, help="Image to make mosaic from")
parser.add_argument("--scale", dest='scale', type=float, required=True, help="How large to make tiles")
parser.add_argument("--resize", dest='resize', type=float, required=False, default=1.0, help="How large to make target: default 1x")

# optional
parser.add_argument("--best-k", dest='best_k', type=int, default=1, help="Choose tile from top K best matches")
parser.add_argument("--no-trim", dest='no_trim', action='store_true', default=False, help="If we shouldn't trim around the outside")
parser.add_argument("--detect-faces", dest='detect_faces', action='store_true', default=False, help="If we should only include pictures with faces in them")
parser.add_argument("--opacity", dest='opacity', type=float, default=0.0, help="Opacity of the original photo")
parser.add_argument("--randomness", dest='randomness', type=float, default=0.0, help="Probability to use random tile")
parser.add_argument("--vectorization-factor", dest='vectorization_factor', type=float, default=1., 
    help="Downsize the image by this much before vectorizing")
parser.add_argument("--no-duplicates-radius", dest='no_duplicates_radius', type=int, default=0, help="No duplicates over a given radius")

args = parser.parse_args()

# get target image
target_image = cv2.imread(args.target)
target_height = np.size(target_image, 0)
target_width = np.size(target_image, 1)
tile_h = int(target_height / float(args.scale))
tile_w = int(target_width / float(args.scale))
aspect_ratio = target_height / float(target_width)
if args.resize != 1.0:
    target_image = cv2.resize(target_image, (int(target_width*args.resize), int(target_height*args.resize)))
    target_height = np.size(target_image, 0)
    target_width = np.size(target_image, 1)

print("=== Creating Mosaic Image ===")
print("Images=%s, target=%s, scale=%d, vectorization=%d, randomness=%.2f, faces=%s" % (
    args.codebook_dir, args.target, args.scale,
    args.vectorization_factor, args.randomness, args.detect_faces))

# normalize images
if args.normalize_dir:
    normalize_images(
        tile_h,
        tile_w,
        path=args.codebook_dir,
        output_path=args.normalize_dir,
        scale=args.scale
    )

# index all those images
tile_index, images = index_images(
    path=args.normalize_dir or args.codebook_dir,
    aspect_ratio=aspect_ratio, 
    height=tile_h,
    width=tile_w,
    vectorization_scaling_factor=args.vectorization_factor,
    caching=False,
    use_detect_faces=args.detect_faces,
    nprocesses=4
)

print("Using %d tile codebook images..." % len(images))

# transform!
mosaic, rect_starts, arr = mosaicify(
    target_image, tile_h, tile_w,
    tile_index, images,
    randomness=args.randomness,
    opacity=args.opacity,
    best_k=args.best_k,
    trim=not args.no_trim,
    no_duplicates_radius=args.no_duplicates_radius,
    verbose=True
    )

# convert to 8 bit unsigned integers
mosaic_img = mosaic.astype(np.uint8)

# save to disk
filename = os.path.basename(args.target).split('.')[0]
savepath = args.savepath % (filename, args.scale)
print("Writing mosaic image to '%s' ..." % savepath)
cv2.imwrite(savepath, mosaic_img)

