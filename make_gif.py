import glob
import os
import argparse
import shutil

import cv2
import moviepy.editor as mpy

from emosaic.utils.gif import create_gif_from_images
from emosaic.utils.misc import ensure_directory
from emosaic.utils.indexing import index_at_multiple_scales
from emosaic.utils.misc import is_running_jupyter

"""
Example:

    $ run make_gif.py \
        --target "media/pics/2018-04-01 12.00.27.jpg" \
        --savepath "media/output/%s-from-%d-to-%d.gif" \
        --codebook-dir /pics \
        --min-scale 1 \
        --max-scale 18 \
        --fps 3 \
        --detect-faces
"""

parser = argparse.ArgumentParser()

# required
parser.add_argument("--detect-faces", dest='detect_faces', action='store_true', default=False, help="If we should only include pictures with faces in them")
parser.add_argument("--codebook-dir", dest='codebook_dir', type=str, required=True, help="Source folder of images")
parser.add_argument("--target", dest='target', type=str, required=True, help="Video to mosaicify")
parser.add_argument("--min-scale", dest='min_scale', type=int, required=True, help="Start scale rendering here")
parser.add_argument("--max-scale", dest='max_scale', type=int, required=True, help="Continue rendering up until this scale")
parser.add_argument("--savepath", dest='savepath', type=str, required=True, help="Final name for the video, will add scale and base path name (use .gif extension)")
parser.add_argument("--fps", dest='fps', type=float, default=3, help="Frames per second to render") 
parser.add_argument("--fuzz", dest='fuzz', type=float, default=5, help="Fuzz factor for moviepy blur rendering") 
parser.add_argument("--vectorization-factor", dest='vectorization_factor', type=float, default=1., 
    help="Downsize the image by this much before vectorizing")

# optional / has default
parser.add_argument("--randomness", dest='randomness', type=float, default=0.0, help="Probability to use random tile")
parser.add_argument("--ascending", dest='ascending', type=int, default=1, help="1 for ascending, 0 for descending order of scales")
parser.add_argument("--height-aspect", dest='height_aspect', type=float, default=4.0, help="Height aspect")
parser.add_argument("--width-aspect", dest='width_aspect', type=float, default=3.0, help="Width aspect")

args = parser.parse_args()

# index at various scales
scale2index, scale2mosaic = index_at_multiple_scales(
    args.codebook_dir,
    min_scale=args.min_scale,
    max_scale=args.max_scale,
    height_aspect=args.height_aspect,
    width_aspect=args.width_aspect,
    vectorization_factor=args.vectorization_factor,
    precompute_target=cv2.imread(args.target),
    use_stabilization=True,
    stabilization_threshold=0.85,
    caching=True,
    use_detect_faces=args.detect_faces,
)

# create a temporary diretory to save images to
tmp_dir = '/tmp/%s-dir' % args.savepath
ensure_directory(tmp_dir)

# create mosaics at various scales, and save them to the folder above
img_paths = []
scales = range(args.min_scale, args.max_scale + 1, 1)


for i, scale in enumerate(scales):
    img_savepath = os.path.join(tmp_dir, "%08d.jpg" % i)
    mosaic = scale2mosaic[scale]
    cv2.imwrite(img_savepath, mosaic)
    img_paths.append(img_savepath)

# create the GIF!
savepath = args.savepath % (
    os.path.basename(args.target), args.min_scale, args.max_scale)
create_gif_from_images(
    img_paths, savepath, 
    fps=args.fps, fuzz=args.fuzz, 
    ascending=bool(args.ascending))

# remove temp directory 
shutil.rmtree(tmp_dir)
