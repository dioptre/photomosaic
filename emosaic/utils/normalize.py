import time
import glob
from multiprocessing.pool import ThreadPool

import numpy as np
import faiss
import cv2 

import os
from PIL import Image
from pillow_heif import register_heif_opener, open_heif
import traceback

register_heif_opener()

def resize_image(target_height, target_width, img):
    border_v = 0
    border_h = 0
    if (target_height/target_width) >= (img.shape[0]/img.shape[1]):
        border_v = int((((target_height/target_width)*img.shape[1])-img.shape[0])/2)
    else:
        border_h = int((((target_width/target_height)*img.shape[0])-img.shape[1])/2)
    #img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_REFLECT, 0)
    img = cv2.resize(img, (int(target_width), int(target_height)))
    return img

def normalize_images(
          target_height,
          target_width,
          path, 
          output_path
       ):
                
        os.makedirs(output_path, exist_ok=True)

        # Get the median image dimensions
        filenames = os.listdir(path)
        
        # Iterate through each file
        for filename in filenames:
            try:
                f = os.path.join(path, filename)
                o = os.path.join(output_path, f"""{filename}.jpg""")
                filename_lower = filename.lower()
                extension = filename_lower.split(".")[-1]
                if extension == "heic":
                    try:
                        heif_file = open_heif(f, convert_hdr_to_8bit=False, bgr_mode=True)
                        np_array = np.asarray(heif_file)                        
                        cv2.imwrite(o, resize_image(target_height, target_width, np_array))
                        continue
                    except:
                        image = cv2.imread(f)
                        cv2.imwrite(o, resize_image(target_height, target_width, image))
                elif extension=="mp4" or extension=="mov":
                    vidcap = cv2.VideoCapture(f)
                    success,image = vidcap.read()
                    cv2.imwrite(o, resize_image(target_height, target_width, image))
                else:
                    image = cv2.imread(f)
                    cv2.imwrite(o, resize_image(target_height, target_width, image))
            except Exception as ex:
                 print(ex, traceback.format_exc())
                 continue
        