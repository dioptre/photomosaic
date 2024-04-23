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

        hsvs = []
        hsvs_files = []
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
                        img = resize_image(target_height, target_width, np_array)              
                        hsvs.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                        hsvs_files.append(f)  
                        cv2.imwrite(o, img)
                        continue
                    except:
                        image = cv2.imread(f)
                        img = resize_image(target_height, target_width, image)
                        hsvs.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                        hsvs_files.append(f)  
                        cv2.imwrite(o, img)
                elif extension=="mp4" or extension=="mov":
                    vidcap = cv2.VideoCapture(f)
                    success,image = vidcap.read()
                    img = resize_image(target_height, target_width, image)
                    hsvs.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    hsvs_files.append(f)  
                    cv2.imwrite(o, img)
                else:
                    image = cv2.imread(f)
                    img = resize_image(target_height, target_width, image)
                    hsvs.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    hsvs_files.append(f)                    
                    cv2.imwrite(o, img)
            except Exception as ex:
                 print(ex, traceback.format_exc())
                 continue
        
        # Compute histograms (focusing on the Hue channel for color images)
        histograms = [cv2.calcHist([img], [0], None, [180], [0, 180]) for img in hsvs]

        # Compare histograms using a method (e.g., correlation)
        similarity_matrix = np.zeros((len(hsvs), len(hsvs)))
        for i in range(len(hsvs)):
            for j in range(len(hsvs)):
                similarity_matrix[i, j] = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CORREL)

        # Sum similarities for each image
        np.fill_diagonal(similarity_matrix, 0)
        similarity_scores = np.sum(similarity_matrix, axis=0)
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
        top_similar = sorted_indices[0:30]
        # Find the index of the image with the highest similarity score
        #most_similar_index = np.argmax(similarity_scores)
        similar = []
        for i in top_similar:
            print(f"""Most similar file to each other:  {hsvs_files[i]}""")
            similar.append(hsvs_files[i])
        return similar
