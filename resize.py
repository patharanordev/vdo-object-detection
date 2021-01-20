import numpy as np
import cv2
import os

target_dir = os.path.join(os.getcwd(), 'output')
output_dir = os.path.join(os.getcwd(), 'output-resized')
if not os.path.exists(output_dir): 
   os.makedirs(output_dir)

for f in os.listdir(target_dir): 
    try:
        fpath = os.path.join(target_dir, f)
        print('Image path : ', fpath)
        if os.path.exists(fpath): 
            print(' - Existing...')
            img = cv2.imread(fpath) 
            resized_image = cv2.resize(img,(28,28)) 
            cv2.imwrite(os.path.join(output_dir, f),resized_image)
    except Exception as e:
        print(str(e)) 