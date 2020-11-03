import cv2
import os
import numpy as np

train_path = '/data/train2014/'

files = []
files += [train_path + f for f in os.listdir(train_path)]
print(files)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=1024)

for file_name in files:
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    kp1, descs1 = sift.detectAndCompute(image, None)

    if len(kp1) < 1024:
        cmd = 'mv ' + file_name + ' /data/train_sub/'
        print(cmd)
        os.system(cmd)

print('done!')