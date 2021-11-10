import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
path='C:/Users/SamSung/Desktop/uni/y4/computer_vision/FD_Item/'
#first image index
i=0
#second image index
j=1
l=os.listdir(path)
img_path1=os.path.join(path,l[i])
img1=cv2.imread(img_path1,0)
img_path2=os.path.join(path,l[j])
img2=cv2.imread(img_path2,0)



stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.show()
