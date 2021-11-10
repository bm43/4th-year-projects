from vp_detection import VPDetection
import cv2
import os
import matplotlib.pyplot as plt
path='C:/Users/SamSung/Desktop/uni/y4/computer_vision/FD_Item/'
l=os.listdir(path)
#print(l)
fl1=3.42*1000
fl2=3.4082*1000

# ith image i
i=7
#print(os.listdir(path))

vp=VPDetection(length_thresh=30,principal_point=None,focal_length=fl1,seed=None)

img_path=os.path.join(path,l[i])
img=cv2.imread(img_path)

print('3D space coord:\n',vp.find_vps(img))
vps_2D=vp.vps_2D
print('2D image coord:\n',vps_2D)
x=[j[0] for j in vps_2D]
y=[j[1] for j in vps_2D]

plt.scatter(x,y,s=6)

plt.imshow(img)

plt.show()
