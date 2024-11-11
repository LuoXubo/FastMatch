"""
    Test the feature matching methods on large size images (5472x3648).
    Test device: Orin Nano.
"""

import cv2
import time

from xfeat.xfeat import XFeat

xfeat = XFeat()

large_size = (4000, 4000)

tik = time.time()
im1 = cv2.imread('./assets/ref.png')
im2 = cv2.imread('./assets/tgt.png')
tok = time.time()
print(f'Time for reading images: {tok-tik}\n')


tik = time.time()
im1_large = cv2.resize(im1, large_size, interpolation=cv2.INTER_LINEAR)
im2_large = cv2.resize(im2, large_size, interpolation=cv2.INTER_LINEAR)
tok = time.time()
print(f'Time for resizing images: {tok-tik}\n')


mkpts_0, mkpts_1, time_det, time_match = xfeat.match_xfeat(im1_large, im2_large, top_k = 4096)
print(f'Time for feature extraction: {time_det}\nTime for feature matching: {time_match}\n')


