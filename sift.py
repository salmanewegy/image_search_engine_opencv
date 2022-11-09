from pathlib import Path
import numpy as np
import cv2
from glob import glob

class Sift:
    def gen_sift_features(self, img1):
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(gray_img, None)
        return kp, desc