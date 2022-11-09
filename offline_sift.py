from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from glob import glob
from Sift import Sift 

fe_sift = Sift()
if __name__== "__main__":
    dataset = glob('./static/img/*.jpeg')
    number_of_imgs = len(dataset)
    print (number_of_imgs)
    for i in range (number_of_imgs):
        img_cv2 = cv2.imread(dataset[i])
        kp, desc = Sift.gen_sift_features(img_cv2)
        feature_path = Path("./static/features_sift/feature{}.npy".format(i))
        print(feature_path)
        np.save(feature_path, desc)


