from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from glob import glob

from LBP import LocalBinaryPatterns

if __name__== "__main__":
    fe = LocalBinaryPatterns(24,8)


    dataset = glob('./static/img/*.jpeg')
    number_of_imgs = len(dataset)
    print (number_of_imgs)
    for i in range (number_of_imgs):
        img_cv2 = cv2.imread(dataset[i])
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        feature = fe.describe(gray)
        feature_path = Path("./static/features_LBP/feature{}.npy".format(i))
        print(feature_path)

        np.save(feature_path, feature)

