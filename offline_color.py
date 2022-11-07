from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from glob import glob

from ColorDescriptor import ColorDescriptor

if __name__== "__main__":
    fe = ColorDescriptor((8, 12, 3))

    dataset = glob('./static/img/*.jpeg')
    number_of_imgs = len(dataset)
    print (number_of_imgs)
    for i in range (number_of_imgs):
        img_cv2 = cv2.imread(dataset[i])
        feature = fe.describe(img_cv2)
        feature_path = Path("./static/features/feature{}.npy".format(i))
        print(feature_path)

        np.save(feature_path, feature)


#------------------------------------------------------
    #for img_path in sorted(Path("./static/img").glob("*.jpeg")):
    #    print(img_path)

        #image = cv2.imread(img_path)
        #feature = fe.describe(image)
        #img=Image.open(img_path)
        #feature = fe.describe(img)
        #print(type(feature), feature.shape)

        #feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        #print(feature_path)

        #np.save(feature_path, feature)
