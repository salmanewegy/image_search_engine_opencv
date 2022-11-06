from PIL import Image
from pathlib import Path
import numpy as np

from feature_extractor import FeatureExtractor

if __name__== "__main__":
    FE = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpeg")):
        print(img_path)

        feature = FE.extract(img=Image.open(img_path))
        print(type(feature), feature.shape)

        feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        #print(feature_path)