import tensorflow
print('tensorflow version', tensorflow.__version__)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weight="imagenet")
        self.model = Model(input=base_model.input, outputs= base_model.get_layer("fc1").output)
        pass

    def extract(self,img):
        img = img.resize((224,224)).convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
