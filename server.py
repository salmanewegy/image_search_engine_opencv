import numpy as np
from PIL import Image
#from feature_extractor import FeatureExtractor
from ColorDescriptor import ColorDescriptor
from datetime import datetime
import flask
import cv2
from flask import Flask, request, render_template
from pathlib import Path
####################################################################################
app = Flask(__name__)
####################################################################################
#fe = FeatureExtractor()
#features = []
#img_paths = []
#for feature_path in Path("./static/feature").glob("*.npy"):
#    features.append(np.load(feature_path))
#    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
#features = np.array(features)
####################################################################################
fe_color = ColorDescriptor((8,12,3))
features = []
img_paths = []
for feature_path in Path("./static/features").glob("*.npy"):
	features.append(np.load(feature_path))
	img_paths.append("./static/img/{}.jpeg".format(feature_path.stem))
features = np.array(features)
####################################################################################

@app.route("/test")
def test():
	return "Working..."
####################################################################################
@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "GET":
		#GET request by the user, return the index page
		return render_template("index.html")

		
	#POST request by the form, upload the image and return the index page again
	if 'image' not in request.files:
		return "no file..."
	file = request.files["image"]

	#saving image on the disk
	extention = file.mimetype.split('/')[1]
	img = Image.open(file.stream)
	filepath = "./static/uploaded/{}.{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), extention)
	print(filepath)
	img.save(filepath)

	#read image again from the disk
	img2 = cv2.imread(filepath)

	#search algo
	query = fe_color.describe(img2)
	print("------------------------------------------- AFTER DESCRIBE")
	dists = np.linalg.norm(features - query, axis=1) #L2 dist
	print("------------------------------------------- AFTER NORM")
	ids = np.argsort(dists)[:30] #top 30 results
	scores = [(dists[id],img_paths[id]) for id in ids]
	print(scores)
	return render_template("index.html", query_path=filepath, scores=scores)
####################################################################################
if __name__ == "__main__":
    app.run()