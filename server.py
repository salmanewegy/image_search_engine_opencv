import numpy as np
from PIL import Image
#from feature_extractor import FeatureExtractor
from ColorDescriptor import ColorDescriptor
from datetime import datetime
import flask
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

fe_color = ColorDescriptor((8,12,3))
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpeg"))
features = np.array(features)

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
	print(file)
    #img_saving
	img = Image.open(file.stream)
	uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",".") + "_" + file.filename
	img.save(uploaded_img_path)
	#search algo
	query = fe_color.describe(img)
	dists = np.linalg.norm(features - query, axis=1) #L2 dist
	ids = np.argsort(dists)[:30] #top 30 results
	scores = [(dists[id],img_paths[id]) for id in ids]
	print(scores)
	return render_template("index.html", query_path=uploaded_img_path)
####################################################################################
if __name__ == "__main__":
    app.run()
