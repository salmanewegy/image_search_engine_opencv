import numpy as np
from PIL import Image
#from feature_extractor import FeatureExtractor
from ColorDescriptor import ColorDescriptor
from Sift import Sift
from LBP import LocalBinaryPatterns
from datetime import datetime
import offline_sift
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
color_features = []
img_paths1 = []
for feature_path in Path("./static/features").glob("*.npy"):
	color_features.append(np.load(feature_path))
	img_paths1.append("./static/img/{}.jpeg".format(feature_path.stem))
color_features = np.array(color_features)
print("------------------------------")

sift = Sift()
sift_features = []
img_paths2 = []
for feature_path in Path("./static/features_sift").glob("*.npy"):
	sift_features.append(np.load(feature_path))
	img_paths2.append("./static/img/{}.jpeg".format(feature_path.stem))
sift_features = np.array(sift_features, dtype=object)
print("------------------------------")

fe_LBP = LocalBinaryPatterns(24,8)
LBP_features = []
img_paths3 = []
for feature_path in Path("./static/features_LBP").glob("*.npy"):
	LBP_features.append(np.load(feature_path))
	img_paths3.append("./static/img/{}.jpeg".format(feature_path.stem))
LBP_features = np.array(LBP_features)
print("------------------------------")
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
	query_img = cv2.imread(filepath)
	#search algo
	if request.form['method'] == '2':
		query = fe_color.describe(query_img)
		print("------------------------------------------- AFTER DESCRIBE")
		dists = np.linalg.norm(color_features - query, axis=1) #L2 dist
		print("------------------------------------------- AFTER NORM")
		ids = np.argsort(dists)[:30] #top 30 results
		scores = [(dists[id],img_paths1[id]) for id in ids]
		print(scores)
		return render_template("index.html", query_path=filepath, scores=scores)

	elif request.form['method'] == '3':
		query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
		query = fe_LBP.describe(query_gray)
		print("------------------------------------------- AFTER DESCRIBE")
		dists = np.linalg.norm(LBP_features - query, axis=1) #L2 dist
		print("------------------------------------------- AFTER NORM")
		ids = np.argsort(dists)[:30] #top 30 results
		scores = [(dists[id],img_paths3[id]) for id in ids]
		print(scores)
		return render_template("index.html", query_path=filepath, scores=scores)

	else: 
		key, desc = sift.gen_sift_features(query_img)
		print("------------------------------------------- AFTER DESCRIBE")
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		matches = []
		for x in sift_features:
			matches.append(bf.match(x, desc))
		print("----------------------------")
		print(matches)
		print("----------------------------")
		#good = []
		#for m in matches:
		#	if m.distance < 0.75:
		#		good.append([m])
		#matches = sorted(matches, key = lambda x:x.distance)
		#dists = np.linalg.norm(sift_features - query, axis=1) #L2 dist
		print("------------------------------------------- AFTER NORM")
		ids = np.argsort(matches)[:30] 
		scores = [(matches[id],img_paths2[id]) for id in ids]
		print(scores)
		return render_template("index.html", query_path=filepath, scores=scores)

#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#matches = bf.match(descriptors_1,descriptors_2)
#matches = sorted(matches, key = lambda x:x.distance)

#img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

####################################################################################
if __name__ == "__main__":
    app.run(debug=True)

