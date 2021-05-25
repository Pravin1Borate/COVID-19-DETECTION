
#::: Import modules and packages :::
# Flask utils
import cv2
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)


# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = "model.json"
MODEL_WEIGHTS = 'weights.hdf5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''
	print(img_path)
	img = cv2.imread(img_path)
	img = cv2.resize(img, (256, 256))
	# image.append(img)
	img = img / 255
	img = img.reshape(-1, 256, 256, 3)
	predict = model.predict(img)
	predict = np.argmax(predict)
	print(predict)
	return predict


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	label_names = {0 : 'Covid-19',
                   1 : 'Normal' ,
                   2: 'Viral Pneumonia',
                   3 : 'Bacterial Pneumonia'}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)
		print(prediction)
		predicted_class = label_names[prediction]
		print('We think that is {}.'.format(predicted_class.lower()))

		return str(predicted_class).lower()

if __name__ == '__main__':
	app.run(debug = True)