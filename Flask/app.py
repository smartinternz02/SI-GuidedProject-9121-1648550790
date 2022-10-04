from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


global graph
#graph=tf.get_default_graph()
# Define a flask app
app = Flask(__name__)
model = load_model('natur1.h5')


print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('digital.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64,64))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        #with graph.as_default():
        preds = np.argmax(model.predict(x))
        found = ["Bird- Antbird",
                 "Bird- Peacock",
                 "Bird- Wild Turkey",
                 "Animal- Gatto",
                 "Animal- Mucca",
                 "Animal- Pecora",
                 "Flower- Rose",
                 "Flower- Sunflower",
                 "Flower- Tulip"]
        print(preds)
        text = found[preds]
        return text

if __name__ == '__main__':
    app.run(threaded = False)

