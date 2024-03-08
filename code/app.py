from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

#loading fine-tuned model
model = load_model('vgg16_fine_tuned_epoch_100.h5')

#image-preprocessing function
def pre_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Custom decoding function for model
def decode_predictions_custom(predictions, top=1):
    decoded_predictions = []

    for prediction in predictions:
        #Retrieving index with highest probability
        top_indices = np.argsort(prediction)[-top:][::-1]
        labels = os.listdir('/Users/lisadujesiefken/Documents/ULondon/Final Project/code/images/test')
        
        #Retrieving corresponding labels
        top_predictions = [(labels[i], prediction[i]) for i in top_indices]
        decoded_predictions.append(top_predictions)

    return decoded_predictions

#route to render home
@app.route('/')
def home():
    return render_template('index.html')

#route handling img classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        file_path = 'temp_image.jpg'
        file.save(file_path)
        #preprocessing image
        img_array = pre_image(file_path)
        #img classification prediction
        prediction = model.predict(img_array)
        #retrieving top prediction
        decoded_predictions = decode_predictions_custom(prediction, top=1)
        result = decoded_predictions[0][0]

        return render_template('index.html', result=result)
    
if __name__ == '__main__':
    app.run(debug=True)