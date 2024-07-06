import tensorflow as tf
import numpy as np
from flask import Flask , render_template , request
from keras.optimizers import Adam, Adamax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import joblib
import pickle


# model = tf.keras.models.load_model("models/Skin.h5", compile=False)
# model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model = joblib.load("Skin.pkl")

class_labels = ['Actinic keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented benign keratosis', 
                'Seborrheic keratosis', 'Squamous cell carcinoma', 'Vascular lesion']



app = Flask(__name__) 

@app.route('/' , methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/predict' , methods=['POST'])
def predict():

    # Get the values from the form in the request
    imagefile=request.files['imagefile']
    image_path = "./img/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path , target_size=(224,224,3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class_label = class_labels[predicted_class_index]

    # predictions = model.predict(image)
    # result = decode_predictions(predictions)
    # result = result[0][0]
    return render_template("result.html" , predictions=predicted_class_label)

if __name__ == '__main__':
    app.run()
