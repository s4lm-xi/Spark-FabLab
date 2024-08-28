from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Define the class labels
class_labels = {
    0: '130DC Motor-samples',
    1: '2-Channel DC Motor Driver-samples',
    2: '6812 RGB LED-samples',
    3: 'A-Buzzer-samples',
    4: 'AD Key-samples',
    5: 'Button-samples',
    6: 'Double Line Tracking-samples',
    7: 'Film pressure-samples',
    8: 'Gesture Recognition-samples',
    9: 'Hall Magnetic-samples',
    10: 'Horn-samples',
    11: 'Humidity Temperature-samples',
    12: 'IR Reciever-samples',
    13: 'LED-samples',
    14: 'One Relay-samples',
    15: 'Photo resistance-samples',
    16: 'Potenimeter-samples',
    17: 'RGB LED-samples',
    18: 'Steam sensor-samples',
    19: 'Tilt Switch-samples',
    20: 'Touch-samples',
    21: 'Traffic Light-samples',
    22: 'Ultrasonic Sensor-samples'
}

component_info = {
    '130DC Motor-samples': {
        'description': 'The 130DC Motor is an electric motor that converts electrical energy into mechanical motion. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    '2-Channel DC Motor Driver-samples': {
        'description': 'The 2-Channel DC Motor Driver is used to control the speed and direction of two DC motors. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    '6812 RGB LED-samples': {
        'description': 'The 6812 RGB LED is a light-emitting diode that can display various colors by mixing red, green, and blue light. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'A-Buzzer-samples': {
        'description': 'The Buzzer is an audio signaling device that produces sound in response to an electrical signal. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'AD Key-samples': {
        'description': 'The AD Key is an analog-to-digital converter used to read multiple key presses with a single analog pin. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Button-samples': {
        'description': 'The Button is a simple switch mechanism for controlling some aspect of a machine or a process. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Double Line Tracking-samples': {
        'description': 'The Double Line Tracking sensor detects lines on the ground and is commonly used in robotics for line-following applications. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Film pressure-samples': {
        'description': 'The Film Pressure sensor detects pressure changes by converting physical pressure into an electrical signal. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Gesture Recognition-samples': {
        'description': 'The Gesture Recognition sensor interprets hand movements as input for controlling devices. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Hall Magnetic-samples': {
        'description': 'The Hall Magnetic sensor detects magnetic fields and converts them into an electrical signal. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Horn-samples': {
        'description': 'The Horn is an electronic sound device that produces a loud sound when activated. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'Humidity Temperature-samples': {
        'description': 'The Humidity and Temperature sensor measures the environmental temperature and humidity levels. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'IR Receiver-samples': {
        'description': 'The IR Receiver detects infrared signals from a remote control or other IR sources. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'LED-samples': {
        'description': 'The LED is a light-emitting diode that emits light when an electric current passes through it. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'One Relay-samples': {
        'description': 'The Relay is an electrically operated switch that can control a high-voltage circuit with a low-voltage signal. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'Photo resistance-samples': {
        'description': 'The Photoresistor is a light-sensitive resistor that changes its resistance based on the amount of light it receives. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Potentiometer-samples': {
        'description': 'The Potentiometer is a variable resistor used to adjust levels of voltage or signal strength in a circuit. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'RGB LED-samples': {
        'description': 'The RGB LED is a light-emitting diode that can produce a wide range of colors by combining red, green, and blue light. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'Steam sensor-samples': {
        'description': 'The Steam Sensor detects the presence of steam or water vapour in the environment. This is an analog component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Tilt Switch-samples': {
        'description': 'The Tilt Switch detects the orientation or inclination of an object. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Touch-samples': {
        'description': 'The Touch Sensor detects physical touch or proximity. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    },
    'Traffic Light-samples': {
        'description': 'The Traffic Light is a visual signaling device used to control the flow of traffic. This is a digital component and serves as an output. Check out this [link] for the tutorial.'
    },
    'Ultrasonic Sensor-samples': {
        'description': 'The Ultrasonic Sensor measures the distance to an object using ultrasonic waves. This is a digital component and serves as an input. Check out this [link] for the tutorial.'
    }
}


IMG_SIZE = 224  # Assuming the input size for the model is 224x224
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the shape required by the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image using MobileNetV2's preprocessing method
    img_array = preprocess_input(img_array)
    
    return img_array

def predict_image(img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)
    
    # Predict the class probabilities
    predictions = model.predict(img_array)
    
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels.get(predicted_class[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict the class of the uploaded image
        predicted_class = predict_image(file_path)
        info = component_info.get(predicted_class)
        return jsonify({'result': info['description']})

    return jsonify({'error': 'Failed to process image'}), 500

if __name__ == "__main__":
    app.run(debug=True)
