from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from PIL import Image
import numpy as np
import io


def relu(x):
    return np.maximum(0, x)
def softmax(x):

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

w1=model["first_layer_weights"]
w2=model["second_layer_weights"]
w3=model["third_layer_weights"]
b1=model["first_layer_bias"]
b2=model["second_layer_bias"]
b3=model["third_layer_bias"]

def nural_predict(img_array):
    # Flatten the image
    img_flat = img_array.flatten().reshape((1, 784)) # Shape (784, 1)
    

    # Forward pass through the network
    z1 = np.dot(img_flat, w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2,w3) + b3
    output = softmax(z3)

    return output.tolist()


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    pixels = data["pixels"]

    # Convert to numpy array
    arr = np.array(pixels, dtype=np.uint8)

    # Reshape into image
    arr = arr.reshape((400, 400, 4))

    # Remove alpha channel
    arr = arr[:, :, :3]

    # Convert to PIL image
    img = Image.fromarray(arr)

    # Convert grayscale
    img = img.convert("L")

    # Resize to 28x28


    # Convert back to numpy
    img_array = np.array(img)
    coords = np.argwhere(img_array > 30)

    if coords.size > 0:

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img_array = img_array[
            y_min:y_max+1,
            x_min:x_max+1
        ]
    cropped = Image.fromarray(img_array)
    padding = 20

    padded = np.pad(
        img_array,
        ((padding, padding), (padding, padding)),
        mode='constant',
        constant_values=0
    )

    cropped = Image.fromarray(padded)
    cropped = cropped.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    img_array = np.array(cropped)

    # Normalize
    img_array = img_array / 255.0
    img_array = (img_array > 0.15).astype(np.float32)

    print(img_array.shape)

    # Use the actual prediction
    prediction = nural_predict(img_array)
    print("Predicted digit:", prediction)

    return jsonify({
    "probabilities": prediction[0],
    "digit": int(np.argmax(prediction[0]))
})

if __name__ == "__main__":
    app.run(debug=True)