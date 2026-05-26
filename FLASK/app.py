from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

w1 = model["first_layer_weights"]
w2 = model["second_layer_weights"]
w3 = model["third_layer_weights"]

b1 = model["first_layer_bias"]
b2 = model["second_layer_bias"]
b3 = model["third_layer_bias"]

# ---------- ACTIVATIONS ----------
def relu(x):
    return np.maximum(0, x)

def softmax(x): 
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ---------- FORWARD ----------
def nural_predict(img_array):
    x = img_array.flatten().reshape(1, 784)

    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, w3) + b3
    output = softmax(z3)

    return output

# ---------- ROUTE ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pixels = np.array(data["pixels"], dtype=np.uint8)

    arr = pixels.reshape(400, 400, 4)
    arr = arr[:, :, :3]

    img = Image.fromarray(arr).convert("L")

    img_array = np.array(img)

    coords = np.argwhere(img_array > 30)

    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img_array = img_array[y_min:y_max+1, x_min:x_max+1]

    padded = np.pad(
        img_array,
        ((20, 20), (20, 20)),
        mode='constant'
    )

    img = Image.fromarray(padded).resize((28, 28))

    img_array = np.array(img)

    img_array = img_array / 255.0
    img_array = (img_array > 0.15).astype(np.float32)

    prediction = nural_predict(img_array)[0]

    return jsonify({
        "digit": int(np.argmax(prediction)),
        "probabilities": prediction.tolist()
    })

# ---------- HOME ----------
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
